"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from typing import List
import string
import json
import requests
import csv
from collections import defaultdict

from dotenv import load_dotenv, find_dotenv
import streamlit as st
# Webページ読み込み（使わないなら WEB_URL_LOAD_TARGETS を空に）
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import constants as ct


############################################################
# 設定関連
############################################################
load_dotenv(find_dotenv(), override=True)


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False


def _scrub_env_for_headers():
    """HTTPヘッダーに載りうる環境変数をASCIIチェックし、非ASCIIなら削除"""
    candidates = [
        "OPENAI_ORGANIZATION", "OPENAI_PROJECT", "OPENAI_TITLE",
        "OPENAI_APP_NAME", "OPENAI_APP_INFO",
        "USER_AGENT", "HTTP_USER_AGENT", "X_TITLE", "TITLE",
        "GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy",
        "ALL_PROXY", "all_proxy",
    ]
    for k in candidates:
        v = os.getenv(k)
        if v and not _is_ascii(v):
            del os.environ[k]


_scrub_env_for_headers()
os.environ.setdefault("USER_AGENT", "company-inner-search-app/1.0")
os.environ.setdefault("OPENAI_NO_TELEMETRY", "1")


############################################################
# Embeddings: OpenAIをrequestsで直叩き（長文は自動分割→平均化）
############################################################
def _ascii_only(s: str) -> str:
    return "".join(ch for ch in s if ch in string.printable)


def _split_by_limit(text: str, max_chars: int) -> List[str]:
    """max_chars を超える長文を適当に段落/行単位で分割して返す"""
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for block in text.split("\n"):
        blk = block + "\n"
        if current_len + len(blk) > max_chars and current:
            parts.append("".join(current))
            current = [blk]
            current_len = len(blk)
        else:
            current.append(blk)
            current_len += len(blk)
    if current:
        parts.append("".join(current))
    return parts


def _avg(vectors: List[List[float]]) -> List[float]:
    n = len(vectors)
    if n == 0:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            out[i] += vec[i]
    return [x / n for x in out]


class SafeOpenAIEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        max_input_chars: int = 12000,   # だいたいの安全長（UTF-8文字数ベース）
        batch_size: int = 64,           # まとめて投げる数
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY が見つかりません。`.env` に設定してください。")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_input_chars = max_input_chars
        self.batch_size = batch_size

        sess = requests.Session()
        sess.trust_env = False
        sess.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "company-inner-search-app/1.0",
            "X-OpenAI-Client-User-Agent": json.dumps({"bindings_version": "1", "lang": "python"}),
        })
        self.session = sess

    def _post_json(self, url: str, payload: dict, timeout: int = 60) -> dict:
        safe_headers = {k: _ascii_only(str(v)) for k, v in self.session.headers.items()}
        resp = self.session.post(url, json=payload, headers=safe_headers,
                                 timeout=timeout, proxies={})
        resp.raise_for_status()
        return resp.json()

    def _embed_list(self, texts: List[str]) -> List[List[float]]:
        """OpenAI embeddingsに texts をバッチで投げ、埋め込みを返す"""
        url = f"{self.base_url}/embeddings"
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            data = self._post_json(url, {"model": self.model, "input": batch}, timeout=120)
            out.extend([item["embedding"] for item in data["data"]])
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 各テキストが長すぎる場合は分割して平均ベクトルを返す
        outputs: List[List[float]] = []
        for t in texts:
            parts = _split_by_limit(t, self.max_input_chars)
            if len(parts) == 1:
                outputs.extend(self._embed_list([t]))
            else:
                vecs = self._embed_list(parts)
                outputs.append(_avg(vecs))
        return outputs

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


############################################################
# CSV（社員名簿）を「部署別ドキュメントだけ」作るローダー
############################################################
def load_employee_csv_merged(path: str) -> List[Document]:
    """
    社員名簿（CSV）を「部署別ドキュメントのみ」で返す。
    （全社まとめドキュメントは作らない＝長文でのAPIエラーを避ける）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    enc_candidates = ["utf-8-sig", "utf-8", "cp932"]
    rows, used_enc = None, None
    for enc in enc_candidates:
        try:
            with open(path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                used_enc = enc
                break
        except Exception:
            continue
    if rows is None:
        raise RuntimeError(f"社員名簿CSV読み込み失敗: {path}")

    # 画面には出さず、ログにのみ残す
    logger.debug(f"社員名簿CSV 読み込み成功: {os.path.basename(path)} / encoding={used_enc} / 総行数={len(rows)}")

    DEPT_KEYS = ["所属部署", "部署", "部署名", "部門", "部門名"]

    def pick(d: dict, keys: list[str]) -> str:
        for k in keys:
            if k in d and d[k]:
                return d[k]
        return ""

    # 部署ごとにグルーピング
    bucket = defaultdict(list)
    for r in rows:
        dept = pick(r, DEPT_KEYS) or "（部署未設定）"
        bucket[dept].append(r)

    for dept, people in bucket.items():
        logger.debug(f"部署: {dept} / {len(people)}名（部署ドキュメント作成）")

    docs: List[Document] = []

    # 部署別ドキュメントのみ生成（＝短く、1ヒットで複数人を返せる）
    for dept, people in bucket.items():
        lines_dept = [f"社員名簿（部署別）: {dept}（{len(people)}名）"]
        for p in people:
            cols = [
                ("所属部署", dept),
                ("社員ID", p.get("社員ID") or p.get("ID") or p.get("EmployeeID")),
                ("氏名", p.get("氏名") or p.get("氏名（フルネーム）") or p.get("名前")),
                ("性別", p.get("性別")),
                ("メール", p.get("メールアドレス") or p.get("Email")),
                ("役職", p.get("役職") or p.get("役割")),
                ("入社日", p.get("入社日")),
                ("保有資格", p.get("保有資格")),
                ("スキル", p.get("スキルセット") or p.get("スキル")),
                ("学部", p.get("学部・学科") or p.get("学部") or p.get("学科")),
                ("卒業年月日", p.get("卒業年月日")),
            ]
            line = " / ".join([f"{k}:{v}" for k, v in cols if v])
            lines_dept.append(f"- {line}")
        text_dept = "\n".join(lines_dept)
        docs.append(
            Document(
                page_content=text_dept,
                metadata={
                    "source": path,                 # 元ファイルは社員名簿.csv
                    "department": dept,             # 部署名でクエリヒットを強化
                    "doc_kind": "employee_master_by_dept",
                },
            )
        )

    return docs


############################################################
# 関数定義
############################################################
def initialize():
    initialize_session_state()
    initialize_session_id()
    initialize_logger()
    initialize_retriever()


def initialize_logger():
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.hasHandlers():
        return
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, "
        f"session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    # ★ DEBUGも記録する
    logger.setLevel(logging.DEBUG)
    log_handler.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)


def initialize_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    if "retriever" in st.session_state:
        return

    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = load_data_sources()
    logger.debug(f"読み込んだドキュメント数: {len(docs_all)}")
    for i, d in enumerate(docs_all[:5]):
        logger.debug(f"doc[{i}].source = {d.metadata.get('source')}")

    # 文字列調整
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # Embeddings クライアント（長文は自動分割&平均）
    _scrub_env_for_headers()
    api_key = os.getenv("OPENAI_API_KEY", "")
    embeddings = SafeOpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small",
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        max_input_chars=12000,
        batch_size=64,
    )

    # === 条件付きチャンク分割（社員名簿の部署別Docは分割しない） ===
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE, chunk_overlap=ct.CHUNK_OVERLAP, separator="\n",
    )
    splitted_docs: List[Document] = []
    for d in docs_all:
        kind = d.metadata.get("doc_kind")
        if kind == "employee_master_by_dept":
            splitted_docs.append(d)  # 部署別は1ドキュメントで保持
        else:
            splitted_docs.extend(text_splitter.split_documents([d]))

    db = Chroma.from_documents(splitted_docs, embedding=embeddings)
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_TOP_K})


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


def load_data_sources():
    docs_all = []
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)
    web_docs_all = []
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        web_docs_all.extend(web_docs)
    docs_all.extend(web_docs_all)
    return docs_all


def recursive_file_check(path, docs_all):
    if os.path.isdir(path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            recursive_file_check(full_path, docs_all)
    else:
        file_load(path, docs_all)


def file_load(path, docs_all):
    file_extension = os.path.splitext(path)[1]
    file_name = os.path.basename(path)
    # ★ 社員名簿.csv だけ特別処理（部署別Docのみを生成）
    if file_extension == ".csv" and file_name == "社員名簿.csv":
        docs = load_employee_csv_merged(path)
        docs_all.extend(docs)
        return
    # それ以外は定義済みローダー
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    if type(s) is not str:
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    return s
