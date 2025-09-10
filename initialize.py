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
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
import streamlit as st
# Webページ読み込み（使わないなら WEB_URL_LOAD_TARGETS を空に）
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS   # ★ Chroma → FAISS
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
# Embeddings: OpenAIをrequestsで直叩き
############################################################
def _ascii_only(s: str) -> str:
    return "".join(ch for ch in s if ch in string.printable)


class SafeOpenAIEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY が見つかりません。`.env` に設定してください。")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        data = self._post_json(url, {"model": self.model, "input": texts}, timeout=120)
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


############################################################
# CSV（社員名簿）を1ドキュメントに統合する専用ローダー
############################################################
def load_employee_csv_merged(path: str) -> List[Document]:
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

    logger.info(f"DEBUG 社員名簿CSV 読み込み成功: {os.path.basename(path)} / encoding={used_enc} / 総行数={len(rows)}")

    DEPT_KEYS = ["所属部署", "部署", "部署名", "部門", "部門名"]

    def pick(d: dict, keys: list[str]) -> str:
        for k in keys:
            if k in d and d[k]:
                return d[k]
        return ""

    bucket = defaultdict(list)
    for r in rows:
        dept = pick(r, DEPT_KEYS) or "（部署未設定）"
        bucket[dept].append(r)

    for dept, people in bucket.items():
        logger.info(f"DEBUG 部署: {dept} / {len(people)}名")

    lines = [f"社員名簿（統合ドキュメント）: {os.path.basename(path)}"]
    for dept, people in bucket.items():
        lines.append(f"\n## 部署: {dept}（{len(people)}名）")
        for p in people:
            cols = [
                ("社員ID", p.get("社員ID") or p.get("ID") or p.get("EmployeeID")),
                ("氏名", p.get("氏名") or p.get("名前")),
                ("性別", p.get("性別")),
                ("メール", p.get("メールアドレス") or p.get("Email")),
                ("役職", p.get("役職")),
                ("入社日", p.get("入社日")),
                ("保有資格", p.get("保有資格")),
                ("スキル", p.get("スキル")),
            ]
            line = " / ".join([f"{k}:{v}" for k, v in cols if v])
            if dept and dept != "（部署未設定）":
                line = f"所属部署:{dept} / " + line
            lines.append(f"- {line}")

    text = "\n".join(lines)
    return [Document(page_content=text, metadata={"source": path})]


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
        when="D", encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, "
        f"session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return

    docs_all = load_data_sources()
    logger.info(f"DEBUG 読み込んだドキュメント数: {len(docs_all)}")
    for i, d in enumerate(docs_all[:5]):
        logger.info(f"DEBUG doc[{i}].source = {d.metadata.get('source')}")

    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    _scrub_env_for_headers()
    api_key = os.getenv("OPENAI_API_KEY", "")
    embeddings = SafeOpenAIEmbeddings(
        api_key=api_key, model="text-embedding-3-small",
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE, chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    # ★ FAISS に変更（永続化あり）
    persist_dir = Path("./data/faiss_index")
    persist_dir.mkdir(parents=True, exist_ok=True)

    if (persist_dir / "index.faiss").exists() and (persist_dir / "store.pkl").exists():
        db = FAISS.load_local(
            folder_path=str(persist_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.from_documents(splitted_docs, embedding=embeddings)
        db.save_local(str(persist_dir))

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
    if file_extension == ".csv" and file_name == "社員名簿.csv":
        docs = load_employee_csv_merged(path)
        docs_all.extend(docs)
        return
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
