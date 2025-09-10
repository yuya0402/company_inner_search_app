"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
import os
import traceback
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# （自作）ユーティリティ＆初期化
import utils
from initialize import initialize

# （自作）表示用コンポーネント（回答の表示に使用）
import components as cn

# （自作）定数（アプリ名・文言・RAG設定など）
import constants as ct


############################################################
# 2. 事前セットアップ（.env 読み込み / 環境のASCII化）
############################################################
# .env を確実に読み込む（日本語パスでもOK & 既存値は上書き）
load_dotenv(find_dotenv(), override=True)

def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False

# http系のライブラリがヘッダー化する可能性のある値をASCIIに限定
for k in ["USER_AGENT", "OPENAI_ORGANIZATION", "OPENAI_PROJECT", "OPENAI_APP_NAME", "OPENAI_APP_INFO"]:
    v = os.getenv(k)
    if v and not _is_ascii(v):
        del os.environ[k]

os.environ.setdefault("USER_AGENT", "company-inner-search-app/1.0")

# 作業フォルダ（アップロードや一時出力）が無ければ作成
for p in ["uploads", "outputs", "tmp", "logs"]:
    Path(p).mkdir(parents=True, exist_ok=True)


############################################################
# 3. 画面の基本設定 & CSS
############################################################
st.set_page_config(page_title=ct.APP_NAME, page_icon=":mag:", layout="wide")
st.markdown(
    """
    <style>
      /* タイトル中央寄せ */
      .center-title h1 { text-align: center; margin-top: .2rem; }
      /* サイドのカード風 */
      .sidecard { background: #eef2ff; padding: 12px 14px; border-radius: 12px; margin: 10px 0; }
      /* 本文の横幅をやや絞る（全体に適用） */
      .block-container { max-width: 980px; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


############################################################
# 4. ホーム画面を描画
############################################################
def render_home() -> str:
    # ---- 左サイド ----
    with st.sidebar:
        st.markdown("### 利用目的")
        mode = st.radio(
            label="",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
            index=0,
        )
        st.session_state.mode = mode  # 後続で使う

        # 利用説明カード
        st.markdown('<div class="sidecard"><b>「社内文書検索」</b> を選択した場合</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidecard">入力内容と関連性が高い社内文書のありかを検索できます。</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidecard"><b>【入力例】</b><br>社員の育成方針に関するMTGの議事録</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidecard"><b>「社内問い合わせ」</b> を選択した場合</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidecard">質問・要望に対して、社内文書の情報をもとに回答します。</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidecard"><b>【入力例】</b><br>人事部に所属している従業員情報を一覧化して</div>', unsafe_allow_html=True)

    # ---- メイン（中央） ----
    st.markdown(f"<div class='center-title'><h1>{ct.APP_NAME}</h1></div>", unsafe_allow_html=True)

    st.info(
        "こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。"
        "サイドバーで利用目的を選び、画面下部のチャット欄からメッセージを送信してください。"
    )
    st.warning("具体的に入力したほうが期待通りの回答を得やすいです。")

    # 下部のチャット入力
    return st.chat_input(placeholder=ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 5. 初期化（RAGの準備など）
############################################################
logger = logging.getLogger(ct.LOGGER_NAME)
logging.basicConfig(level=logging.INFO)

try:
    initialize()  # 初期化（retriever作成など）
except Exception as e:
    logger.exception("初期化処理に失敗")
    st.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{ct.COMMON_ERROR_MESSAGE}", icon=ct.ERROR_ICON)
    st.code(traceback.format_exc())
    st.stop()

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 6. 画面描画 & 入力受付
############################################################
user_text = render_home()  # ホーム画面を描画し、入力を受け取る

# 過去ログの表示
try:
    cn.display_conversation_log()
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(ct.CONVERSATION_LOG_ERROR_MESSAGE, icon=ct.ERROR_ICON)

############################################################
# 7. チャット送信時の処理
############################################################
if user_text:
    # 7-1. ユーザーメッセージ表示 & ログ
    logger.info({"message": user_text, "application_mode": st.session_state.mode})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 7-2. 回答生成（RAG → LLM）
    with st.spinner(ct.SPINNER_TEXT):
        try:
            llm_response = utils.get_llm_response(user_text)
        except Exception as e:
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            st.error(ct.GET_LLM_RESPONSE_ERROR_MESSAGE, icon=ct.ERROR_ICON)
            st.code(traceback.format_exc())
            st.stop()

    # 7-3. 回答表示
    with st.chat_message("assistant"):
        try:
            if st.session_state.mode == ct.ANSWER_MODE_1:
                content = cn.display_search_llm_response(llm_response)
            else:
                content = cn.display_contact_llm_response(llm_response)

            logger.info({"message": content, "application_mode": st.session_state.mode})
        except Exception as e:
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            st.error(ct.DISP_ANSWER_ERROR_MESSAGE, icon=ct.ERROR_ICON)
            st.code(traceback.format_exc())
            st.stop()

    # 7-4. 会話ログに追記
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": content})
