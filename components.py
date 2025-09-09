"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct


############################################################
# ユーティリティ（このファイル専用）
############################################################
def _format_with_page(path: str, page_or_none):
    """
    ファイルパスにページ番号 (0-based) が付いている場合は
    「（ページNo.X）」を付与して返す。
    """
    if page_or_none is None:
        return path
    try:
        # LangChain の loader は 0 始まりのことが多いので +1
        return f"{path} （ページNo.{int(page_or_none) + 1}）"
    except Exception:
        # 万一数値化できなかったらそのまま
        return path


############################################################
# 関数定義
############################################################
def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    回答モードのラジオボタンを表示
    """
    col1, _ = st.columns([100, 1])
    with col1:
        st.session_state.mode = st.radio(
            label="",
            options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
            label_visibility="collapsed",
        )


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant"):
        st.markdown(
            "こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。"
            "上記で利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。"
        )

        # 「社内文書検索」の機能説明
        st.markdown("**【「社内文書検索」を選択した場合】**")
        st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)

        # 「社内問い合わせ」の機能説明
        st.markdown("**【「社内問い合わせ」を選択した場合】**")
        st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
                continue

            # assistant 側の描画
            if message["content"]["mode"] == ct.ANSWER_MODE_1:
                # ========== 社内文書検索 ==========
                if "no_file_path_flg" not in message["content"]:
                    # メイン候補
                    st.markdown(message["content"]["main_message"])
                    icon = utils.get_source_icon(message["content"]["main_file_path"])
                    main_text = _format_with_page(
                        message["content"]["main_file_path"],
                        message["content"].get("main_page_number"),
                    )
                    st.success(main_text, icon=icon)

                    # サブ候補
                    if "sub_message" in message["content"]:
                        st.markdown(message["content"]["sub_message"])
                        for sub_choice in message["content"]["sub_choices"]:
                            icon = utils.get_source_icon(sub_choice["source"])
                            sub_text = _format_with_page(
                                sub_choice["source"], sub_choice.get("page_number")
                            )
                            st.info(sub_text, icon=icon)
                else:
                    # 候補ゼロ
                    st.markdown(message["content"]["answer"])

            else:
                # ========== 社内問い合わせ ==========
                st.markdown(message["content"]["answer"])

                if "file_info_list" in message["content"]:
                    st.divider()
                    st.markdown(f"##### {message['content']['message']}")
                    for file_info in message["content"]["file_info_list"]:
                        # file_info はすでにページ番号込みの文字列
                        icon = utils.get_source_icon(file_info)
                        st.info(file_info, icon=icon)


def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答（utils.get_llm_response の戻り値）

    Returns:
        画面再描画用に保存する辞書
    """
    # ドキュメントが見つかった場合
    if llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:
        # ===== メイン候補 =====
        main_doc = llm_response["context"][0]
        main_file_path = main_doc.metadata.get("source", "不明なファイル")
        main_page_number = main_doc.metadata.get("page", None)

        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)

        icon = utils.get_source_icon(main_file_path)
        st.success(_format_with_page(main_file_path, main_page_number), icon=icon)

        # ===== サブ候補 =====
        sub_choices = []
        duplicate_paths = {main_file_path}

        for document in llm_response["context"][1:]:
            sub_path = document.metadata.get("source", "不明なファイル")
            if sub_path in duplicate_paths:
                continue
            duplicate_paths.add(sub_path)

            sub_page = document.metadata.get("page", None)
            sub_choices.append({"source": sub_path, "page_number": sub_page})

        if sub_choices:
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)
            for sub_choice in sub_choices:
                icon = utils.get_source_icon(sub_choice["source"])
                st.info(
                    _format_with_page(sub_choice["source"], sub_choice.get("page_number")),
                    icon=icon,
                )

        # 会話ログ保存用オブジェクト
        content = {
            "mode": ct.ANSWER_MODE_1,
            "main_message": main_message,
            "main_file_path": main_file_path,
        }
        if main_page_number is not None:
            content["main_page_number"] = main_page_number
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices

    else:
        # ドキュメントが見つからなかった場合
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)
        content = {
            "mode": ct.ANSWER_MODE_1,
            "answer": ct.NO_DOC_MATCH_MESSAGE,
            "no_file_path_flg": True,
        }

    return content


def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        画面再描画用に保存する辞書
    """
    st.markdown(llm_response["answer"])

    file_info_list = []
    message = "情報源"

    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        st.divider()
        st.markdown(f"##### {message}")

        seen = set()
        for document in llm_response["context"]:
            src = document.metadata.get("source", "不明なファイル")
            if src in seen:
                continue
            seen.add(src)

            page = document.metadata.get("page", None)
            file_info = _format_with_page(src, page)

            icon = utils.get_source_icon(src)
            st.info(file_info, icon=icon)
            file_info_list.append(file_info)

    content = {
        "mode": ct.ANSWER_MODE_2,
        "answer": llm_response["answer"],
    }
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        content["message"] = message
        content["file_info_list"] = file_info_list

    return content
