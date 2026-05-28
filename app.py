"""
遊戯王OCGカード検索アプリ。

DuckDB の FTS (全文検索) と VSS (ベクトル類似度検索) を組み合わせて
遊戯王カードを日本語で検索できる Streamlit アプリ。
クエリに OOV (辞書未登録語) が含まれる場合は FTS (BM25)、
すべて既知語の場合は VSS (コサイン類似度) を使用する。

SQL クエリは sql/fts.sql・sql/vss.sql に外部化されており、
モジュール起動時に一度だけ読み込まれる。
"""

import streamlit as st

from components import apply_global_styles, render_card_grid
from consts import (
    ATTRIBUTE_BADGE_MAP,
    DB_COL_ATTRIBUTE,
    DB_COL_FRAME_TYPE,
    DISPLAY_COLUMNS,
    FRAME_TYPE_BADGE_MAP,
    ICON,
    LABEL_CAPTION,
    LABEL_CLEAR,
    LABEL_FILTER_ATTRIBUTE,
    LABEL_FILTER_FRAME_TYPE,
    LABEL_NO_RESULTS,
    LABEL_RESULT_COUNT,
    LABEL_SEARCH_INPUT,
    LABEL_SEARCH_PLACEHOLDER,
    LABEL_SEARCHING,
    LABEL_SUBMIT,
    TITLE,
)
from search import search

# page config
st.set_option("client.showErrorDetails", False)
st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")

apply_global_styles()

# header
st.title(TITLE)
st.caption(LABEL_CAPTION)

# session state
if "raw_results" not in st.session_state:
    st.session_state.raw_results = None
if "form_key" not in st.session_state:
    st.session_state.form_key = 0

# form
with st.form(f"form_{st.session_state.form_key}"):
    q_col, button_col = st.columns([3, 1], vertical_alignment="bottom")

    q = q_col.text_input(
        LABEL_SEARCH_INPUT,
    )
    with button_col:
        submitted = st.form_submit_button(LABEL_SUBMIT, type="primary")

if submitted and q:
    with st.spinner(LABEL_SEARCHING):
        st.session_state.raw_results = search(q)

# filter + result
if st.session_state.raw_results is None:
    st.info(LABEL_SEARCH_PLACEHOLDER)
else:
    df = st.session_state.raw_results.copy()
    df = df.set_index("id")

    # 検索結果に存在する値だけを選択肢として動的生成
    available_types = sorted(df[DB_COL_FRAME_TYPE].dropna().unique().tolist())
    available_attrs = sorted(df[DB_COL_ATTRIBUTE].dropna().unique().tolist())

    filter_col1, filter_col2, clear_col = st.columns(
        [2, 2, 1], vertical_alignment="bottom"
    )
    selected_types = filter_col1.multiselect(LABEL_FILTER_FRAME_TYPE, available_types)
    selected_attrs = filter_col2.multiselect(LABEL_FILTER_ATTRIBUTE, available_attrs)
    with clear_col:
        if st.button(LABEL_CLEAR):
            st.session_state.raw_results = None
            st.session_state.form_key += 1
            st.rerun()

    if selected_types:
        df = df[df[DB_COL_FRAME_TYPE].isin(selected_types)]
    if selected_attrs:
        df = df[df[DB_COL_ATTRIBUTE].isin(selected_attrs)]

    df[DB_COL_FRAME_TYPE] = df[DB_COL_FRAME_TYPE].map(FRAME_TYPE_BADGE_MAP)
    df[DB_COL_ATTRIBUTE] = df[DB_COL_ATTRIBUTE].map(ATTRIBUTE_BADGE_MAP)
    df.columns = DISPLAY_COLUMNS

    if df.empty:
        st.info(LABEL_NO_RESULTS)
    else:
        st.markdown(f"**{LABEL_RESULT_COUNT.format(len(df))}**")
        render_card_grid(df)
