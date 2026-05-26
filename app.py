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

from components import render_card_grid
from consts import (
    ATTRIBUTE_BADGE_MAP,
    DB_COL_ATTRIBUTE,
    DB_COL_FRAME_TYPE,
    DISPLAY_COLUMNS,
    FRAME_TYPE_BADGE_MAP,
    ICON,
    LABEL_CLEAR,
    LABEL_FILTER_ATTRIBUTE,
    LABEL_FILTER_FRAME_TYPE,
    LABEL_LIMIT,
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

st.markdown(
    """<style>
    [data-testid='StyledFullScreenButton'],
    button[title='View fullscreen'],
    button[aria-label='Fullscreen'],
    .stImage button { display: none !important; }
    [data-testid='stFormSubmitButton'] button {
        background-color: #1E88E5 !important;
        border-color: #1E88E5 !important;
        color: white !important;
    }
    [data-testid='stButton'] button,
    [data-testid='baseButton-secondary'],
    button[kind='secondary'] {
        background-color: #adb5bd !important;
        border-color: #adb5bd !important;
        color: white !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# header
st.title(TITLE)

# session state
if "raw_results" not in st.session_state:
    st.session_state.raw_results = None
if "form_key" not in st.session_state:
    st.session_state.form_key = 0

# form
with st.form(f"form_{st.session_state.form_key}"):
    q_col, limit_col, button_col = st.columns([3, 1, 1])

    q = q_col.text_input(LABEL_SEARCH_INPUT)
    limit = limit_col.slider(LABEL_LIMIT, 0, 20, 10)
    with button_col:
        st.write("")  # adjust height
        submitted = st.form_submit_button(LABEL_SUBMIT, type="primary")

if submitted and q:
    with st.spinner(LABEL_SEARCHING):
        st.session_state.raw_results = search(q, limit)

# filter + result
if st.session_state.raw_results is None:
    st.info(LABEL_SEARCH_PLACEHOLDER)
else:
    df = st.session_state.raw_results.copy()
    df = df.set_index("id")

    # 検索結果に存在する値だけを選択肢として動的生成
    available_types = sorted(df[DB_COL_FRAME_TYPE].dropna().unique().tolist())
    available_attrs = sorted(df[DB_COL_ATTRIBUTE].dropna().unique().tolist())

    filter_col1, filter_col2, clear_col = st.columns([2, 2, 1])
    selected_types = filter_col1.multiselect(LABEL_FILTER_FRAME_TYPE, available_types)
    selected_attrs = filter_col2.multiselect(LABEL_FILTER_ATTRIBUTE, available_attrs)
    with clear_col:
        st.write("")
        st.write("")  # adjust height
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
        st.caption(LABEL_RESULT_COUNT.format(len(df)))
        render_card_grid(df)
