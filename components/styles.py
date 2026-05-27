"""アプリ全体に適用するグローバル CSS スタイル。"""

import streamlit as st

_CSS = """
[data-testid='StyledFullScreenButton'],
button[title='View fullscreen'],
button[aria-label='Fullscreen'],
.stImage button { display: none !important; }

[data-testid='stButton'] button,
[data-testid='baseButton-secondary'],
button[kind='secondary'] {
    background-color: #adb5bd !important;
    border-color: #adb5bd !important;
    color: white !important;
}
"""


def apply_global_styles() -> None:
    """グローバル CSS を Streamlit に注入する。"""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
