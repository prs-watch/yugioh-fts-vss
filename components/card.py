"""カード情報を表示する Streamlit コンポーネント。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import streamlit as st
from pandas import DataFrame, Series
from streamlit.delta_generator import DeltaGenerator

from components.image import get_image
from consts import (
    COL_ATK,
    COL_ATTRIBUTE,
    COL_DEF,
    COL_FRAME_TYPE,
    COL_LEVEL,
    COL_LINK,
    COL_NAME,
    COL_RACE,
    COL_SCALE,
    COL_TEXT,
    COLS_PER_ROW,
    DB_NULL,
)


def _build_stats(card: Series) -> list[str]:  # type: ignore[type-arg]
    """種族・レベル・ATK・DEF などのステータス文字列リストを構築する。"""
    stats: list[str] = []
    if card[COL_RACE] != DB_NULL:
        stats.append(str(card[COL_RACE]))
    if card[COL_LEVEL] != DB_NULL:
        stats.append(f"Lv {card[COL_LEVEL]}")
    if card[COL_ATK] != DB_NULL:
        stats.append(f"ATK {card[COL_ATK]}")
    if card[COL_DEF] != DB_NULL:
        stats.append(f"DEF {card[COL_DEF]}")
    if card[COL_SCALE] != DB_NULL:
        stats.append(f"⚖{card[COL_SCALE]}")
    if card[COL_LINK] != DB_NULL:
        stats.append(f"LINK {card[COL_LINK]}")
    return stats


def _render_card(card: Series) -> DeltaGenerator:  # type: ignore[type-arg]
    """1枚のカードをコンテナに描画し、画像プレースホルダーを返す。"""
    with st.container(border=True):
        img_col, info_col = st.columns([1, 2])

        img_placeholder = img_col.empty()
        img_placeholder.markdown(
            '<div class="card-img-loading"></div>', unsafe_allow_html=True
        )

        with info_col:
            st.markdown(f"**{card[COL_NAME]}**")

            col_type, col_attr = st.columns(2)
            col_type.markdown(card[COL_FRAME_TYPE])
            if DB_NULL not in card[COL_ATTRIBUTE]:
                col_attr.markdown(card[COL_ATTRIBUTE])

            stats = _build_stats(card)
            if stats:
                st.caption("  ·  ".join(stats))

        with st.expander(COL_TEXT):
            st.write(card[COL_TEXT])

    return img_placeholder


def render_card_grid(df: DataFrame) -> None:
    """カード一覧をグリッドレイアウトで描画する。"""
    # 第1パス: 全カードを即時描画（画像はプレースホルダー）
    placeholders: dict[int, DeltaGenerator] = {}
    for i in range(0, len(df), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for col, (card_id, card) in zip(cols, df.iloc[i : i + COLS_PER_ROW].iterrows()):
            with col:
                placeholders[cast(int, card_id)] = _render_card(card)

    # 第2パス: 画像を並列取得し、到着順にプレースホルダーへ埋める
    card_ids = list(placeholders.keys())
    with ThreadPoolExecutor(max_workers=len(card_ids) or 1) as executor:
        futures = {executor.submit(get_image, str(cid)): cid for cid in card_ids}
        for future in as_completed(futures):
            cid = futures[future]
            image = future.result()
            if image is not None:
                placeholders[cid].image(image)
            else:
                placeholders[cid].markdown("🎴")
