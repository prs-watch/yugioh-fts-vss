"""カード情報を表示する Streamlit コンポーネント。"""

from concurrent.futures import ThreadPoolExecutor
from typing import cast

import streamlit as st
from pandas import DataFrame, Series

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
    """カードのステータス文字列リストを構築する。

    Args:
        card: カード情報を持つ Series。

    Returns:
        種族・レベル・ATK・DEF などのステータス文字列リスト。
    """
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


def render_card(card: Series, card_id: int, image: bytes | None) -> None:  # type: ignore[type-arg]
    """1枚のカードをコンテナに描画する。

    Args:
        card: カード情報を持つ Series。
        card_id: カード ID。
        image: カード画像のバイト列。None の場合は画像を表示しない。
    """
    with st.container(border=True):
        img_col, info_col = st.columns([1, 2])

        if image is not None:
            img_col.image(image)

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


def render_card_grid(df: DataFrame) -> None:
    """カード一覧をグリッドレイアウトで描画する。

    Args:
        df: 表示するカード情報の DataFrame。
    """
    card_ids = [cast(int, idx) for idx, _ in df.iterrows()]
    with ThreadPoolExecutor(max_workers=len(card_ids)) as executor:
        images = dict(
            zip(card_ids, executor.map(get_image, [str(cid) for cid in card_ids]))
        )

    for i in range(0, len(df), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for col, (card_id, card) in zip(cols, df.iloc[i : i + COLS_PER_ROW].iterrows()):
            with col:
                render_card(card, cast(int, card_id), images.get(cast(int, card_id)))
