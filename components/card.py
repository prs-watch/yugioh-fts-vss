import streamlit as st
from pandas import DataFrame, Series

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


def render_card(card: Series) -> None:  # type: ignore[type-arg]
    with st.container(border=True):
        st.markdown(f"**{card[COL_NAME]}**")

        col_type, col_attr = st.columns(2)
        col_type.markdown(card[COL_FRAME_TYPE])
        if DB_NULL not in card[COL_ATTRIBUTE]:
            col_attr.markdown(card[COL_ATTRIBUTE])

        stats: list[str] = []
        race = card[COL_RACE]
        if race != DB_NULL:
            stats.append(race)
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
        if stats:
            st.caption("  ·  ".join(str(s) for s in stats))

        with st.expander(COL_TEXT):
            st.write(card[COL_TEXT])


def render_card_grid(df: DataFrame) -> None:
    for i in range(0, len(df), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for col, (_, card) in zip(cols, df.iloc[i : i + COLS_PER_ROW].iterrows()):
            with col:
                render_card(card)
