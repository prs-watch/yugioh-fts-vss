"""
遊戯王OCGカード検索アプリ。

DuckDB の FTS (全文検索) と VSS (ベクトル類似度検索) を組み合わせて
遊戯王カードを日本語で検索できる Streamlit アプリ。
クエリに OOV (辞書未登録語) が含まれる場合は FTS (BM25)、
すべて既知語の場合は VSS (コサイン類似度) を使用する。

SQL クエリは sql/fts.sql・sql/vss.sql に外部化されており、
モジュール起動時に一度だけ読み込まれる。
"""

from functools import lru_cache

import duckdb
from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer
from sudachipy import MorphemeList, SplitMode, dictionary, tokenizer  # type: ignore
from torch import Tensor
import streamlit as st

# --- consts ---
from consts import (
    FTS_ALLOW_TYPE,
    MODEL_NAME,
    INIT_TABLE_SQL,
    INIT_FTS_SQL,
    INIT_VSS_SQL,
    FTS_SQL,
    VSS_SQL,
    TITLE,
    ICON,
    COL_NAME,
    COL_FRAME_TYPE,
    COL_RACE,
    COL_ATTRIBUTE,
    COL_ATK,
    COL_DEF,
    COL_LEVEL,
    COL_SCALE,
    COL_LINK,
    COL_TEXT,
    DISPLAY_COLUMNS,
    LABEL_SEARCH_INPUT,
    LABEL_LIMIT,
    LABEL_SUBMIT,
    LABEL_SEARCHING,
    LABEL_SEARCH_PLACEHOLDER,
    LABEL_FILTER_FRAME_TYPE,
    LABEL_FILTER_ATTRIBUTE,
    LABEL_RESULT_COUNT,
    LABEL_NO_RESULTS,
    ATTRIBUTE_BADGE_MAP,
    FRAME_TYPE_BADGE_MAP,
)


# --- init ---
@st.cache_resource(show_spinner=False)
def init() -> tuple[
    duckdb.DuckDBPyConnection,
    tokenizer.Tokenizer,
    SplitMode,
    SentenceTransformer,
]:
    """アプリ起動時に一度だけ実行されるリソース初期化。

    DuckDB インメモリ接続を作成し、cards テーブルのロード・FTS/VSS インデックス構築、
    Sudachi トークナイザーおよび SentenceTransformer モデルの初期化を行う。

    Returns:
        Tuple[DuckDBPyConnection, Tokenizer, SplitMode, SentenceTransformer]:
            - con: DuckDB インメモリ接続。FTS/VSS インデックス構築済み。
            - dic: Sudachi トークナイザーインスタンス。
            - mode: Sudachi のトークナイズ分割モード (C モード)。
            - model: SentenceTransformer エンコードモデル。
    """
    con = duckdb.connect()
    con.execute(INIT_TABLE_SQL)
    con.execute(INIT_FTS_SQL)
    con.execute(INIT_VSS_SQL)

    dic = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    model = SentenceTransformer(MODEL_NAME)

    return con, dic, mode, model


# execute init
con, dic, mode, model = init()


# --- search logics ---
@lru_cache(maxsize=256)
def encode(q: str) -> Tensor:
    """クエリ文字列を正規化済みエンベディングに変換する (LRUキャッシュ付き)。

    Args:
        q: エンコード対象のクエリ文字列。

    Returns:
        正規化済みエンベディングテンソル (shape: [384])。
    """
    return model.encode(q, normalize_embeddings=True)  # type: ignore


@lru_cache(maxsize=256)
def tokenize(q: str) -> MorphemeList:
    """クエリ文字列を Sudachi C モードでトークナイズする (LRUキャッシュ付き)。

    Args:
        q: トークナイズ対象のクエリ文字列。

    Returns:
        Sudachi が解析した形態素リスト。
    """
    return dic.tokenize(q, mode)


def fts(tokens: MorphemeList, limit: int) -> DataFrame:
    """BM25 を用いた全文検索を実行する。

    Args:
        tokens: Sudachi によってトークナイズされた形態素リスト。
            名詞・動詞・形容詞 (FTS_ALLOW_TYPE) のみ検索クエリに使用する。
        limit: 返却する最大件数。

    Returns:
        name_ja, frame_type, race, attribute, atk, def,
        level, scale, linkval, text_ja の順で BM25 スコア降順に並んだ DataFrame。
    """
    fts_q = " ".join(
        [t.surface() for t in tokens if t.part_of_speech()[0] in FTS_ALLOW_TYPE]
    )

    df = con.sql(FTS_SQL, params={"q": fts_q, "limit": limit}).to_df()

    return df


def vss(q: str, limit: int) -> DataFrame:
    """コサイン類似度を用いたベクトル類似度検索を実行する。

    Args:
        q: 検索クエリ文字列。SentenceTransformer でエンベディングに変換される。
        limit: 返却する最大件数。

    Returns:
        name_ja, frame_type, race, attribute, atk, def,
        level, scale, linkval, text_ja の順でコサイン類似度スコア降順に並んだ DataFrame。
    """
    vss_q = encode(q)

    df = con.sql(
        VSS_SQL, params={"text_q": q, "embedding_q": vss_q, "limit": limit}
    ).to_df()

    return df


def search(q: str, limit: int = 10) -> DataFrame:
    """クエリを解析し、FTS または VSS を選択してカード検索を実行する。

    クエリをトークナイズし、OOV (辞書未登録語) が1つでも含まれる場合は FTS、
    すべて既知語の場合は VSS にルーティングする。
    返却カラムは DB カラム名のままで、表示用へのリネームは呼び出し元が行う。

    Args:
        q: 検索クエリ文字列。
        limit: 返却する最大件数。デフォルトは 10。

    Returns:
        name_ja, frame_type, race, attribute, atk, def,
        level, scale, linkval, text_ja の順で検索スコア降順に並んだ DataFrame。
    """
    tokens = tokenize(q)
    has_oov = any(t.is_oov() for t in tokens)

    if has_oov:
        return fts(tokens, limit)
    return vss(q, limit)


# --- ui ---
COLS_PER_ROW = 3


def render_card(card: Series) -> None:
    with st.container(border=True):
        st.markdown(f"**{card[COL_NAME]}**")

        col_type, col_attr = st.columns(2)
        col_type.markdown(card[COL_FRAME_TYPE])
        if "ー" not in card[COL_ATTRIBUTE]:
            col_attr.markdown(card[COL_ATTRIBUTE])

        stats: list[str] = []
        race = card[COL_RACE]
        if race != "ー":
            stats.append(race)
        if card[COL_LEVEL] != "ー":
            stats.append(f"Lv {card[COL_LEVEL]}")
        if card[COL_ATK] != "ー":
            stats.append(f"ATK {card[COL_ATK]}")
        if card[COL_DEF] != "ー":
            stats.append(f"DEF {card[COL_DEF]}")
        if card[COL_SCALE] != "ー":
            stats.append(f"⚖{card[COL_SCALE]}")
        if card[COL_LINK] != "ー":
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


# page config
st.set_option("client.showErrorDetails", False)
st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")

# header
st.title(TITLE)

# session state
if "raw_results" not in st.session_state:
    st.session_state.raw_results = None

# form
with st.form("form"):
    q_col, limit_col, button_col = st.columns([3, 1, 1])

    q = q_col.text_input(LABEL_SEARCH_INPUT)
    limit = limit_col.slider(LABEL_LIMIT, 0, 20, 10)
    with button_col:
        st.write("")  # adjust height
        submitted = st.form_submit_button(LABEL_SUBMIT)

if submitted and q:
    with st.spinner(LABEL_SEARCHING):
        st.session_state.raw_results = search(q, limit)

# filter + result
if st.session_state.raw_results is None:
    st.info(LABEL_SEARCH_PLACEHOLDER)
else:
    df = st.session_state.raw_results.copy()

    # 検索結果に存在する値だけを選択肢として動的生成
    available_types = sorted(df["frame_type"].dropna().unique().tolist())
    available_attrs = sorted(df["attribute"].dropna().unique().tolist())

    filter_col1, filter_col2 = st.columns(2)
    selected_types = filter_col1.multiselect(LABEL_FILTER_FRAME_TYPE, available_types)
    selected_attrs = filter_col2.multiselect(LABEL_FILTER_ATTRIBUTE, available_attrs)

    if selected_types:
        df = df[df["frame_type"].isin(selected_types)]
    if selected_attrs:
        df = df[df["attribute"].isin(selected_attrs)]

    df["frame_type"] = df["frame_type"].map(FRAME_TYPE_BADGE_MAP)
    df["attribute"] = df["attribute"].map(ATTRIBUTE_BADGE_MAP)
    df.columns = DISPLAY_COLUMNS

    if df.empty:
        st.info(LABEL_NO_RESULTS)
    else:
        st.caption(LABEL_RESULT_COUNT.format(len(df)))
        render_card_grid(df)
