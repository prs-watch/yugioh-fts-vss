"""
遊戯王OCGカード検索アプリ。

DuckDB の FTS (全文検索) と VSS (ベクトル類似度検索) を組み合わせて
遊戯王カードを日本語で検索できる Streamlit アプリ。
クエリに OOV (辞書未登録語) が含まれる場合は FTS (BM25)、
すべて既知語の場合は VSS (コサイン類似度) を使用する。

SQL クエリは sql/fts.sql・sql/vss.sql に外部化されており、
モジュール起動時に一度だけ読み込まれる。
"""

from typing import Any, Tuple

import duckdb
from pandas import DataFrame
import streamlit as st
from functools import lru_cache
from sudachipy import MorphemeList, SplitMode, dictionary, tokenizer  # type: ignore
from sentence_transformers import SentenceTransformer
from torch import Tensor

# --- consts ---
from consts import (
    FTS_ALLOW_TYPE,
    MODEL_NAME,
    FTS_SQL,
    VSS_SQL,
    DISPLAY_COLUMNS,
    ATTRIBUTE_BADGE_MAP,
    FRAME_TYPE_BADGE_MAP,
)


# --- init ---
@st.cache_resource(show_spinner=False)
def init() -> Tuple[
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
    # init data
    con = duckdb.connect()
    con.execute("CREATE TABLE cards AS SELECT * FROM read_parquet('cards.parquet');")

    # fts / vss prepare
    con.execute("""
    ALTER TABLE cards ADD COLUMN embeddings_tmp FLOAT[384];
    UPDATE cards SET embeddings_tmp = CAST(embeddings AS FLOAT[384]);
    ALTER TABLE cards DROP COLUMN embeddings;
    ALTER TABLE cards RENAME COLUMN embeddings_tmp TO embeddings;
    """)

    con.execute("""
    INSTALL fts; LOAD fts;
    PRAGMA create_fts_index('cards', 'id', 'name_ja', 'fts_text');
    
    INSTALL vss; LOAD vss;
    CREATE INDEX IF NOT EXISTS cards_embeddings_idx ON cards USING HNSW (embeddings) WITH (metric = 'cosine', ef_search = 200);
    """)

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
    fts_q = " ".join([
        t.surface() for t in tokens if t.part_of_speech()[0] in FTS_ALLOW_TYPE
    ])

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

    is_fts = False
    for t in tokens:
        if t.is_oov():
            is_fts = True
            break

    if is_fts:
        return fts(tokens, limit)
    return vss(q, limit)


# --- ui ---
TITLE = "💳YUGIOH-FTS-VSS"
ICON = "💳"


def dim_bar(val: Any) -> str:
    """Pandas Styler 用セル単位スタイル関数。

    欠損値プレースホルダ "ー" のセルをグレーアウトする。
    `df.style.map(dim_bar)` で使用する。

    Args:
        val: セルの値。

    Returns:
        "ー" の場合は `"color: #aaa;"`、それ以外は `""`。
    """
    if val == "ー":
        return "color: #aaa;"
    return ""


# page config
st.set_option("client.showErrorDetails", False)
st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")

# header
st.title(TITLE)

# form
with st.form("form"):
    q_col, limit_col, button_col = st.columns([3, 1, 1])

    q = q_col.text_input("FTS / VSS検索")
    limit = limit_col.slider("件数", 0, 20, 10)
    with button_col:
        st.write("")  # adjust height
        submitted = st.form_submit_button("実行")

# result
if submitted and q:
    df = search(q, limit)

    # add badge
    df["frame_type"] = df["frame_type"].map(FRAME_TYPE_BADGE_MAP)
    df["attribute"] = df["attribute"].map(ATTRIBUTE_BADGE_MAP)

    # set display label
    df.columns = DISPLAY_COLUMNS

    st.table(df.style.map(dim_bar))
