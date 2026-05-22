"""
遊戯王OCGカード検索アプリ。

DuckDB の FTS (全文検索) と VSS (ベクトル類似度検索) を組み合わせて
遊戯王カードを日本語で検索できる Streamlit アプリ。
クエリにOOV (辞書未登録語) が含まれる場合は FTS、そうでない場合は VSS を使用する。
"""

import duckdb
import streamlit as st
import pandas as pd
from functools import lru_cache
from pathlib import Path
from sudachipy import dictionary, tokenizer
from sentence_transformers import SentenceTransformer

# --- consts ---
# --- for fts / vss ---
DATASET_URL = "https://github.com/prs-watch/yugioh-ja-dataset/releases/download/latest/dataset.parquet"
FTS_ALLOW_TYPE = ["名詞", "動詞", "形容詞"]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SQL_DIR = Path(__file__).parent / "sql"
FTS_SQL = (SQL_DIR / "fts.sql").read_text(encoding="utf-8")
VSS_SQL = (SQL_DIR / "vss.sql").read_text(encoding="utf-8")

# --- for ui ---
DISPLAY_COLUMNS = [
    "カード名",
    "カード種",
    "種族 / 魔法罠種類",
    "属性",
    "攻",
    "守",
    "レベル/ランク",
    "スケール",
    "リンク",
    "テキスト",
]
ATTRIBUTE_BADGE_MAP = {
    "光": ":yellow-badge[光]",
    "闇": ":violet-badge[闇]",
    "炎": ":red-badge[炎]",
    "水": ":blue-badge[水]",
    "風": ":green-badge[風]",
    "地": ":orange-badge[地]",
    "神": ":gray-badge[神]",
    "ー": ":gray-badge[ー]",
}
FRAME_TYPE_BADGE_MAP = {
    "魔法": ":green-badge[魔法]",
    "罠": ":red-badge[罠]",
    "通常モンスター": ":gray-badge[通常]",
    "効果モンスター": ":yellow-badge[効果]",
    "融合モンスター": ":violet-badge[融合]",
    "シンクロモンスター": ":orange-badge[シンクロ]",
    "エクシーズモンスター": ":black-badge[エクシーズ]",
    "リンクモンスター": ":blue-badge[リンク]",
    "儀式モンスター": ":green-badge[儀式]",
    "トークン": ":gray-badge[トークン]",
    "ペンデュラム効果モンスター": ":yellow-badge[P効果]",
    "通常ペンデュラムモンスター": ":gray-badge[P通常]",
    "融合ペンデュラムモンスター": ":violet-badge[P融合]",
    "シンクロペンデュラムモンスター": ":orange-badge[Pシンクロ]",
    "エクシーズペンデュラムモンスター": ":black-badge[Pエクシーズ]",
    "儀式ペンデュラムモンスター": ":green-badge[P儀式]",
}


# --- init ---
@st.cache_resource(show_spinner=False)
def __init():
    """アプリ起動時に一度だけ実行されるリソース初期化。

    DuckDB インメモリ接続を作成し、cards テーブルのロード・FTS/VSS インデックス構築、
    Sudachi トークナイザーおよび SentenceTransformer モデルの初期化を行う。

    Returns:
        tuple: (con, dic, mode, model)
            - con: duckdb.DuckDBPyConnection
            - dic: sudachipy.MorphemeList を生成するトークナイザー辞書
            - mode: Sudachi のトークナイズ分割モード (C)
            - model: SentenceTransformer エンコードモデル
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


con, dic, mode, model = __init()


@lru_cache(maxsize=256)
def _encode(q: str):
    return model.encode(q, normalize_embeddings=True)


@lru_cache(maxsize=256)
def _tokenize(q: str):
    return dic.tokenize(q, mode)


# --- search logics ---
def __fts(tokens, limit):
    """BM25 を用いた全文検索を実行する。

    Args:
        tokens: Sudachi によってトークナイズされた形態素リスト。
            名詞・動詞・形容詞のみ検索クエリに使用する。
        limit (int): 返却する最大件数。

    Returns:
        pandas.DataFrame: DISPLAY_COLUMNS のカラム構成で検索スコア降順に並んだ結果。
    """
    fts_q = " ".join(
        [t.surface() for t in tokens if t.part_of_speech()[0] in FTS_ALLOW_TYPE]
    )

    df = con.sql(FTS_SQL, params={"q": fts_q, "limit": limit}).to_df()

    return df


def __vss(q, limit):
    """コサイン類似度を用いたベクトル類似度検索を実行する。

    Args:
        q (str): 検索クエリ文字列。SentenceTransformer でエンベディングに変換される。
        limit (int): 返却する最大件数。

    Returns:
        pandas.DataFrame: DISPLAY_COLUMNS のカラム構成でスコア降順に並んだ結果。
    """
    vss_q = _encode(q)

    df = con.sql(
        VSS_SQL, params={"text_q": q, "embedding_q": vss_q, "limit": limit}
    ).to_df()

    return df


def search(q, limit=10):
    """クエリを解析し、FTS または VSS を選択してカード検索を実行する。

    クエリをトークナイズし、OOV (辞書未登録語) が1つでも含まれる場合は
    FTS、すべて既知語の場合は VSS にルーティングする。

    Args:
        q (str): 検索クエリ文字列。
        limit (int): 返却する最大件数。デフォルトは 10。

    Returns:
        pandas.DataFrame: DISPLAY_COLUMNS のカラム構成で検索スコア降順に並んだ結果。
    """
    tokens = _tokenize(q)

    is_fts = False
    for t in tokens:
        if t.is_oov():
            is_fts = True
            break

    if is_fts:
        return __fts(tokens, limit)
    return __vss(q, limit)


# --- ui ---
TITLE = "💳YUGIOH-FTS-VSS"
ICON = "💳"


def dim_bar(val):
    if val == "ー":
        return "color: #aaa;"
    return ""


st.set_option("client.showErrorDetails", False)
st.set_page_config(page_title=TITLE, page_icon="💳", layout="wide")

st.header(TITLE)

with st.form("form"):
    qcol, limitcol, buttoncol = st.columns([3, 1, 1])

    q = qcol.text_input("キーワード")
    limit = limitcol.slider("件数", 0, 20, 10)
    with buttoncol:
        st.write("")  # adjust height
        submitted = st.form_submit_button("検索")

table = st.table(pd.DataFrame(columns=DISPLAY_COLUMNS))
if submitted and q:
    df = search(q, limit)
    df["frame_type"] = df["frame_type"].map(FRAME_TYPE_BADGE_MAP)
    df["attribute"] = df["attribute"].map(ATTRIBUTE_BADGE_MAP)
    df.columns = DISPLAY_COLUMNS

    table.table(df.style.map(dim_bar))
