"""
遊戯王OCGカード検索アプリ。

DuckDB の FTS (全文検索) と VSS (ベクトル類似度検索) を組み合わせて
遊戯王カードを日本語で検索できる Streamlit アプリ。
クエリにOOV (辞書未登録語) が含まれる場合は FTS、そうでない場合は VSS を使用する。
"""

import duckdb
import streamlit as st
from functools import lru_cache
from sudachipy import dictionary, tokenizer
from sentence_transformers import SentenceTransformer

# --- consts ---
DATASET_URL = "https://github.com/prs-watch/yugioh-ja-dataset/releases/download/latest/dataset.parquet"
FTS_ALLOW_TYPE = ["名詞", "動詞", "形容詞"]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DISPLAY_COLUMNS = [
    "カード名",
    "テキスト",
    "カード種",
    "種族 / 魔法罠種類",
    "属性",
    "攻",
    "守",
    "レベル/ランク",
    "スケール",
    "リンク",
    "検索スコア",
]


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
    con = duckdb.connect()
    con.execute("CREATE TABLE cards AS SELECT * FROM read_parquet('cards.parquet');")

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
    CREATE INDEX IF NOT EXISTS cards_embeddings_idx ON cards USING HNSW (embeddings) WITH (metric = 'cosine');
    """)

    dic = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    model = SentenceTransformer(MODEL_NAME)

    return con, dic, mode, model


con, dic, mode, model = __init()


@lru_cache(maxsize=256)
def _encode(q: str):
    return model.encode(q)


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

    df = con.sql(
        """
    WITH raw AS (
        SELECT
            name_ja,
            text_ja,
            frame_type,
            COALESCE(race, '✕') AS race,
            COALESCE(attribute, '✕') AS attribute,
            COALESCE(CAST(CAST(ROUND(atk, 0) AS BIGINT) AS VARCHAR), '✕') AS atk,
            COALESCE(CAST(CAST(ROUND(def, 0) AS BIGINT) AS VARCHAR), '✕') AS def,
            COALESCE(CAST(CAST(ROUND(level, 0) AS BIGINT) AS VARCHAR), '✕') AS level,
            COALESCE(CAST(CAST(ROUND(scale, 0) AS BIGINT) AS VARCHAR), '✕') AS scale,
            COALESCE(CAST(CAST(ROUND(linkval, 0) AS BIGINT) AS VARCHAR), '✕') AS linkval,
            fts_main_cards.match_bm25(id, $q, conjunctive := 1) AS score
        FROM
            cards
        WHERE
            name_ja IS NOT NULL
        ORDER BY
            score DESC
        LIMIT
            $limit
    )
    SELECT * REPLACE (score / NULLIF(MAX(score) OVER (), 0) AS score)
    FROM raw
    ORDER BY score DESC
    ;
    """,
        params={"q": fts_q, "limit": limit},
    ).to_df()

    df.columns = DISPLAY_COLUMNS

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
        """
    WITH raw AS (
        SELECT
            name_ja,
            text_ja,
            frame_type,
            COALESCE(race, '✕') AS race,
            COALESCE(attribute, '✕') AS attribute,
            COALESCE(CAST(CAST(ROUND(atk, 0) AS BIGINT) AS VARCHAR), '✕') AS atk,
            COALESCE(CAST(CAST(ROUND(def, 0) AS BIGINT) AS VARCHAR), '✕') AS def,
            COALESCE(CAST(CAST(ROUND(level, 0) AS BIGINT) AS VARCHAR), '✕') AS level,
            COALESCE(CAST(CAST(ROUND(scale, 0) AS BIGINT) AS VARCHAR), '✕') AS scale,
            COALESCE(CAST(CAST(ROUND(linkval, 0) AS BIGINT) AS VARCHAR), '✕') AS linkval,
            GREATEST(0, array_cosine_similarity(embeddings, CAST($q AS FLOAT[384]))) AS score
        FROM
            cards
        WHERE
            name_ja IS NOT NULL
        ORDER BY
            embeddings <-> CAST($q AS FLOAT[384])
        LIMIT
            $limit
    )
    SELECT * REPLACE (score / NULLIF(MAX(score) OVER (), 0) AS score)
    FROM raw
    ORDER BY score DESC
    ;
    """,
        params={"q": vss_q, "limit": limit},
    ).to_df()

    df.columns = DISPLAY_COLUMNS

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
st.set_option("client.showErrorDetails", False)
st.set_page_config(page_title="YUGIOH-FTS-VSS", page_icon="💳", layout="wide")

form = st.container(border=True)
qcol, limitcol, buttoncol = form.columns([3, 1, 1])

q = qcol.text_input("キーワード")
limit = limitcol.slider("件数", 0, 50, 10)
with buttoncol:
    st.write("")  # adjust height
    button = st.button("検索")

result = st.table(search("", limit=0))

if button:
    df = search(q, limit)
    result.table(df, width="content")
