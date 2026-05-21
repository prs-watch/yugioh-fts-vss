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

    df = con.sql(
        """
    WITH scored AS (
    SELECT
        *,
        fts_main_cards.match_bm25(id, $q, conjunctive := 1) AS bm25_score
    FROM cards
    WHERE name_ja IS NOT NULL
      AND text_ja IS NOT NULL
    ),
    raw AS (
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
        CASE
            WHEN name_ja = $q THEN 1000
            WHEN name_ja LIKE '%' || $q || '%' THEN 500
            ELSE COALESCE(bm25_score, 0)
        END AS score
    FROM scored
    WHERE
        name_ja LIKE '%' || $q || '%'
        OR bm25_score IS NOT NULL
    ORDER BY score DESC
    LIMIT $limit
    )
    SELECT * REPLACE (score / NULLIF(MAX(score) OVER (), 0) AS score)
    FROM raw
    ORDER BY score DESC;
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
        WITH name_hits AS (
        SELECT
            *,
            CASE
                WHEN name_ja = $text_q THEN 1.0
                WHEN name_ja LIKE '%' || $text_q || '%' THEN 0.95
                ELSE 0
            END AS score,
            'name' AS search_type
        FROM cards
        WHERE name_ja IS NOT NULL
        AND text_ja IS NOT NULL
        AND name_ja LIKE '%' || $text_q || '%'
        LIMIT $limit
        ),
        vss_hits AS (
        SELECT
            *,
            GREATEST(0, array_cosine_similarity(embeddings, CAST($embedding_q AS FLOAT[384]))) AS score,
            'vss' AS search_type
        FROM cards
        WHERE name_ja IS NOT NULL
        AND text_ja IS NOT NULL
        ORDER BY embeddings <-> CAST($embedding_q AS FLOAT[384])
        LIMIT $limit
        ),
        merged AS (
        SELECT * FROM name_hits
        UNION ALL
        SELECT * FROM vss_hits
        ),
        dedup AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY name_ja
                ORDER BY score DESC
            ) AS rn
        FROM merged
        )
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
            score
        FROM dedup
        WHERE rn = 1
        ORDER BY score DESC
        LIMIT $limit;
        """,
        params={"text_q": q, "embedding_q": vss_q, "limit": limit},
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

st.header("💳YUGIOH-FTS-VSS")

with st.form("form"):
    qcol, limitcol, buttoncol = st.columns([3, 1, 1])

    q = qcol.text_input("キーワード")
    limit = limitcol.slider("件数", 0, 20, 10)
    with buttoncol:
        st.write("")  # adjust height
        submitted = st.form_submit_button("検索")

if submitted and q:
    df = search(q, limit)
    st.table(df)
