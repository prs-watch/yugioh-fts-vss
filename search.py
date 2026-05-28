"""FTS / VSS 検索ロジック。クエリのトークナイズと検索方式の選択を担う。"""

from functools import lru_cache

import streamlit as st
from pandas import DataFrame
from sudachipy import MorphemeList  # type: ignore
from torch import Tensor

from consts import FTS_ALLOW_TYPE, FTS_SQL, MAX_RESULTS, VSS_SQL
from resources import con, dic, model, split_mode


@lru_cache(maxsize=256)
def _encode(q: str) -> Tensor:
    return model.encode(q, normalize_embeddings=True)  # type: ignore


@lru_cache(maxsize=256)
def _tokenize(q: str) -> MorphemeList:
    return dic.tokenize(q, split_mode)


def _search_fts(tokens: MorphemeList) -> DataFrame:
    fts_query = " ".join(
        t.surface() for t in tokens if t.part_of_speech()[0] in FTS_ALLOW_TYPE
    )
    return con.sql(FTS_SQL, params={"q": fts_query, "limit": MAX_RESULTS}).to_df()


def _search_vss(q: str) -> DataFrame:
    embedding = _encode(q)
    return con.sql(
        VSS_SQL, params={"text_q": q, "embedding_q": embedding, "limit": MAX_RESULTS}
    ).to_df()


@st.cache_data(ttl=3600)
def search(q: str) -> DataFrame:
    """OOV の有無に応じて FTS と VSS を切り替えて検索する。"""
    tokens = _tokenize(q)
    # OOVが含まれる場合はFTS(BM25)、すべて既知語ならVSS(コサイン類似度)
    if any(t.is_oov() for t in tokens):
        return _search_fts(tokens)
    return _search_vss(q)
