from functools import lru_cache

from pandas import DataFrame
from sudachipy import MorphemeList  # type: ignore
from torch import Tensor

from consts import FTS_ALLOW_TYPE, FTS_SQL, VSS_SQL
from resources import con, dic, model, split_mode


@lru_cache(maxsize=256)
def encode(q: str) -> Tensor:
    return model.encode(q, normalize_embeddings=True)  # type: ignore


@lru_cache(maxsize=256)
def tokenize(q: str) -> MorphemeList:
    return dic.tokenize(q, split_mode)


def fts(tokens: MorphemeList, limit: int) -> DataFrame:
    fts_query = " ".join(
        t.surface() for t in tokens if t.part_of_speech()[0] in FTS_ALLOW_TYPE
    )
    return con.sql(FTS_SQL, params={"q": fts_query, "limit": limit}).to_df()


def vss(q: str, limit: int) -> DataFrame:
    embedding = encode(q)
    return con.sql(
        VSS_SQL, params={"text_q": q, "embedding_q": embedding, "limit": limit}
    ).to_df()


def search(q: str, limit: int = 10) -> DataFrame:
    tokens = tokenize(q)
    # OOVが含まれる場合はFTS(BM25)、すべて既知語ならVSS(コサイン類似度)
    if any(t.is_oov() for t in tokens):
        return fts(tokens, limit)
    return vss(q, limit)
