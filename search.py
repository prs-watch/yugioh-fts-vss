"""FTS / VSS 検索ロジック。クエリのトークナイズと検索方式の選択を担う。"""

from functools import lru_cache

from pandas import DataFrame
from sudachipy import MorphemeList  # type: ignore
from torch import Tensor

from consts import FTS_ALLOW_TYPE, FTS_SQL, MAX_RESULTS, VSS_SQL
from resources import con, dic, model, split_mode


@lru_cache(maxsize=256)
def encode(q: str) -> Tensor:
    """クエリ文字列を正規化済みエンベディングに変換する（LRU キャッシュ付き）。

    Args:
        q: エンベディング化するクエリ文字列。

    Returns:
        正規化済みエンベディングテンソル。
    """
    return model.encode(q, normalize_embeddings=True)  # type: ignore


@lru_cache(maxsize=256)
def tokenize(q: str) -> MorphemeList:
    """クエリ文字列を Sudachi でトークナイズする（LRU キャッシュ付き）。

    Args:
        q: トークナイズするクエリ文字列。

    Returns:
        形態素リスト。
    """
    return dic.tokenize(q, split_mode)


def fts(tokens: MorphemeList) -> DataFrame:
    """トークンリストを使って BM25 全文検索を実行する。

    Args:
        tokens: Sudachi でトークナイズ済みの形態素リスト。

    Returns:
        検索結果 DataFrame。
    """
    fts_query = " ".join(
        t.surface() for t in tokens if t.part_of_speech()[0] in FTS_ALLOW_TYPE
    )
    return con.sql(FTS_SQL, params={"q": fts_query, "limit": MAX_RESULTS}).to_df()


def vss(q: str) -> DataFrame:
    """クエリのエンベディングを使ってコサイン類似度検索を実行する。

    Args:
        q: 検索クエリ文字列。

    Returns:
        検索結果 DataFrame。
    """
    embedding = encode(q)
    return con.sql(
        VSS_SQL, params={"text_q": q, "embedding_q": embedding, "limit": MAX_RESULTS}
    ).to_df()


def search(q: str) -> DataFrame:
    """OOV の有無に応じて FTS と VSS を切り替えて検索する。

    Args:
        q: 検索クエリ文字列。

    Returns:
        検索結果 DataFrame。
    """
    tokens = tokenize(q)
    # OOVが含まれる場合はFTS(BM25)、すべて既知語ならVSS(コサイン類似度)
    if any(t.is_oov() for t in tokens):
        return fts(tokens)
    return vss(q)
