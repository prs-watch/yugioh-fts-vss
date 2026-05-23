from pathlib import Path
from typing import Tuple

import duckdb
from sudachipy import SplitMode, dictionary, tokenizer  # type: ignore
from sentence_transformers import SentenceTransformer

# --- consts ---
from consts import DATASET_URL, FTS_ALLOW_TYPE, MODEL_NAME, BATCH_SIZE

ROOT_DIR = Path(__file__).parent.resolve()
PARQUET_PATH = ROOT_DIR / "tmp" / "cards.parquet"


def init() -> Tuple[tokenizer.Tokenizer, SplitMode, SentenceTransformer]:
    """Sudachi トークナイザーと SentenceTransformer モデルを初期化する。"""
    dic = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    model = SentenceTransformer(MODEL_NAME)

    return dic, mode, model


def main(dic: tokenizer.Tokenizer, mode: SplitMode, model: SentenceTransformer):
    """データセットを取得し、FTS用テキストとエンベディングを付与して parquet に保存する。"""
    df = duckdb.read_parquet(DATASET_URL).to_df()

    # fts
    df["fts_text"] = df["text_ja"].apply(
        lambda x: (  # type: ignore
            " ".join([
                m.surface()
                for m in dic.tokenize(x, mode)
                if m.part_of_speech()[0] in FTS_ALLOW_TYPE
            ])
            if isinstance(x, str)
            else ""
        )
    )

    # vss
    embeddings = model.encode(  # type: ignore
        df["text_ja"].to_list(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    df["embeddings"] = embeddings.tolist()  # type: ignore

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)


if __name__ == "__main__":
    dic, mode, model = init()
    main(dic, mode, model)
