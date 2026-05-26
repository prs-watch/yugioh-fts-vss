from pathlib import Path

import duckdb
from sudachipy import SplitMode, dictionary, tokenizer  # type: ignore
from sentence_transformers import SentenceTransformer

# --- consts ---
from consts import DATASET_URL, FTS_ALLOW_TYPE, MODEL_NAME, BATCH_SIZE

ROOT_DIR = Path(__file__).parent.resolve()
PARQUET_PATH = ROOT_DIR / "tmp" / "cards.parquet"


def init() -> tuple[tokenizer.Tokenizer, SplitMode, SentenceTransformer]:
    """Sudachi トークナイザーと SentenceTransformer モデルを初期化する。"""
    dic = dictionary.Dictionary().create()
    split_mode = tokenizer.Tokenizer.SplitMode.C
    model = SentenceTransformer(MODEL_NAME)

    return dic, split_mode, model


def build_fts_text(text: object, dic: tokenizer.Tokenizer, split_mode: SplitMode) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(
        m.surface()
        for m in dic.tokenize(text, split_mode)
        if m.part_of_speech()[0] in FTS_ALLOW_TYPE
    )


def main(dic: tokenizer.Tokenizer, split_mode: SplitMode, model: SentenceTransformer) -> None:
    """データセットを取得し、FTS用テキストとエンベディングを付与して parquet に保存する。"""
    df = duckdb.read_parquet(DATASET_URL).to_df()

    # fts
    df["fts_text"] = df["text_ja"].apply(  # type: ignore[misc]
        lambda x: build_fts_text(x, dic, split_mode)  # type: ignore[misc]
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
    dic, split_mode, model = init()
    main(dic, split_mode, model)
