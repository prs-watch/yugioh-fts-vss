from pathlib import Path

# common
FTS_ALLOW_TYPE = ["名詞", "動詞", "形容詞"]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# for pipeline script
DATASET_URL = "https://github.com/prs-watch/yugioh-ja-dataset/releases/download/latest/dataset.parquet"
BATCH_SIZE = 32

# for fts / vss
SQL_DIR = Path(__file__).parent / "sql"
FTS_SQL = (SQL_DIR / "fts.sql").read_text(encoding="utf-8")
VSS_SQL = (SQL_DIR / "vss.sql").read_text(encoding="utf-8")

# for ui
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
    "通常モンスター": ":primary-badge[通常]",
    "効果モンスター": ":yellow-badge[効果]",
    "融合モンスター": ":violet-badge[融合]",
    "シンクロモンスター": ":orange-badge[シンクロ]",
    "エクシーズモンスター": ":gray-badge[エクシーズ]",
    "リンクモンスター": ":blue-badge[リンク]",
    "儀式モンスター": ":green-badge[儀式]",
    "トークン": ":gray-badge[トークン]",
    "ペンデュラム効果モンスター": ":yellow-badge[P効果]",
    "通常ペンデュラムモンスター": ":gray-badge[P通常]",
    "融合ペンデュラムモンスター": ":violet-badge[P融合]",
    "シンクロペンデュラムモンスター": ":orange-badge[Pシンクロ]",
    "エクシーズペンデュラムモンスター": ":gray-badge[Pエクシーズ]",
    "儀式ペンデュラムモンスター": ":green-badge[P儀式]",
}
