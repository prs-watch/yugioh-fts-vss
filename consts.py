from pathlib import Path

# common
FTS_ALLOW_TYPE = ["名詞", "動詞", "形容詞"]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# pipeline
DATASET_URL = "https://github.com/prs-watch/yugioh-ja-dataset/releases/download/latest/dataset.parquet"
BATCH_SIZE = 32

# fts / vss
SQL_DIR = Path(__file__).parent / "sql"
INIT_TABLE_SQL = (SQL_DIR / "init_table.sql").read_text(encoding="utf-8")
INIT_FTS_SQL = (SQL_DIR / "init_fts.sql").read_text(encoding="utf-8")
INIT_VSS_SQL = (SQL_DIR / "init_vss.sql").read_text(encoding="utf-8")
FTS_SQL = (SQL_DIR / "fts.sql").read_text(encoding="utf-8")
VSS_SQL = (SQL_DIR / "vss.sql").read_text(encoding="utf-8")

# db raw column names (before display rename)
DB_COL_FRAME_TYPE = "frame_type"
DB_COL_ATTRIBUTE = "attribute"

# ui
TITLE = "💳YUGIOH-FTS-VSS"
ICON = "💳"

COL_NAME = "カード名"
COL_FRAME_TYPE = "カード種"
COL_RACE = "種族 / 魔法罠種類"
COL_ATTRIBUTE = "属性"
COL_ATK = "攻"
COL_DEF = "守"
COL_LEVEL = "レベル/ランク"
COL_SCALE = "スケール"
COL_LINK = "リンク"
COL_TEXT = "テキスト"

LABEL_SEARCH_INPUT = "FTS / VSS検索"
LABEL_LIMIT = "件数"
LABEL_SUBMIT = "実行"
LABEL_SEARCHING = "検索中..."
LABEL_SEARCH_PLACEHOLDER = "キーワードを入力して検索してください"
LABEL_FILTER_FRAME_TYPE = "カード種で絞り込み"
LABEL_FILTER_ATTRIBUTE = "属性で絞り込み"
LABEL_RESULT_COUNT = "{} 件"
LABEL_NO_RESULTS = "絞り込み条件に一致するカードがありません。"

DISPLAY_COLUMNS = [
    COL_NAME,
    COL_FRAME_TYPE,
    COL_RACE,
    COL_ATTRIBUTE,
    COL_ATK,
    COL_DEF,
    COL_LEVEL,
    COL_SCALE,
    COL_LINK,
    COL_TEXT,
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
