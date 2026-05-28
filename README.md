# yugioh-fts-vss

DuckDB の FTS (全文検索) と VSS (ベクトル類似度検索) を組み合わせた遊戯王OCGカード日本語検索アプリ。

## 概要

クエリの内容に応じて検索方式を自動切替する。

| 条件 | 検索方式 | アルゴリズム |
|------|----------|-------------|
| クエリに辞書未登録語 (OOV) が含まれる | FTS (全文検索) | BM25 |
| クエリがすべて既知語 | VSS (ベクトル類似度検索) | コサイン類似度 |

## 開発者向け情報

### 使用技術

- **[DuckDB](https://duckdb.org/)** — インメモリ分析DB。`fts` / `vss` 拡張で全文検索・HNSW ベクトル検索を実現
- **[Sudachi](https://github.com/WorksApplications/SudachiPy)** — 日本語形態素解析。OOV 判定とクエリトークナイズに使用
- **[SentenceTransformers](https://www.sbert.net/)** — `paraphrase-multilingual-MiniLM-L12-v2` でクエリをベクトルに変換
- **[Streamlit](https://streamlit.io/)** — Web UI

### 動作環境

- Python 3.14 以上

### SQL ファイル構成

`sql/` ディレクトリに全クエリを外部化している。

| ファイル | 役割 |
|---------|------|
| `init_table.sql` | カードテーブルの作成と `embeddings` カラムの型定義 |
| `init_fts.sql` | FTS インデックスの初期化 |
| `init_vss.sql` | VSS (HNSW) インデックスの初期化 |
| `fts.sql` | BM25 全文検索クエリ |
| `vss.sql` | コサイン類似度ベクトル検索クエリ |

### セットアップ


```bash
# 依存パッケージのインストール
uv sync

# Pyrightチェックを実行する.githooks/pre-commitを有効化
git config --local core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

### 起動

```bash
uv run streamlit run app.py
```

ブラウザで `http://localhost:8501` を開く。

### カード画像

カード画像は外部 API から署名付き URL で取得する。

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `IMAGE_BASE_URL` | `http://localhost:8787` | カード画像 API のベース URL |
| `IMAGE_SIGNING_SECRET` | `dev-secret` | HMAC-SHA256 署名の秘密鍵 |

本番環境では `.streamlit/secrets.toml` に設定する。
画像は 24 時間キャッシュされ、グリッド表示時に並列取得する。

### 検索項目

| カラム | 説明 |
|--------|------|
| カード名 | 日本語カード名 |
| カード種 | モンスター / 魔法 / 罠など |
| 種族 / 魔法罠種類 | 種族または魔法・罠の種類 |
| 属性 | 光・闇・水・炎・地・風・神 |
| 攻 | 攻撃力 |
| 守 | 守備力 |
| レベル/ランク | レベルまたはランク |
| スケール | ペンデュラムスケール |
| リンク | リンク数 |
| テキスト | カードテキスト |

### カードデータ更新

カードデータ `cards.parquet` は GitHub Actions により毎週月曜 JST 9:00 に自動更新される。

`update_cards.py` が以下を実行する。

1. [yugioh-ja-dataset](https://github.com/prs-watch/yugioh-ja-dataset) から最新データセットを取得
2. Sudachi で形態素解析し、FTS 用テキスト (`fts_text`) を生成
3. SentenceTransformer でエンベディング (`embeddings`) を生成
4. `tmp/cards.parquet` として保存
5. ルートの `cards.parquet` と置き換えてコミット・プッシュ

手動実行する場合は GitHub Actions の `workflow_dispatch` から実行できる。