# yugioh-fts-vss

DuckDB の FTS (全文検索) と VSS (ベクトル類似度検索) を組み合わせた遊戯王OCGカード日本語検索アプリ。

## 概要

クエリの内容に応じて検索方式を自動切替する。

| 条件 | 検索方式 | アルゴリズム |
|------|----------|-------------|
| クエリに辞書未登録語 (OOV) が含まれる | FTS (全文検索) | BM25 |
| クエリがすべて既知語 | VSS (ベクトル類似度検索) | コサイン類似度 |

## 使用技術

- **[DuckDB](https://duckdb.org/)** — インメモリ分析DB。`fts` / `vss` 拡張で全文検索・HNSW ベクトル検索を実現
- **[Sudachi](https://github.com/WorksApplications/SudachiPy)** — 日本語形態素解析。OOV 判定とクエリトークナイズに使用
- **[SentenceTransformers](https://www.sbert.net/)** — `paraphrase-multilingual-MiniLM-L12-v2` でクエリをベクトルに変換
- **[Streamlit](https://streamlit.io/)** — Web UI

## セットアップ

```bash
# 依存パッケージのインストール
uv sync
```

## 起動

```bash
uv run streamlit run main.py
```

ブラウザで `http://localhost:8501` を開く。

## 検索項目

| カラム | 説明 |
|--------|------|
| カード名 | 日本語カード名 |
| テキスト | カードテキスト |
| カード種 | モンスター / 魔法 / 罠など |
| 種族 / 魔法罠種類 | 種族または魔法・罠の種類 |
| 属性 | 光・闇・水・炎・地・風・神 |
| 攻 | 攻撃力 |
| 守 | 守備力 |
| レベル/ランク | レベルまたはランク |
| スケール | ペンデュラムスケール |
| リンク | リンク数 |
| 検索スコア | BM25 スコアまたはコサイン類似度 |
