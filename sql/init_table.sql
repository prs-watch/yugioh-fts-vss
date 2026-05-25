CREATE TABLE cards AS SELECT * FROM read_parquet('cards.parquet');

ALTER TABLE cards ADD COLUMN embeddings_tmp FLOAT[384];
UPDATE cards SET embeddings_tmp = CAST(embeddings AS FLOAT[384]);
ALTER TABLE cards DROP COLUMN embeddings;
ALTER TABLE cards RENAME COLUMN embeddings_tmp TO embeddings;
