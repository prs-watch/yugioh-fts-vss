WITH scored AS (
SELECT
    *,
    fts_main_cards.match_bm25(id, $q, conjunctive := 1) AS bm25_score
FROM cards
WHERE name_ja IS NOT NULL
AND text_ja IS NOT NULL
AND text_ja <> ''
),
raw AS (
SELECT
    name_ja,
    text_ja,
    frame_type,
    COALESCE(race, '✕') AS race,
    COALESCE(attribute, '✕') AS attribute,
    COALESCE(CAST(CAST(ROUND(atk, 0) AS BIGINT) AS VARCHAR), '✕') AS atk,
    COALESCE(CAST(CAST(ROUND(def, 0) AS BIGINT) AS VARCHAR), '✕') AS def,
    COALESCE(CAST(CAST(ROUND(level, 0) AS BIGINT) AS VARCHAR), '✕') AS level,
    COALESCE(CAST(CAST(ROUND(scale, 0) AS BIGINT) AS VARCHAR), '✕') AS scale,
    COALESCE(CAST(CAST(ROUND(linkval, 0) AS BIGINT) AS VARCHAR), '✕') AS linkval
FROM scored
WHERE
    name_ja LIKE '%' || $q || '%'
OR bm25_score IS NOT NULL
ORDER BY bm25_score DESC
LIMIT $limit
)
SELECT *
FROM raw
