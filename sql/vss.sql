WITH name_hits AS (
SELECT
    *,
    CASE
        WHEN name_ja = $text_q THEN 1.0
        WHEN name_ja LIKE '%' || $text_q || '%' THEN 0.95
        ELSE 0
    END AS score,
    'name' AS search_type
FROM cards
WHERE name_ja IS NOT NULL
AND text_ja IS NOT NULL
AND text_ja <> ''
AND name_ja LIKE '%' || $text_q || '%'
LIMIT $limit
),
vss_hits AS (
SELECT
    *,
    GREATEST(0, array_cosine_similarity(embeddings, CAST($embedding_q AS FLOAT[384]))) AS score,
    'vss' AS search_type
FROM cards
WHERE name_ja IS NOT NULL
AND text_ja IS NOT NULL
AND text_ja <> ''
ORDER BY embeddings <-> CAST($embedding_q AS FLOAT[384])
LIMIT $limit
),
merged AS (
SELECT * FROM name_hits
UNION ALL
SELECT * FROM vss_hits
),
dedup AS (
SELECT
    *,
    ROW_NUMBER() OVER (
        PARTITION BY name_ja
        ORDER BY score DESC
    ) AS rn
FROM merged
)
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
FROM dedup
WHERE rn = 1
AND CASE
    WHEN search_type = 'name' THEN score >= 0.85
    WHEN search_type = 'vss' THEN score >= 0.65
    ELSE FALSE
END
ORDER BY score DESC
LIMIT $limit
