INSTALL fts;
LOAD fts;
PRAGMA create_fts_index('cards', 'id', 'name_ja', 'ruby', 'fts_text');
