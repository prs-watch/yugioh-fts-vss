import duckdb
import streamlit as st
from sentence_transformers import SentenceTransformer
from sudachipy import SplitMode, dictionary, tokenizer  # type: ignore

from consts import INIT_FTS_SQL, INIT_TABLE_SQL, INIT_VSS_SQL, MODEL_NAME


@st.cache_resource(show_spinner=False)
def init() -> tuple[
    duckdb.DuckDBPyConnection,
    tokenizer.Tokenizer,
    SplitMode,
    SentenceTransformer,
]:
    con = duckdb.connect()
    con.execute(INIT_TABLE_SQL)
    con.execute(INIT_FTS_SQL)
    con.execute(INIT_VSS_SQL)

    dic = dictionary.Dictionary().create()
    split_mode = tokenizer.Tokenizer.SplitMode.C
    model = SentenceTransformer(MODEL_NAME)

    return con, dic, split_mode, model


con, dic, split_mode, model = init()
