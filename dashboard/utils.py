import json

import streamlit as st


@st.cache_data
def load_dict(path: str):
    with open(path) as fp:
        d = json.load(fp)
    return d
