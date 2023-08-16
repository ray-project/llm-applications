import json
from pathlib import Path

import streamlit as st

from app.config import ROOT_DIR


@st.cache_data
def load_file(dir, file_name):
    with open(Path(ROOT_DIR, dir, file_name), "r") as f:
        responses = json.load(f)
    return responses
