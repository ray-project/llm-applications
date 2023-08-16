import os

import pandas as pd
import streamlit as st

from dashboard.utils import load_file

# Title
st.title("Experiments")
st.write("Table of our experiment results.")
experiments = {}
dir = "evaluation"
for file_name in [file for file in os.listdir(dir) if file.endswith(".json")]:
    evaluation = load_file(dir="evaluation", file_name=file_name)
    experiments[file_name] = {
        "retrieval_score": evaluation["retrieval_score"],
        "quality_score": evaluation["quality_score"],
    }
st.write(pd.DataFrame(experiments).T)
