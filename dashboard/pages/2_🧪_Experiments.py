import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import ROOT_DIR
from dashboard.utils import load_dict

# Title
st.title("Experiments")

# Summary table
st.header("Summary")
evaluations_dir = Path(ROOT_DIR, "experiments", "evaluations")
evaluators = [item.name for item in evaluations_dir.iterdir() if item.is_dir()]
evaluations = {}
for evaluator in evaluators:
    evaluator_dir = Path(evaluations_dir, evaluator)
    evaluations[evaluator] = {}
    for evaluation_file in [file.name for file in evaluator_dir.iterdir() if file.is_file()]:
        evaluation_fp = Path(evaluator_dir, evaluation_file)
        if os.path.exists(evaluation_fp):
            evaluation = load_dict(path=evaluation_fp)
            evaluations[evaluator][evaluation_file.split(".json")[0]] = evaluation["quality_score"]
st.write(pd.DataFrame(evaluations))

# Responses
st.header("Responses")
responses_dir = Path(ROOT_DIR, "experiments", "responses")
response_files = [file.name for file in responses_dir.iterdir() if file.is_file()]
response_file = st.selectbox(label="Responses", options=response_files)
responses = load_dict(path=Path(responses_dir, response_file))
st.write(responses)

# Evaluations
st.header("Evaluations")
evaluator = st.selectbox(label="Evaluator", options=evaluators)
evaluator_dir = Path(evaluations_dir, evaluator)
evaluations_for_evaluator = [file.name for file in evaluator_dir.iterdir() if file.is_file()]
evaluation_file = st.selectbox(label="Evaluation", options=evaluations_for_evaluator)
evaluation = load_dict(path=Path(evaluator_dir, evaluation_file))
st.write(evaluation)
