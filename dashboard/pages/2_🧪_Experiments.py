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
experiments_dir = Path(ROOT_DIR, "experiments")
experiments = {}
for experiment_name in [file for file in os.listdir(experiments_dir)]:
    evaluation_fp = f"{experiments_dir}/{experiment_name}/evaluation.json"
    if os.path.exists(evaluation_fp):
        evaluation = load_dict(path=evaluation_fp)
        experiments[experiment_name] = {
            "retrieval_score": evaluation["retrieval_score"],
            "quality_score": evaluation["quality_score"],
        }
st.write(pd.DataFrame(experiments).T)

# Load
experiment_name = st.selectbox(label="Experiments", options=list(experiments.keys()))
experiment_dir = Path(ROOT_DIR, "experiments", experiment_name)
gen_config = load_dict(path=f"{experiments_dir}/{experiment_name}/gen_config.json")
responses = load_dict(path=f"{experiments_dir}/{experiment_name}/responses.json")
eval_config = load_dict(path=f"{experiments_dir}/{experiment_name}/eval_config.json")
evaluation = load_dict(path=f"{experiments_dir}/{experiment_name}/evaluation.json")

# Quality score
st.header("Quality")
sorted_results = sorted(evaluation["results"], key=lambda x: x["score"])
st.write(f"Quality score: {evaluation['quality_score']}")
st.write("Sorted samples:")
st.write(sorted_results)

# Retrieval errors
st.header("Retrieval")
st.write(f"Retrieval score: {evaluation['retrieval_score']}")
st.write("Samples that didn't retrieve the best source:")
eval_config = load_dict(path=f"{experiments_dir}/{experiment_name}/eval_config.json")
references = load_dict(path=eval_config["reference_loc"])
errors = []
for i in range(len(references)):
    reference_source = references[i]["source"].split("#")[0]
    if not reference_source:
        continue
    generated_sources = [source.split("#")[0] for source in responses[i]["sources"]]
    if reference_source not in generated_sources:
        errors.append(
            {
                "index": i,
                "question": references[i]["question"],
                "reference_source": reference_source,
                "generated_sources": generated_sources,
            }
        )
st.write(errors)

# Configs
st.header("Configurations")
st.subheader("Generation")
st.write(gen_config)
st.subheader("Evaluation")
st.write(eval_config)
