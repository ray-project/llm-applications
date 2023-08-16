import os
from pathlib import Path

import streamlit as st

from app.config import ROOT_DIR
from dashboard.utils import load_file


def get_overlapping_files(dir1, dir2, file_format):
    files1 = set(file for file in os.listdir(dir1) if file.endswith(file_format))
    files2 = set(file for file in os.listdir(dir2) if file.endswith(file_format))
    overlapping_files = files1.intersection(files2)
    return list(overlapping_files)


# Title
st.title("Evaluation")
st.write("Quantity the quality of our LLM's performance.")

# Load responses
st.header("Responses")
references_file_name = st.text_input(
    label="Reference responses", value="gpt4-with-context.json", disabled=True
)
generated_file_options = get_overlapping_files(
    dir1=Path(ROOT_DIR, "responses"),
    dir2=Path(ROOT_DIR, "evaluation"),
    file_format=".json",
)
generated_file_name = st.selectbox(label="Generated responses", options=generated_file_options)
references = load_file(dir="responses", file_name=references_file_name)
generated = load_file(dir="responses", file_name=generated_file_name)
st.write("Reference responses", references)
st.write("Generated responses", generated)

# Scores
st.header("Scores")
evaluation = load_file(dir="evaluation", file_name=generated_file_name)
st.write(evaluation)

# Quality inspection
st.header("Sorted samples")
sorted_results = sorted(evaluation["results"], key=lambda x: x["score"])
st.write(sorted_results)

# Retrieval errors
st.header("Retrieval errors")
st.write("How often did we not retrieve the best resource?")
errors = []
for i in range(len(references)):
    reference_source = references[i]["source"].split("#")[0]
    if not reference_source:
        continue
    generated_sources = [source.split("#")[0] for source in generated[i]["sources"]]
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
