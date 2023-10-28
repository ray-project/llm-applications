import json
import re
from pathlib import Path

import numpy as np
from IPython.display import JSON, clear_output, display
from tqdm import tqdm

from rag.generate import generate_response
from rag.utils import get_num_tokens, trim


def get_retrieval_score(references, generated):
    matches = np.zeros(len(references))
    for i in range(len(references)):
        reference_source = references[i]["source"].split("#")[0]
        if not reference_source:
            matches[i] = 1
            continue
        for source in generated[i]["sources"]:
            # sections don't have to perfectly match
            if reference_source == source.split("#")[0]:
                matches[i] = 1
                continue
    retrieval_score = np.mean(matches)
    return retrieval_score


def extract_from_response(response):
    # Define regular expressions for extracting values
    answer_pattern = r'"answer"\s*:\s*"([^"]*)"'
    score_pattern = r'"score"\s*:\s*([0-9]+)'
    reasoning_pattern = r'"reasoning"\s*:\s*"([^"]*)"'

    # Extract values using regular expressions
    answer_match = re.search(answer_pattern, response)
    score_match = re.search(score_pattern, response)
    reasoning_match = re.search(reasoning_pattern, response)

    # Convert
    if answer_match and score_match and reasoning_match:
        answer = answer_match.group(1)
        score = float(score_match.group(1))
        reasoning = reasoning_match.group(1)
        return answer, score, reasoning

    return "", "", ""


def evaluate_responses(
    experiment_name,
    evaluator,
    temperature,
    max_context_length,
    system_content,
    assistant_content,
    experiments_dir,
    references_fp,
    responses_fp,
    num_samples=None,
):
    # Load answers
    with open(Path(references_fp), "r") as f:
        references = [item for item in json.load(f)][:num_samples]
    with open(Path(responses_fp), "r") as f:
        generated = [item for item in json.load(f)["results"]][:num_samples]
    assert len(references) == len(generated)

    # Quality score
    results = []
    context_length = max_context_length - get_num_tokens(system_content + assistant_content)
    for ref, gen in tqdm(zip(references, generated), total=len(references)):
        assert ref["question"] == gen["question"]
        user_content = trim(
            str(
                {
                    "question": gen["question"],
                    "generated_answer": gen["answer"],
                    "reference_answer": ref["answer"],
                }
            ),
            context_length,
        )

        # Generate response
        response = generate_response(
            llm=evaluator,
            temperature=temperature,
            system_content=system_content,
            assistant_content=assistant_content,
            user_content=user_content,
        )

        # Extract from response
        score, reasoning = response.split("\n", 1) if "\n" in response else (0, "")
        result = {
            "question": gen["question"],
            "generated_answer": gen["answer"],
            "reference_answer": ref["answer"],
            "score": float(score),
            "reasoning": reasoning.lstrip("\n"),
            "sources": gen["sources"],
        }
        results.append(result)
        clear_output(wait=True)
        display(JSON(json.dumps(result, indent=2)))

    # Save to file
    evaluator_name = evaluator.split("/")[-1].lower()
    evaluation_fp = Path(
        experiments_dir, "evaluations", f"{experiment_name}_{evaluator_name}.json"
    )
    evaluation_fp.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": experiment_name,
        "evaluator": evaluator,
        "temperature": temperature,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
        "experiments_dir": str(experiments_dir),
        "references_fp": str(references_fp),
        "responses_fp": str(responses_fp),
    }
    evaluation = {
        "config": config,
        "retrieval_score": get_retrieval_score(references, generated),
        "quality_score": np.mean(
            [item["score"] for item in results if (item["score"] and item["reference_answer"])]
        ),
        "results": results,
    }
    with open(evaluation_fp, "w") as fp:
        json.dump(evaluation, fp, indent=4)
