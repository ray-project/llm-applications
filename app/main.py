import json
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from app.config import ROOT_DIR
from app.query import QueryAgent, generate_response

app = typer.Typer()


@app.command()
def generate_responses(
    experiment_name: Annotated[str, typer.Option(help="experiment name")] = "",
    docs_path: Annotated[str, typer.Option(help="location of docs to index")] = "",
    data_path: Annotated[str, typer.Option(help="location of dataset with questions")] = "",
    chunk_size: Annotated[int, typer.Option(help="chunk size")] = 300,
    chunk_overlap: Annotated[int, typer.Option(help="chunk overlap")] = 50,
    embedding_model: Annotated[str, typer.Option(help="embedder")] = "thenlper/gte-base",
    llm: Annotated[str, typer.Option(help="name of LLM")] = "gpt-3.5-turbo-16k",
    temperature: Annotated[float, typer.Option(help="temperature")] = 0,
    max_context_length: Annotated[int, typer.Option(help="max context length")] = 16384,
    system_content: Annotated[str, typer.Option(help="system content")] = "",
    assistant_content: Annotated[str, typer.Option(help="assistant content")] = "",
):
    # Query agent
    agent = QueryAgent(
        embedding_model=embedding_model,
        llm=llm,
        temperature=temperature,
        max_context_length=max_context_length,
        system_content=system_content,
        assistant_content=assistant_content,
    )

    # Generate responses
    results = []
    with open(Path(ROOT_DIR, data_path), "r") as f:
        questions = [json.loads(item)["question"] for item in list(f)]
    for query in tqdm(questions):
        result = agent.get_response(query=query)
        results.append(result)

    # Save to file
    responses_fp = Path(ROOT_DIR, "experiments", "responses", f"{experiment_name}.json")
    responses_fp.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": experiment_name,
        "docs_path": docs_path,
        "data_path": data_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "llm": llm,
        "temperature": temperature,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
    }
    responses = {
        "config": config,
        "results": results,
    }
    with open(responses_fp, "w") as fp:
        json.dump(responses, fp, indent=4)


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


def clean_score(score):
    """For LLMs that aren't so great at following instructions."""
    score = score.lower()
    if len(score) == 1:
        return float(score)
    elif score.startswith("score: "):
        score = score.split("score: ")[-1]
        if "/" in score:
            return float(score.split("/")[0])
        return float(score)
    else:
        print(score)
        return 0.0


@app.command()
def evaluate_responses(
    experiment_name: Annotated[str, typer.Option(help="experiment name")] = "",
    reference_loc: Annotated[str, typer.Option(help="location of reference responses")] = "",
    response_loc: Annotated[str, typer.Option(help="location of generated responses")] = "",
    evaluator: Annotated[str, typer.Option(help="name of evaluator LLM")] = "gpt-4",
    temperature: Annotated[float, typer.Option(help="temperature")] = 0,
    max_context_length: Annotated[int, typer.Option(help="max context length")] = 8192,
    system_content: Annotated[str, typer.Option(help="system content")] = "",
    assistant_content: Annotated[str, typer.Option(help="assistant content")] = "",
):
    # Load answers
    with open(Path(reference_loc), "r") as f:
        references = [item for item in json.load(f)]
    with open(Path(response_loc), "r") as f:
        generated = [item for item in json.load(f)["results"]]
    assert len(references) == len(generated)

    # Quality score
    results = []
    context_length = max_context_length - len(system_content + assistant_content)
    for ref, gen in tqdm(zip(references, generated), total=len(references)):
        assert ref["question"] == gen["question"]
        user_content = str(
            {
                "question": gen["question"],
                "generated_answer": gen["answer"],
                "reference_answer": ref["answer"],
            }
        )[:context_length]

        # Generate response
        response = generate_response(
            llm=evaluator,
            temperature=temperature,
            system_content=system_content,
            assistant_content=assistant_content,
            user_content=user_content,
        )

        # Extract from response
        score, reasoning = response.split("\n", 1)

        # Store result
        result = {
            "question": gen["question"],
            "generated_answer": gen["answer"],
            "reference_answer": ref["answer"],
            "score": clean_score(score=score),
            "reasoning": reasoning.lstrip("\n"),
            "sources": gen["sources"],
        }
        results.append(result)

    # Save to file
    evaluation_fp = Path(
        ROOT_DIR,
        "experiments",
        "evaluations",
        evaluator.split("/")[-1].lower(),
        f"{experiment_name}.json",
    )
    evaluation_fp.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": experiment_name,
        "reference_loc": reference_loc,
        "response_loc": response_loc,
        "evaluator": evaluator,
        "temperature": temperature,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
    }
    evaluation = {
        "config": config,
        "retrieval_score": get_retrieval_score(references, generated),
        "quality_score": np.mean([item["score"] for item in results]),
        "results": results,
    }
    with open(evaluation_fp, "w") as fp:
        json.dump(evaluation, fp, indent=4)


if __name__ == "__main__":
    app()
