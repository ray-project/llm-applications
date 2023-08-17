import json
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from app.config import ROOT_DIR
from app.index import create_index, reset_index
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
    max_context_length: Annotated[int, typer.Option(help="max context length")] = 16384,
    system_content: Annotated[str, typer.Option(help="system content")] = "",
    assistant_content: Annotated[str, typer.Option(help="assistant content")] = "",
):
    # Reset index (if any)
    # TODO: create multiple indexes (efficient)
    reset_index()

    # Create index
    create_index(
        docs_path=docs_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Query agent
    agent = QueryAgent(
        embedding_model=embedding_model,
        llm=llm,
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
    experiment_dir = Path(ROOT_DIR, "experiments", experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(experiment_dir, "responses.json"), "w") as fp:
        json.dump(results, fp, indent=4)

    # Save config
    config = {
        "experiment_name": experiment_name,
        "docs_path": docs_path,
        "data_path": data_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "llm": llm,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
    }
    with open(Path(experiment_dir, "gen_config.json"), "w") as fp:
        json.dump(config, fp, indent=4)


@app.command()
def evaluate_responses(
    reference_loc: Annotated[str, typer.Option(help="location of reference responses")] = "",
    response_loc: Annotated[str, typer.Option(help="location of generated responses")] = "",
    evaluator: Annotated[str, typer.Option(help="name of evaluator LLM")] = "gpt-4",
    max_context_length: Annotated[int, typer.Option(help="max context length")] = 8192,
    system_content: Annotated[str, typer.Option(help="system content")] = "",
    assistant_content: Annotated[str, typer.Option(help="assistant content")] = "",
):
    # Load answers
    with open(Path(reference_loc), "r") as f:
        references = [item for item in json.load(f)]
    with open(Path(response_loc), "r") as f:
        generated = [item for item in json.load(f)]
    assert len(references) == len(generated)

    # Retrieval score
    matches = np.zeros(len(references))
    for i in range(len(references)):
        reference_source = references[i]["source"].split("#")[0]
        if not reference_source:
            matches[i] = 1
            continue
        for source in generated[i]["sources"]:
            if reference_source == source.split("#")[0]:  # sections don't have to perfectly match
                matches[i] = 1
                continue
    retrieval_score = np.mean(matches)

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
            "score": float(score),
            "reasoning": reasoning.lstrip("\n"),
        }
        results.append(result)

    # Save to file
    experiment_name = response_loc.split("/")[-2]
    experiment_dir = Path(ROOT_DIR, "experiments", experiment_name)
    evaluation = {
        "retrieval_score": retrieval_score,
        "quality_score": np.mean([item["score"] for item in results]),
        "results": results,
    }
    with open(Path(experiment_dir, "evaluation.json"), "w") as fp:
        json.dump(evaluation, fp, indent=4)

    # Save config
    config = {
        "experiment_name": experiment_name,
        "reference_loc": reference_loc,
        "response_loc": response_loc,
        "evaluator": evaluator,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
    }
    with open(Path(experiment_dir, "eval_config.json"), "w") as fp:
        json.dump(config, fp, indent=4)


if __name__ == "__main__":
    app()
