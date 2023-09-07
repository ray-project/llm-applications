import re


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
    reference_loc,
    response_loc,
    evaluator,
    temperature,
    max_context_length,
    system_content,
    assistant_content="",
    num_samples=None,
):
    # Set credentials
    set_credentials(llm=evaluator)

    # Load answers
    with open(Path(reference_loc), "r") as f:
        references = [item for item in json.load(f)][:num_samples]
    with open(Path(response_loc), "r") as f:
        generated = [item for item in json.load(f)["results"]][:num_samples]
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
    evaluation_fp = Path(ROOT_DIR, EXPERIMENTS_DIR, "evaluations", f"{experiment_name}_{evaluator_name}.json")
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
        "quality_score": np.mean([item["score"] for item in results if (item["score"] and item["reference_answer"])]),
        "results": results,
    }
    with open(evaluation_fp, "w") as fp:
        json.dump(evaluation, fp, indent=4)
