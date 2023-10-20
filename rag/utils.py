import os
import subprocess

import tiktoken


def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def trim(text, max_context_length):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_context_length])


def get_credentials(llm):
    if llm.startswith("gpt"):
        return os.environ["OPENAI_API_BASE"], os.environ["OPENAI_API_KEY"]
    else:
        return os.environ["ANYSCALE_API_BASE"], os.environ["ANYSCALE_API_KEY"]


def execute_bash(command):
    results = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return results


def lexical_search(index, query, chunks, k):
    query_tokens = query.lower().split()  # preprocess query
    scores = index.get_scores(query_tokens)  # get best matching (BM) scores
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]  # sort and get top k
    results = [
        {"id": chunks[i][0], "text": chunks[i][1], "source": chunks[i][2], "score": scores[i]}
        for i in indices
    ]  # top-k with scores with the text
    return results
