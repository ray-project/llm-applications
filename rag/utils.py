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
