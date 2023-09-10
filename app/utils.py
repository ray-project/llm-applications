import os
import subprocess

import openai


def set_credentials(llm):
    if llm.startswith("gpt"):
        openai.api_base = os.environ["OPENAI_API_BASE"]
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_base = os.environ["ANYSCALE_API_BASE"]
        openai.api_key = os.environ["ANYSCALE_API_KEY"]


def execute_bash(command):
    results = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return results
