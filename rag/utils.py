import os
import subprocess


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
