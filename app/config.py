import os
from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).parent.parent.absolute()


DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
DOCS_PATH = os.environ.get("DOCS_PATH")

# Credentials
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.endpoints.anyscale.com/v1")
OPENAI_API_KEY = ""  # https://app.endpoints.anyscale.com/credentials

# Response generation properties
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "llama-2-7b-gtebase")
DATA_PATH = os.environ.get("DATA_PATH", "datasets/eval-dataset-v1.jsonl")
CHUNK_SIZE = os.environ.get("CHUNK_SIZE", 300)
CHUNK_OVERLAP = os.environ.get("CHUNK_OVERLAP", 50)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "thenlper/gte-base")
LLM = os.environ.get("LLM", "meta-llama/Llama-2-7b-chat-hf")
TEMPERATURE = os.environ.get("TEMPERATURE", 0)
MAX_CONTEXT_LENGTH = os.environ.get("MAX_CONTEXT_LENGTH", 4096)

# Evaluation properties
REFERENCE_LOC = os.environ.get("REFERENCE_LOC", "experiments/responses/gpt-4-with-source.json")
RESPONSE_LOC = os.environ.get("RESPONSE_LOC", "experiments/responses/$EXPERIMENT_NAME.json")
EVALUATOR = os.environ.get("EVALUATOR", "meta-llama/Llama-2-70b-chat-hf")
EVALUATOR_TEMPERATURE = os.environ.get("EVALUATOR_TEMPERATURE", 0)
EVALUATOR_MAX_CONTEXT_LENGTH = os.environ.get("EVALUATOR_MAX_CONTEXT_LENGTH", 4096)
