from pathlib import Path

# Directories
EFS_DIR = Path("/efs/shared_storage/goku")
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Mappings
EMBEDDING_DIMENSIONS = {
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    "BAAI/bge-large-en": 1024,
    "text-embedding-ada-002": 1536,
    "gte-large-fine-tuned": 1024,
}
MAX_CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4-1106-preview": 128000,
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Llama-2-13b-chat-hf": 4096,
    "meta-llama/Llama-2-70b-chat-hf": 4096,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
    "mistralai/Mistral-7B-Instruct-v0.1": 65536,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
}

# Pricing per 1M tokens
PRICING = {
    "gpt-3.5-turbo": {"prompt": 1.5, "sampled": 2},
    "gpt-4": {"prompt": 30, "sampled": 60},
    "gpt-4-1106-preview": {"prompt": 10, "sampled": 30},
    "llama-2-7b-chat-hf": {"prompt": 0.15, "sampled": 0.15},
    "llama-2-13b-chat-hf": {"prompt": 0.25, "sampled": 0.25},
    "llama-2-70b-chat-hf": {"prompt": 1, "sampled": 1},
    "codellama-34b-instruct-hf": {"prompt": 1, "sampled": 1},
    "mistral-7b-instruct-v0.1": {"prompt": 0.15, "sampled": 0.15},
    "mixtral-8x7b-instruct-v0.1": {"prompt": 0.50, "sampled": 0.50},
}
