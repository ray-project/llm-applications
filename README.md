# LLM Applications

An end-to-end guide for scaling and serving LLM application in production. This repo currently contains one such application: a retrieval-augmented generation (RAG) app for answering questions about supplied information.

## Setup

### API keys
We'll be using [OpenAI](https://platform.openai.com/docs/models/) to access ChatGPT models like `gpt-3.5-turbo`, `gpt-4`, etc. and [Anyscale Endpoints](https://endpoints.anyscale.com/) to access OSS LLMs like `Llama-2-70b`. Be sure to create your accounts for both and have your credentials ready.

### Compute
- Start a new [Anyscale workspace on staging](https://console.anyscale-staging.com/o/anyscale-internal/workspaces) using an [`g3.8xlarge`](https://instances.vantage.sh/aws/ec2/g3.8xlarge) head node (you can also add GPU worker nodes to run the workloads faster).
- Use the [`default_cluster_env_2.6.2_py39`](https://docs.anyscale.com/reference/base-images/ray-262/py39#ray-2-6-2-py39) cluster environment.
- Use the `us-east-1` if you'd like to use the artifacts in our shared storage (source docs, vector DB dumps, etc.).

### Repository
```bash
git clone https://github.com/ray-project/llm-applications.git .  # git checkout -b goku origin/goku
git config --global user.name <GITHUB-USERNAME>
git config --global user.email <EMAIL-ADDRESS>
```

### Data
Our data is already ready at `/efs/shared_storage/goku/docs.ray.io/en/master/` (on Staging, `us-east-1`) but if you wanted to load it yourself, run this bash command (change `/desired/output/directory`, but make sure it's on the shared storage,
so that it's accessible to the workers)
```bash
export DOCS_PATH=/desired/output/directory
wget -e robots=off --recursive --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.ray.io --no-parent --accept=html \
  -P $DOCS_PATH \
  https://docs.ray.io/en/master/
```

### Environment
```bash
pip install --user -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD
pre-commit install
pre-commit autoupdate
```

### Variables
```bash
touch .env
# Add environment variables to .env
OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_API_KEY=""  # https://platform.openai.com/account/api-keys
ANYSCALE_API_BASE="https://api.endpoints.anyscale.com/v1"
ANYSCALE_API_KEY=""  # https://app.endpoints.anyscale.com/credentials
DB_CONNECTION_STRING="dbname=postgres user=postgres host=localhost password=postgres"
source .env
```

## Steps

1. Open [rag.ipynb](notebooks/rag.ipynb) to interactively go through all the concepts and run experiments.
2. Use the best configuration (in `serve.py`) from the notebook experiments to serve the LLM.
```bash
python app/main.py
```
3. Query your service.
```python
import json
import requests
data = {"query": "What is the default batch size for map_batches?"}
response = requests.post("http://127.0.0.1:8000/query", json=data)
print(response.text)
```
3. Shutdown the service
```python
from ray import serve
serve.shutdown()
```
