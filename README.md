# LLM Applications

An end-to-end guide for scaling and serving LLM application in production.

This repo currently contains one such application: a retrieval-augmented generation (RAG)
app for answering questions about supplied information. By default, the app uses
the [Ray documentation](https://docs.ray.io/en/master/) as the source of information.
This app first [indexes the documentation in a vector database](./app/index.py)
and then uses an LLM to generate responses for questions that got augmented with
relevant info retrieved from the index.

## Setup

### Compute

Start a new [Anyscale workspace on staging](https://console.anyscale-staging.com/o/anyscale-internal/workspaces)
using an `g3.8xlarge` head node. The current version of the app requires at least one GPU.
Creating the index will be faster if you also use  some GPU worker nodes.
Use the [`default_cluster_env_2.6.2_py39`](https://docs.anyscale.com/reference/base-images/ray-262/py39#ray-2-6-2-py39) cluster environment.

### Repository

```bash
git clone https://github.com/ray-project/llm-applications.git .
git config --global user.email <YOUR_EMAIL>
git config --global user.name <YOUR_NAME>
```

### Data

Our data is already ready at `/efs/shared_storage/pcmoritz/docs.ray.io/en/master/` (on Staging) but if you wanted to load it yourself, run this bash command (change `/desired/output/directory`, but make sure it's on the shared storage,
so that it's accessible to the workers)
```bash
wget -e robots=off --recursive --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.ray.io --no-parent --accept=html \
  -P /desired/output/directory \
  https://docs.ray.io/en/master/
```

### Environment
```bash
pip install --user -r requirements.txt
pre-commit install
pre-commit autoupdate
export PYTHONPATH=$PYTHONPATH:$PWD
export OPENAI_API_KEY=""  # https://platform.openai.com/account/api-keys
export DB_CONNECTION_STRING="dbname=postgres user=postgres host=localhost password=postgres"
# Change to /desired/output/directory/docs.ray.io/en/master/ if you downloaded the data yourself
export DOCS_PATH="/efs/shared_storage/pcmoritz/docs.ray.io/en/master/"
```

### Vector DB

```bash
bash setup-pgvector.sh
sudo -u postgres psql -f migrations/initial.sql
python app/index.py create-index --docs-path $DOCS_PATH
```

### Query

Just a sample and uses the current index that's been created.

```python
import json
from app.query import QueryAgent
query = "What is the default batch size for map_batches?"
system_content = "Your job is to answer a question using the additional context provided."
agent = QueryAgent(
    embedding_model="thenlper/gte-base",
    llm="gpt-3.5-turbo-16k",
    max_context_length=16384,
    system_content=system_content,
)
result = agent.get_response(query=query)
print(json.dumps(result, indent=2))
```

### Experiments

#### Generate responses

```bash
export EXPERIMENT_NAME="gpt3.5-16k-gtebase"
export DOCS_PATH="/efs/shared_storage/pcmoritz/docs.ray.io/en/master/"
export DATA_PATH="datasets/eval-dataset-v1.jsonl"
export CHUNK_SIZE=300
export CHUNK_OVERLAP=50
export EMBEDDING_MODEL="thenlper/gte-base"
export LLM="gpt-3.5-turbo-16k"
export MAX_CONTEXT_LENGTH=16384
```

```bash
python app/main.py generate-responses \
    --experiment-name $EXPERIMENT_NAME \
    --docs-path $DOCS_PATH \
    --data-path $DATA_PATH \
    --chunk-size $CHUNK_SIZE \
    --chunk-overlap $CHUNK_OVERLAP \
    --embedding-model $EMBEDDING_MODEL \
    --llm $LLM \
    --max-context-length $MAX_CONTEXT_LENGTH \
    --system-content "Answer the {query} using the additional {context} provided."
```

#### Evaluate responses

```bash
export REFERENCE_LOC="datasets/gpt4-with-source.json"
export RESPONSE_LOC="experiments/$EXPERIMENT_NAME/responses.json"
export EVALUATOR="gpt-4"
export EVALUATOR_MAX_CONTEXT_LENGTH=8192
```

```bash
python app/main.py evaluate-responses \
    --reference-loc $REFERENCE_LOC \
    --response-loc $RESPONSE_LOC \
    --evaluator $EVALUATOR \
    --max-context-length $EVALUATOR_MAX_CONTEXT_LENGTH \
    --system-content """
    Your job is to rate the quality of our generated answer {generated_answer}
    given a query {query} and a reference answer {reference_answer}.
    Your score has to be between 1 and 5.
    You must return your response in a line with only the score.
    Do not return answers in any other format.
    On a separate line provide your reasoning for the score as well.
    """
```


### Dashboard
```bash
export APP_PORT=8501
echo https://$APP_PORT-port-$ANYSCALE_SESSION_DOMAIN
streamlit run dashboard/Home.py
```


### TODO
- notebook cleanup
- evaluator (ex. GPT4) response script
- DB dump & load
- experiments
- routing b/w LLMs
- CI/CD workflows
