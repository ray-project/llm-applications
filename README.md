# LLM Applications

An end-to-end guide for scaling and serving LLM application in production.

This repo currently contains one such application: a retrieval-augmented generation (RAG)
app for answering questions about supplied information. By default, the app uses
the [Ray documentation](https://docs.ray.io/en/master/) as the source of information.
This app first [indexes](./app/index.py) the documentation in a vector database
and then uses an LLM to generate responses for questions that got augmented with
relevant info retrieved from the index.

## Setup

### API keys
We'll be using [OpenAI](https://platform.openai.com/docs/models/){:target="_blank"} to access ChatGPT models like `gpt-3.5-turbo`, `gpt-4`, etc. and [Anyscale Endpoints](https://endpoints.anyscale.com/){:target="_blank"} to access OSS LLMs like `Llama-2-70b`. Be sure to create your accounts for both and have your credentials ready.

### Compute
- Start a new [Anyscale workspace on staging](https://console.anyscale-staging.com/o/anyscale-internal/workspaces) using an [`g3.8xlarge`](https://instances.vantage.sh/aws/ec2/g3.8xlarge) head node (you can also add GPU worker nodes to run the workloads faster).
- Use the [`default_cluster_env_2.6.2_py39`](https://docs.anyscale.com/reference/base-images/ray-262/py39#ray-2-6-2-py39) cluster environment.
- Use the `us-east-1` if you'd like to use the artifacts in our shared storage (source docs, vector DB dumps, etc.).

### Repository
```bash
git clone https://github.com/ray-project/llm-applications.git .
git config --global user.name <GITHUB-USERNAME>
git config --global user.email <EMAIL-ADDRESS>
```

### Data
Our data is already ready at `/efs/shared_storage/pcmoritz/docs.ray.io/en/master/` (on Staging, `us-east-1`) but if you wanted to load it yourself, run this bash command (change `/desired/output/directory`, but make sure it's on the shared storage,
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

### Vector DB
Preload from saved SQL dump:
```bash
bash setup-pgvector.sh
export SQL_DUMP="/efs/shared_storage/pcmoritz/gtebase-300-50.sql"
psql "$DB_CONNECTION_STRING" -f $SQL_DUMP
```

Create new index:
```bash
bash setup-pgvector.sh
sudo -u postgres psql -f migrations/initial.sql
export DOCS_PATH="/efs/shared_storage/pcmoritz/docs.ray.io/en/master/"
python app/index.py create-index \
    --docs-path $DOCS_PATH \
    --embedding-model "thenlper/gte-base" \
    --chunk-size 300 \
    --chunk-overlap 50
```

Inspect DB:
```bash
psql "$DB_CONNECTION_STRING" -c "SELECT count(*) FROM document;"  # rows
psql "$DB_CONNECTION_STRING" -c "\d document;"  # columns
psql "$DB_CONNECTION_STRING" -c "SELECT source FROM document LIMIT 1;"  # sample
```

Save/load DB (`{embedding_model_name}-{chunk_size}-{chunk_overlap}.sql`):
```bash
export SQL_DUMP="/efs/shared_storage/pcmoritz/gtebase-300-50.sql"
sudo -u postgres pg_dump > $SQL_DUMP  # save
psql "$DB_CONNECTION_STRING" -c "DELETE FROM document;"
psql "$DB_CONNECTION_STRING" -f $SQL_DUMP  # load
```



### Query
Just a sample and uses the current index that's been created.

```python
import os
import json
import openai
from app.query import QueryAgent
openai.api_base = os.environ["ANYSCALE_API_BASE"]
openai.api_key = os.environ["ANYSCALE_API_KEY"]
query = "What is the default batch size for map_batches?"
system_content = "Your job is to answer a question using the additional context provided."
agent = QueryAgent(
    embedding_model_name="thenlper/gte-base",
    llm="meta-llama/Llama-2-7b-chat-hf",
    max_context_length=4096,
    system_content=system_content,
)
result = agent.get_response(query=query)
print(json.dumps(result, indent=2))
```

### Experiments

#### Generate responses
```bash
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"
export OPENAI_API_KEY=""  # https://app.endpoints.anyscale.com/credentials
export EXPERIMENT_NAME="llama-2-7b-gtebase"
export DATA_PATH="datasets/eval-dataset-v1.jsonl"
export CHUNK_SIZE=300
export CHUNK_OVERLAP=50
export EMBEDDING_MODEL_NAME="thenlper/gte-base"
export LLM="meta-llama/Llama-2-7b-chat-hf"
export TEMPERATURE 0
export MAX_CONTEXT_LENGTH=4096
```
```bash
python app/main.py generate-responses \
    --experiment-name $EXPERIMENT_NAME \
    --docs-path $DOCS_PATH \
    --data-path $DATA_PATH \
    --chunk-size $CHUNK_SIZE \
    --chunk-overlap $CHUNK_OVERLAP \
    --embedding-model-name $EMBEDDING_MODEL_NAME \
    --llm $LLM \
    --temperature $TEMPERATURE \
    --max-context-length $MAX_CONTEXT_LENGTH \
    --system-content "Answer the {query} using the additional {context} provided."
```

#### Evaluate responses
```bash
export OPENAI_API_BASE="https://api.endpoints.anyscale.com/v1"
export OPENAI_API_KEY=""  # https://app.endpoints.anyscale.com/credentials
export REFERENCE_LOC="experiments/responses/gpt-4-with-source.json"
export RESPONSE_LOC="experiments/responses/$EXPERIMENT_NAME.json"
export EVALUATOR="meta-llama/Llama-2-70b-chat-hf"
export EVALUATOR_TEMPERATURE=0
export EVALUATOR_MAX_CONTEXT_LENGTH=4096
```
```bash
python app/main.py evaluate-responses \
    --experiment-name $EXPERIMENT_NAME \
    --reference-loc $REFERENCE_LOC \
    --response-loc $RESPONSE_LOC \
    --evaluator $EVALUATOR \
    --temperature $EVALUATOR_TEMPERATURE \
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
- [x] notebook cleanup
- [x] evaluator (ex. GPT4) response script
- [x] DB dump & load
- experiments (in order and fixing choices along the way)
    - Evaluator
        - [x] `gpt-4`
        - [x] `Llama-2-70b`
    - Context (value of RAG)
        - [ ] without context
        - [ ] with context
    - Sections
        - [ ] without sections
        - [ ] with sections
    - Chunking size
        - [ ] Chunk-size: 100, 300, 600
        - Fix chunk overlap to 50
        - related to # of chunks + context length but we'll treat as indepdent variable
        - larger chunk size [isn't always](https://arxiv.org/pdf/2307.03172.pdf) better
    - Number of chunks
        - [ ] # chunks: 1, 5, 10
        - based on chunk size chosen from experiment and needs to fit in context length
        - Does using more resources help/harm?
    - Embedding models
        - [ ] `text-embedding-ada-002` (OpenAI)
        - [ ] `gte-base` ([MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard))
        - global leaderboard may not be your leaderboard (empirically validate)
    - OSS vs. Closed
        - [ ] `gpt-3.5-turbo`
        - [ ] `gpt-4`
        - [ ] `Llama-2-7b`
        - [ ] `Llama-2-13b`
        - [ ] `Llama-2-70b`
    - Later
        - [ ] Commercial Assistant evaluation
        - [ ] Human Assistant evaluation
        - [ ] Data sources
    - Much later
        - [ ] Prompt itself
        - [ ] Prompt-tuning on query
        - [ ] Embedding vs. LLM for retreival
- [ ] Ray Tune to tweak a subset of components
- [ ] CI/CD workflows
