# LLM Applications

A Comprehensive Guide for Developing and Serving RAG Applications in Production.

- **Blog post**: https://www.anyscale.com/blog
- **GitHub repository**: https://github.com/ray-project/llm-applications
- **Interactive notebook**: https://github.com/ray-project/llm-applications/blob/main/notebooks/rag.ipynb
- **Anyscale Endpoints**: https://endpoints.anyscale.com/
- **Ray documentation**: https://docs.ray.io/

In this guide, we will learn how to:

- 💻 Develop a retrieval augmented generation (RAG) based LLM application.
- 🚀 Scale the major components (embed, index, serve, etc.) in our application.
- ✅ Evaluate different configurations of our application to optimize for both per-component (ex. `retrieval_score`) and overall performance (`quality_score`).
- 🔀 Implement a hybrid routing approach that closes the gap between open-source and closed-source LLMs.
- 📦 Serve the application in a highlight available and scalable manner.

## Setup

### API keys
We'll be using [OpenAI](https://platform.openai.com/docs/models/) to access ChatGPT models like `gpt-3.5-turbo`, `gpt-4`, etc. and [Anyscale Endpoints](https://endpoints.anyscale.com/) to access OSS LLMs like `Llama-2-70b`. Be sure to create your accounts for both and have your credentials ready.

### Compute
- Start a new [Anyscale workspace on staging](https://console.anyscale-staging.com/o/anyscale-internal/workspaces) using an [`g3.8xlarge`](https://instances.vantage.sh/aws/ec2/g3.8xlarge) head node (you can also add GPU worker nodes to run the workloads faster). If you're not on Anyscale, you can configure a similar instance on your cloud.
- Use the [`default_cluster_env_2.6.2_py39`](https://docs.anyscale.com/reference/base-images/ray-262/py39#ray-2-6-2-py39) cluster environment.
- Use the `us-west-2` if you'd like to use the artifacts in our shared storage (source docs, vector DB dumps, etc.).

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
git clone https://github.com/ray-project/llm-applications.git .
```

### Environment

Then set up the environment correctly by specifying the values in your `.env` file,
and installing the dependencies:

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
