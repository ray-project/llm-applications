import os
import sys
import time

import numpy as np
import openai
import psycopg
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pgvector.psycopg import register_vector


def prepare_response(response, stream):
    if stream:
        answer = []
        for chunk in response:
            if "content" in chunk["choices"][0]["delta"].keys():
                content = chunk["choices"][0]["delta"]["content"]
                answer.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
        return "".join([item for item in answer])
    else:
        return response["choices"][-1]["message"]["content"]


def generate_response(
    llm,
    temperature=0.0,
    stream=False,
    system_content="",
    assistant_content="",
    user_content="",
    max_retries=3,
    retry_interval=60,
):
    """Generate response from an LLM."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=llm,
                temperature=temperature,
                stream=stream,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": user_content},
                ],
            )
            return prepare_response(response=response, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""


def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    return embedding_model


def get_sources_and_context(query, embedding_model, num_chunks):
    embedding = np.array(embedding_model.embed_query(query))
    with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM document ORDER BY embedding <-> %s LIMIT %s", (embedding, num_chunks))
            rows = cur.fetchall()
            context = [{"text": row[1]} for row in rows]
            sources = [row[2] for row in rows]
    return sources, context


class QueryAgent:
    def __init__(
        self,
        embedding_model_name="thenlper/gte-base",
        llm="meta-llama/Llama-2-70b-chat-hf",
        temperature=0.0,
        max_context_length=4096,
        system_content="",
        assistant_content="",
    ):
        # Embedding model
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"device": "cuda", "batch_size": 100}
        if embedding_model_name == "text-embedding-ada-002":
            self.embedding_model = OpenAIEmbeddings(
                model=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
            )
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )

        # LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length = max_context_length - len(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(self, query, num_chunks=5):
        # Get context
        embedding = np.array(self.embedding_model.embed_query(query))
        with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM document ORDER BY embedding <-> %s LIMIT %s", (embedding, num_chunks))
                rows = cur.fetchall()
                context = [{"text": row[1]} for row in rows]
                sources = [row[2] for row in rows]

        # Generate response
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            stream=True,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=user_content[: self.context_length],
        )

        # Result
        result = {
            "question": query,
            "sources": sources,
            "answer": answer,
        }
        return result


# Generate responses
def generate_responses(
    experiment_name,
    data_path,
    sections,
    chunk_size,
    chunk_overlap,
    num_chunks,
    embedding_model_name,
    llm,
    temperature,
    max_context_length,
    system_content,
    assistant_content="",
    num_samples=None,
):
    # Build index
    set_index(
        sections=sections, embedding_model_name=embedding_model_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Query agent
    agent = QueryAgent(
        embedding_model_name=embedding_model_name,
        llm=llm,
        temperature=temperature,
        system_content=system_content,
        assistant_content=assistant_content,
    )

    # Generate responses
    results = []
    with open(Path(data_path), "r") as f:
        questions = [json.loads(item)["question"] for item in list(f)][:num_samples]
    for query in tqdm(questions):
        result = agent(query=query, num_chunks=num_chunks)
        results.append(result)
        clear_output(wait=True)
        display(JSON(json.dumps(result, indent=2)))

    # Save to file
    responses_fp = Path(ROOT_DIR, EXPERIMENTS_DIR, "responses", f"{experiment_name}.json")
    responses_fp.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": experiment_name,
        "data_path": data_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": num_chunks,
        "embedding_model_name": embedding_model_name,
        "llm": llm,
        "temperature": temperature,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
    }
    responses = {
        "config": config,
        "results": results,
    }
    with open(responses_fp, "w") as fp:
        json.dump(responses, fp, indent=4)
