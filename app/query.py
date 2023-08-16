import os
import time

import numpy as np
import openai
import psycopg
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pgvector.psycopg import register_vector

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_response(
    llm,
    system_content,
    assistant_content,
    user_content,
    max_retries=3,
    retry_interval=60,
    api_base="https://api.openai.com/v1",
):
    """Generate response from an LLM."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                api_base=api_base,
                model=llm,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": user_content},
                ],
            )
            return response["choices"][-1]["message"]["content"]
        except Exception as e:  # NOQA: F841
            print(e)
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""


class QueryAgent:
    def __init__(
        self,
        embedding_model="thenlper/gte-base",
        llm="gpt-3.5-turbo-16k",
        max_context_length=16000,
        system_content="",
        assistant_content="",
    ):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.llm = llm
        self.context_length = max_context_length - len(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content

        # VectorDB connection
        self.conn = psycopg.connect(os.getenv("DB_CONNECTION_STRING"))
        register_vector(self.conn)

    def get_response(self, query):
        # Get context
        embedding = np.array(self.embedding_model.embed_query(query))
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM document ORDER BY embedding <-> %s LIMIT 5", (embedding,))
            rows = cur.fetchall()
            context = [{"text": row[1]} for row in rows]
            sources = [row[2] for row in rows]

        # Generate response
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
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
