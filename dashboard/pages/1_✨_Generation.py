import os
import time
from pathlib import Path

import numpy as np
import openai
import psycopg
import ray
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector

from app.index import parse_file


@st.cache_data
def get_ds(docs_path):
    return ray.data.from_items(
        [{"path": path} for path in docs_path.rglob("*.html") if not path.is_dir()]
    )


@st.cache_data
def generate_response(
    llm,
    system_content,
    assistant_content,
    user_content,
    max_retries=3,
    retry_interval=60,
):
    """Generate response from an LLM."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=llm,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": user_content},
                ],
            )
            return response["choices"][-1]["message"]["content"]
        except Exception as e:  # NOQA: F841
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1


# Title
st.title("Generation Inspector")

# Load data
st.header("Load data")
docs_path_str = st.text_input(
    "Location of the docs", "/efs/shared_storage/pcmoritz/docs.ray.io/en/master/"
)
docs_path = Path(docs_path_str)
ds = get_ds(docs_path=docs_path)
st.write(f"{ds.count()} documents")

# Sections
st.header("Sections")
st.text("View the sections for a particular docs page")
docs_page_url = st.text_input("Docs page URL", "https://docs.ray.io/en/master/train/faq.html")
docs_page_path = docs_path_str + docs_page_url.split("docs.ray.io/en/master/")[-1]
with st.expander("View sections"):
    sections = parse_file({"path": docs_page_path})
    st.write(sections)

# Chunks
st.header("Chunks")
st.markdown(f"View the chunks for the sections in [{docs_page_url}]({docs_page_url})")
separators = ["\n\n", "\n", " ", ""]
st.write(f"Separators: {separators}")
chunk_size = st.number_input(label="chunk size", min_value=1, value=300)
chunk_overlap = st.number_input(label="chunk overlap", min_value=0, value=50)
text_splitter = RecursiveCharacterTextSplitter(
    separators=separators,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)
chunks = text_splitter.create_documents(
    texts=[section["text"] for section in sections],
    metadatas=[{"source": section["source"]} for section in sections],
)
st.write(f"{len(chunks)} chunks")
with st.expander("View chunks"):
    st.write(
        [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]
    )

# Retrieval
st.header("Retrieval")
st.text("Retieve the top N closest chunks (nodes) based on your query.")
st.markdown("The index is built for `chunk_size=300` and `chunk_overlap=50`")
embedding_model = st.text_input("Embedding model", "thenlper/gte-base")
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
conn = psycopg.connect(os.environ["DB_CONNECTION_STRING"])
register_vector(conn)
query = st.text_input("Query", "What is the default batch size for map_batches?")
embedding = np.array(embedding_model.embed_query(query))
st.write(f"Embedding dimension: {len(embedding)}")
with conn.cursor() as cur:
    cur.execute("SELECT * FROM document ORDER BY embedding <-> %s LIMIT 5", (embedding,))
    rows = cur.fetchall()
    context = [{"text": row[2], "source": row[1]} for row in rows]
with st.expander("View context"):
    st.write(context)

# Generation
st.header("Generation")
llm = st.text_input("Name of LLM to use", "gpt-3.5-turbo-16k")
system_content = st.text_input(
    "Instructions for the LLM", "Answer the {query} using the provided {context}"
)
user_content = f"query: {query}, context: {context}"
response = generate_response(
    llm=llm,
    system_content=system_content,
    assistant_content="",
    user_content=user_content,
)
st.write(response)
