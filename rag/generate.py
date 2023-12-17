import json
import pickle
import re
import time
from pathlib import Path

from IPython.display import JSON, clear_output, display
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from rag.config import EFS_DIR, ROOT_DIR
from rag.embed import get_embedding_model
from rag.index import build_or_load_index
from rag.rerank import custom_predict, get_reranked_indices
from rag.search import lexical_search, semantic_search
from rag.utils import get_client, get_num_tokens, trim


def response_stream(chat_completion):
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content


def prepare_response(chat_completion, stream):
    if stream:
        return response_stream(chat_completion)
    else:
        return chat_completion.choices[0].message.content


def generate_response(
    llm,
    max_tokens=None,
    temperature=0.0,
    stream=False,
    system_content="",
    assistant_content="",
    user_content="",
    max_retries=1,
    retry_interval=60,
):
    """Generate response from an LLM."""
    retry_count = 0
    client = get_client(llm=llm)
    messages = [
        {"role": role, "content": content}
        for role, content in [
            ("system", system_content),
            ("assistant", assistant_content),
            ("user", user_content),
        ]
        if content
    ]
    while retry_count <= max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=llm,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                messages=messages,
            )
            return prepare_response(chat_completion, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""


class QueryAgent:
    def __init__(
        self,
        embedding_model_name="thenlper/gte-base",
        chunks=None,
        lexical_index=None,
        reranker=None,
        llm="meta-llama/Llama-2-70b-chat-hf",
        temperature=0.0,
        max_context_length=4096,
        system_content="",
        assistant_content="",
    ):
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100},
        )

        # Lexical search
        self.chunks = chunks
        self.lexical_index = lexical_index

        # Reranker
        self.reranker = reranker

        # LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length = int(
            0.5 * max_context_length
        ) - get_num_tokens(  # 50% of total context reserved for input
            system_content + assistant_content
        )
        self.max_tokens = int(
            0.5 * max_context_length
        )  # max sampled output (the other 50% of total context)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(
        self,
        query,
        num_chunks=5,
        lexical_search_k=1,
        rerank_threshold=0.2,
        rerank_k=7,
        stream=True,
    ):
        # Get top_k context
        context_results = semantic_search(
            query=query, embedding_model=self.embedding_model, k=num_chunks
        )

        # Add lexical search results
        if self.lexical_index:
            lexical_context = lexical_search(
                index=self.lexical_index, query=query, chunks=self.chunks, k=lexical_search_k
            )
            # Insert after <lexical_search_k> worth of semantic results
            context_results[lexical_search_k:lexical_search_k] = lexical_context

        # Rerank
        if self.reranker:
            predicted_tag = custom_predict(
                inputs=[query], classifier=self.reranker, threshold=rerank_threshold
            )[0]
            if predicted_tag != "other":
                sources = [item["source"] for item in context_results]
                reranked_indices = get_reranked_indices(sources, predicted_tag)
                context_results = [context_results[i] for i in reranked_indices]
            context_results = context_results[:rerank_k]

        # Generate response
        document_ids = [item["id"] for item in context_results]
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=trim(user_content, self.context_length),
        )

        # Result
        result = {
            "question": query,
            "sources": sources,
            "document_ids": document_ids,
            "answer": answer,
            "llm": self.llm,
        }
        return result


# Generate responses
def generate_responses(
    experiment_name,
    chunk_size,
    chunk_overlap,
    num_chunks,
    embedding_model_name,
    embedding_dim,
    use_lexical_search,
    lexical_search_k,
    use_reranking,
    rerank_threshold,
    rerank_k,
    llm,
    temperature,
    max_context_length,
    system_content,
    assistant_content,
    docs_dir,
    experiments_dir,
    references_fp,
    num_samples=None,
    sql_dump_fp=None,
):
    # Build index
    chunks = build_or_load_index(
        embedding_model_name=embedding_model_name,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        docs_dir=docs_dir,
        sql_dump_fp=sql_dump_fp,
    )

    # Lexical index
    lexical_index = None
    if use_lexical_search:
        texts = [re.sub(r"[^a-zA-Z0-9]", " ", chunk[1]).lower().split() for chunk in chunks]
        lexical_index = BM25Okapi(texts)

    # Reranker
    reranker = None
    if use_reranking:
        reranker_fp = Path(EFS_DIR, "reranker.pkl")
        with open(reranker_fp, "rb") as file:
            reranker = pickle.load(file)

    # Query agent
    agent = QueryAgent(
        embedding_model_name=embedding_model_name,
        chunks=chunks,
        lexical_index=lexical_index,
        reranker=reranker,
        llm=llm,
        temperature=temperature,
        system_content=system_content,
        assistant_content=assistant_content,
    )

    # Generate responses
    results = []
    with open(Path(references_fp), "r") as f:
        questions = [item["question"] for item in json.load(f)][:num_samples]
    for query in tqdm(questions):
        result = agent(
            query=query,
            num_chunks=num_chunks,
            lexical_search_k=lexical_search_k,
            rerank_threshold=rerank_threshold,
            rerank_k=rerank_k,
            stream=False,
        )
        results.append(result)
        clear_output(wait=True)
        display(JSON(json.dumps(result, indent=2)))

    # Save to file
    responses_fp = Path(ROOT_DIR, experiments_dir, "responses", f"{experiment_name}.json")
    responses_fp.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_name": experiment_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": num_chunks,
        "embedding_model_name": embedding_model_name,
        "llm": llm,
        "temperature": temperature,
        "max_context_length": max_context_length,
        "system_content": system_content,
        "assistant_content": assistant_content,
        "docs_dir": str(docs_dir),
        "experiments_dir": str(experiments_dir),
        "references_fp": str(references_fp),
        "num_samples": len(questions),
    }
    responses = {
        "config": config,
        "results": results,
    }
    with open(responses_fp, "w") as fp:
        json.dump(responses, fp, indent=4)
