import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import ray
import requests
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from ray import serve
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from starlette.responses import StreamingResponse

from rag.config import EMBEDDING_DIMENSIONS, MAX_CONTEXT_LENGTHS
from rag.generate import QueryAgent
from rag.index import build_or_load_index

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_secret(secret_name):
    import boto3

    client = boto3.client("secretsmanager", region_name="us-west-2")
    response = client.get_secret_value(SecretId=os.environ["RAY_ASSISTANT_SECRET"])
    return json.loads(response["SecretString"])[secret_name]


@ray.remote
class SlackApp:
    def __init__(self):
        slack_app = App(token=get_secret("SLACK_BOT_TOKEN"))

        @slack_app.event("app_mention")
        def event_mention(body, say):
            event = body["event"]
            thread_ts = event.get("thread_ts", None) or event["ts"]
            text = event["text"][15:]  # strip slack user id of bot mention
            result = requests.post("http://127.0.0.1:8000/query/", json={"query": text}).json()
            reply = result["answer"] + "\n" + "\n".join(result["sources"])
            say(reply, thread_ts=thread_ts)

        self.slack_app = slack_app

    def run(self):
        SocketModeHandler(self.slack_app, get_secret("SLACK_APP_TOKEN")).start()


class Query(BaseModel):
    query: str


class Answer(BaseModel):
    question: str
    answer: str
    sources: List[str]
    llm: str


@serve.deployment(
    route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 6, "num_gpus": 1}
)
@serve.ingress(app)
class RayAssistantDeployment:
    def __init__(
        self,
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
        sql_dump_fp=None,
        run_slack=False,
    ):
        # Configure logging
        logging.basicConfig(
            filename=os.environ["RAY_ASSISTANT_LOGS"], level=logging.INFO, encoding="utf-8"
        )
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        self.logger = structlog.get_logger()

        # Set credentials
        os.environ["ANYSCALE_API_BASE"] = "https://api.endpoints.anyscale.com/v1"
        os.environ["ANYSCALE_API_KEY"] = get_secret("ANYSCALE_API_KEY")
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_API_KEY")
        os.environ["DB_CONNECTION_STRING"] = get_secret("DB_CONNECTION_STRING")

        # Set up
        chunks = build_or_load_index(
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sql_dump_fp=sql_dump_fp,
        )

        # Lexical index
        lexical_index = None
        self.lexical_search_k = lexical_search_k
        if use_lexical_search:
            texts = [re.sub(r"[^a-zA-Z0-9]", " ", chunk[1]).lower().split() for chunk in chunks]
            lexical_index = BM25Okapi(texts)

        # Reranker
        reranker = None
        self.rerank_threshold = rerank_threshold
        self.rerank_k = rerank_k
        if use_reranking:
            reranker_fp = Path(os.environ["RAY_ASSISTANT_RERANKER_MODEL"])
            with open(reranker_fp, "rb") as file:
                reranker = pickle.load(file)

        # Query agent
        self.num_chunks = num_chunks
        system_content = (
            "Answer the query using the context provided. Be succinct. "
            "Contexts are organized in a list of dictionaries [{'text': <context>}, {'text': <context>}, ...]. "
            "Feel free to ignore any contexts in the list that don't seem relevant to the query. "
        )
        self.oss_agent = QueryAgent(
            embedding_model_name=embedding_model_name,
            chunks=chunks,
            lexical_index=lexical_index,
            reranker=reranker,
            llm=llm,
            max_context_length=MAX_CONTEXT_LENGTHS[llm],
            system_content=system_content,
        )
        self.gpt_agent = QueryAgent(
            embedding_model_name=embedding_model_name,
            chunks=chunks,
            lexical_index=lexical_index,
            reranker=reranker,
            llm="gpt-4",
            max_context_length=MAX_CONTEXT_LENGTHS["gpt-4"],
            system_content=system_content,
        )

        # Router
        router_fp = Path(os.environ["RAY_ASSISTANT_ROUTER_MODEL"])
        with open(router_fp, "rb") as file:
            self.router = pickle.load(file)

        if run_slack:
            # Run the Slack app in the background
            self.slack_app = SlackApp.remote()
            self.runner = self.slack_app.run.remote()

    def predict(self, query: Query, stream: bool) -> Dict[str, Any]:
        use_oss_agent = self.router.predict([query.query])[0]
        agent = self.oss_agent if use_oss_agent else self.gpt_agent
        result = agent(
            query=query.query,
            num_chunks=self.num_chunks,
            lexical_search_k=self.lexical_search_k,
            rerank_threshold=self.rerank_threshold,
            rerank_k=self.rerank_k,
            stream=stream,
        )
        return result

    @app.post("/query")
    def query(self, query: Query) -> Answer:
        result = self.predict(query, stream=False)
        return Answer.parse_obj(result)

    def produce_streaming_answer(self, query, result):
        answer = []
        for answer_piece in result["answer"]:
            answer.append(answer_piece)
            yield answer_piece

        if result["sources"]:
            yield "\n\n**Sources:**\n"
            for source in result["sources"]:
                yield "* " + source + "\n"

        self.logger.info(
            "finished streaming query",
            query=query,
            document_ids=result["document_ids"],
            llm=result["llm"],
            answer="".join(answer),
        )

    @app.post("/stream")
    def stream(self, query: Query) -> StreamingResponse:
        result = self.predict(query, stream=True)
        return StreamingResponse(
            self.produce_streaming_answer(query.query, result), media_type="text/plain"
        )


# Deploy the Ray Serve app
deployment = RayAssistantDeployment.bind(
    chunk_size=700,
    chunk_overlap=50,
    num_chunks=50, # these will be filtered by rerank_k below
    embedding_model_name=os.environ["RAY_ASSISTANT_EMBEDDING_MODEL"],
    embedding_dim=EMBEDDING_DIMENSIONS["thenlper/gte-large"],
    use_lexical_search=False,
    lexical_search_k=0,
    use_reranking=True,
    rerank_threshold=0.9,
    rerank_k=9,
    llm="mistralai/Mixtral-8x7B-Instruct-v0.1",
    sql_dump_fp=Path(os.environ["RAY_ASSISTANT_INDEX"]),
)


# Simpler, non-fine-tuned version
# deployment = RayAssistantDeployment.bind(
#     chunk_size=700,
#     chunk_overlap=50,
#     num_chunks=9,
#     embedding_model_name="thenlper/gte-large",  # fine-tuned is slightly better
#     embedding_dim=EMBEDDING_DIMENSIONS["thenlper/gte-large"],
#     use_lexical_search=False,
#     lexical_search_k=0,
#     use_reranking=True,
#     rerank_threshold=0.9,
#     rerank_k=9,
#     llm="codellama/CodeLlama-34b-Instruct-hf",
# )
