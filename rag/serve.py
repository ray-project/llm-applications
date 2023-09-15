# You can run the whole script locally with
# serve run rag.serve:deployment

import json
import os
import pickle
from pathlib import Path
from typing import List

import openai
import ray
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ray import serve
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from starlette.responses import StreamingResponse

from rag.config import MAX_CONTEXT_LENGTHS, ROOT_DIR
from rag.generate import QueryAgent

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
    response = client.get_secret_value(SecretId="ray-assistant")
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
    route_prefix="/", num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 1}
)
@serve.ingress(app)
class RayAssistantDeployment:
    def __init__(self, num_chunks, embedding_model_name, llm, run_slack=False):
        # Set credentials
        os.environ["ANYSCALE_API_BASE"] = "https://api.endpoints.anyscale.com/v1"
        os.environ["ANYSCALE_API_KEY"] = get_secret("ANYSCALE_API_KEY")
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_API_KEY")
        os.environ["DB_CONNECTION_STRING"] = get_secret("DB_CONNECTION_STRING")

        # Query agent
        self.num_chunks = num_chunks
        system_content = "Answer the query using the context provided. Be succint."
        self.oss_agent = QueryAgent(
            embedding_model_name=embedding_model_name,
            llm=llm,
            max_context_length=MAX_CONTEXT_LENGTHS[llm],
            system_content=system_content,
        )
        self.gpt_agent = QueryAgent(
            embedding_model_name=embedding_model_name,
            llm="gpt-4",
            max_context_length=MAX_CONTEXT_LENGTHS["gpt-4"],
            system_content=system_content,
        )

        # Router
        router_fp = Path(ROOT_DIR, "datasets", "router.pkl")
        with open(router_fp, "rb") as file:
            self.router = pickle.load(file)

        if run_slack:
            # Run the Slack app in the background
            self.slack_app = SlackApp.remote()
            self.runner = self.slack_app.run.remote()

    @app.post("/query")
    def query(self, query: Query) -> Answer:
        use_oss_agent = self.router.predict([query.query])[0]
        agent = self.oss_agent if use_oss_agent else self.gpt_agent
        result = agent(query=query.query, num_chunks=self.num_chunks, stream=False)
        return Answer.parse_obj(result)

    def produce_streaming_answer(self, result):
        for answer_piece in result["answer"]:
            yield answer_piece
        if result["sources"]:
            yield "\n\n**Sources:**\n"
            for source in result["sources"]:
                yield "* " + source + "\n"

    @app.post("/stream")
    def stream(self, query: Query) -> StreamingResponse:
        use_oss_agent = self.router.predict([query.query])[0]
        agent = self.oss_agent if use_oss_agent else self.gpt_agent
        result = agent(query=query.query, num_chunks=self.num_chunks, stream=True)
        return StreamingResponse(
            self.produce_streaming_answer(result), media_type="text/plain")


# Deploy the Ray Serve app
deployment = RayAssistantDeployment.bind(
    num_chunks=7,
    embedding_model_name="thenlper/gte-large",
    llm="meta-llama/Llama-2-70b-chat-hf",
)
