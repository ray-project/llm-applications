# You can run the whole script locally with
# serve run serve:deployment

import os

import query
import ray
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = FastAPI()


@ray.remote
class SlackApp:
    def __init__(self):
        slack_app = App(token=os.environ["SLACK_BOT_TOKEN"])

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
        SocketModeHandler(self.slack_app, os.environ["SLACK_APP_TOKEN"]).start()


ray.init(
    runtime_env={
        "env_vars": {
            "DB_CONNECTION_STRING": os.environ["DB_CONNECTION_STRING"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "SLACK_APP_TOKEN": os.environ["SLACK_APP_TOKEN"],
            "SLACK_BOT_TOKEN": os.environ["SLACK_BOT_TOKEN"],
        }
    },
    ignore_reinit_error=True,
)


class Query(BaseModel):
    query: str


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[str]


@serve.deployment()
@serve.ingress(app)
class RayAssistantDeployment:
    def __init__(self):
        self.agent = query.QueryAgent()
        self.app = SlackApp.remote()
        # Run the slack app in the background
        self.runner = self.app.run.remote()

    @app.post("/query")
    def query(self, query: Query) -> Answer:
        result = self.agent.get_response(query.query)
        return Answer.parse_obj(result)


# Deploy the Ray Serve application.
deployment = RayAssistantDeployment.bind()
