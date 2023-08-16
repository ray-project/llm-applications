# You can run the whole script locally with
# serve run serve:deployment

import os

import query
import ray
from fastapi import FastAPI
from ray import serve
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = FastAPI()


@ray.remote
class SlackApp:
    def __init__(self):
        slack_app = App(token=os.environ["SLACK_BOT_TOKEN"])
        self.agent = query.QueryAgent()

        @slack_app.event("app_mention")
        def event_mention(body, say):
            text = body["event"]["text"][15:]  # strip slack user id of bot mention
            result = self.agent.get_response(text)
            reply = result["answer"] + "\n" + "\n".join(result["sources"])
            say(reply)

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
    }
)


@serve.deployment()
@serve.ingress(app)
class RayAssistantDeployment:
    def __init__(self):
        self.app = SlackApp.remote()
        # Run the slack app in the background
        self.runner = self.app.run.remote()


# Deploy the Ray Serve application.
deployment = RayAssistantDeployment.bind()
