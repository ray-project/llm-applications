import os

import ray
from ray import serve

from app.serve import deployment

# Credentials
ray.init(
    runtime_env={
        "env_vars": {
            "OPENAI_API_BASE": os.environ["OPENAI_API_BASE"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "ANYSCALE_API_BASE": os.environ["ANYSCALE_API_BASE"],
            "ANYSCALE_API_KEY": os.environ["ANYSCALE_API_KEY"],
            "DB_CONNECTION_STRING": os.environ["DB_CONNECTION_STRING"],
        }
    }
)

# Serve
serve.run(deployment)
