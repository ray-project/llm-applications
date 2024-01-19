#!/bin/bash

# Load documents
# export EFS_DIR=$(python -c "from rag.config import EFS_DIR; print(EFS_DIR)")
# wget -e robots=off --recursive --no-clobber --page-requisites \
#   --html-extension --convert-links --restrict-file-names=windows \
#   --domains docs.ray.io --no-parent --accept=html --retry-on-http-error=429 \
#   -P $EFS_DIR https://docs.ray.io/en/master/

# Drop and create table
export DB_CONNECTION_STRING="dbname=postgres user=postgres host=localhost password=postgres"  # TODO: move to CI/CD secrets manager
export EMBEDDING_MODEL_NAME="thenlper/gte-large"  # TODO: use service env vars
export MIGRATION_FP="migrations/vector-1024.sql"  # TODO: dynamically set this
export SQL_DUMP_FILE="/mnt/shared_storage/ray-assistant-data/index.sql"
psql "$DB_CONNECTION_STRING" -c "DROP TABLE IF EXISTS document;"
sudo -u postgres psql -f $MIGRATION_FP

# Build index (fixed for now, need to make dynamic)
python << EOF
import os
from pathlib import Path
from rag.config import EFS_DIR
from rag.index import build_index
build_index(
    docs_dir=Path(EFS_DIR, "docs.ray.io/en/master/"),
    chunk_size=700,
    chunk_overlap=50,
    embedding_model_name=os.environ["EMBEDDING_MODEL_NAME"],
    sql_dump_fp=os.environ["SQL_DUMP_FILE"])
EOF
