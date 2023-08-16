import os
from pathlib import Path

import psycopg
import ray
import typer
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector
from ray.data import ActorPoolStrategy
from typing_extensions import Annotated

app = typer.Typer()


def load_html_file(path):
    with open(path) as f:
        soup = BeautifulSoup(f.read())

    html_tags = [
        ("div", {"role": "main"}),
        ("main", {"id": "main-content"}),
    ]

    text = None
    for tag, attrs in html_tags:
        text = soup.find(tag, attrs)
        # if found, break
        if text is not None:
            break

    return text


class TaggedStr:
    def __init__(self, value, tag):
        self.value = value
        self.tag = tag

    def __repr__(self):
        return repr(self.value) + f" [{self.tag}]" if self.tag else ""


def convert_to_tagged_text(path, element, section=None):
    "Recursively convert a BeautifulSoup element to text, keeping track of sections."
    results = []
    for child in element.children:
        if isinstance(child, NavigableString):
            results.append(TaggedStr(str(child), section))
        elif isinstance(child, Tag):
            if child.name == "section" and "id" in child.attrs:
                results.extend(convert_to_tagged_text(path, child, section=child.attrs["id"]))
            elif not child.find_all("section"):
                results.append(TaggedStr(child.get_text(), section))
            else:
                results.extend(convert_to_tagged_text(path, child, section))
    return results


def group_tagged_text(chunks):
    result = []
    for item in chunks:
        if result and item.value.strip() == "":
            result[-1].value += item.value
        elif result and item.tag == result[-1].tag:
            result[-1].value += item.value
        else:
            result.append(item)
    return result


def path_to_uri(path, scheme="https://", domain="docs.ray.io"):
    return scheme + domain + path.split(domain)[-1]


def parse_file(record):
    html_content = load_html_file(record["path"])
    if not html_content:
        return []
    return [
        {
            "source": path_to_uri(str(record["path"])) + ("#" + chunk.tag if chunk.tag else ""),
            "text": chunk.value,
        }
        for chunk in group_tagged_text(convert_to_tagged_text(record["path"], html_content))
    ]


class EmbedNodes:
    def __init__(self, model_name):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100},
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {
            "text": batch["text"],
            "source": batch["source"],
            "embeddings": embeddings,
        }


class StoreResults:
    def __call__(self, batch):
        with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding in zip(
                    batch["text"], batch["source"], batch["embeddings"]
                ):
                    cur.execute(
                        "INSERT INTO document (text, source, embedding) VALUES (%s, %s, %s)",
                        (
                            text,
                            source,
                            embedding,
                        ),
                    )
        return {}


@app.command()
def create_index(
    docs_path: Annotated[str, typer.Option(help="location of data")] = "",
    embedding_model: Annotated[str, typer.Option(help="embedder")] = "thenlper/gte-base",
    chunk_size: Annotated[int, typer.Option(help="chunk size")] = 300,
    chunk_overlap: Annotated[int, typer.Option(help="chunk overlap")] = 50,
):
    # Initialize ray
    ray.init(
        runtime_env={"env_vars": {"DB_CONNECTION_STRING": os.environ["DB_CONNECTION_STRING"]}}
    )

    # Dataset
    ds = ray.data.from_items(
        [{"path": path} for path in Path(docs_path).rglob("*.html") if not path.is_dir()]
    )

    # Sections
    sections_ds = ds.flat_map(parse_file)
    sections = sections_ds.take_all()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"] for section in sections],
        metadatas=[{"source": section["source"]} for section in sections],
    )
    chunks_ds = ray.data.from_items(
        [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]
    )

    # Embed data
    embedded_nodes = chunks_ds.map_batches(
        EmbedNodes,
        fn_constructor_kwargs={"model_name": embedding_model},
        batch_size=100,
        num_gpus=1,
        compute=ActorPoolStrategy(size=2),
    )

    # Index data
    embedded_nodes.map_batches(
        StoreResults,
        batch_size=128,
        num_cpus=1,
        compute=ActorPoolStrategy(size=20),
    ).count()


@app.command()
def reset_index():
    with psycopg.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document")


if __name__ == "__main__":
    app()
