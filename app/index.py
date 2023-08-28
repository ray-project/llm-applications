from pathlib import Path

import psycopg
import ray
import typer
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from pgvector.psycopg import register_vector
from ray.data import ActorPoolStrategy
from typing import Any, List, Optional
from typing_extensions import Annotated

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_CONNECTION_STRING,
    DEVICE,
    DOCS_PATH,
    EMBEDDING_ACTORS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    INDEXING_ACTORS,
    INDEXING_BATCH_SIZE,
    NUM_GPUS,
)

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
    """Recursively convert a BeautifulSoup element to text, keeping track of sections."""
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


def parse_html_file(record):
    html_content = load_html_file(record["path"])
    if not html_content:
        return []
    return [
        {
            "source": path_to_uri(str(record["path"])) + ("#" + chunk.tag if chunk.tag else ""),
            "text": chunk.value,
        }
        for chunk in group_tagged_text(convert_to_tagged_text(record["path"], html_content))
        if chunk.value.strip() != ""
    ]


def parse_text_file(record):
    with open(record["path"]) as f:
        text = f.read()
    return [
        {
            "source": str(record["path"]),
            "text": text,
        }
    ]


class SimpleTextSplitter(TextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        min_chunk_size = 100,
        max_chunk_size = 300,
        keep_separator: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new SimpleTextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._separators = separators or ["\n\n", "\n", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        chunks = []
        index = -1
        while len(text) > 0:
            # First see if the text is already small enough
            if len(text) <= self._max_chunk_size:
                chunks.append(text)
                break
            # If not, try to break it up
            for separator in separators:
                index = text.find(separator, self._min_chunk_size)
                if index != -1 and index <= self._max_chunk_size:
                    chunks.append(text[:index])
                    text = text[index:]
                    break
            if index == -1 or index > self._max_chunk_size:
                # In this case, none of the separators was applicable,
                # so we just make a hard cut
                chunks.append(text[:self._max_chunk_size])
                text = text[self._max_chunk_size:]
        return chunks


    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)


class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"device": DEVICE, "batch_size": EMBEDDING_BATCH_SIZE},
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
        with psycopg.connect(DB_CONNECTION_STRING) as conn:
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
    docs_path: Annotated[str, typer.Option(help="location of data")] = DOCS_PATH,
    extension_type: Annotated[str, typer.Option(help="type of data")] = "html",
    embedding_model: Annotated[str, typer.Option(help="embedder")] = EMBEDDING_MODEL,
    chunk_size: Annotated[int, typer.Option(help="chunk size")] = CHUNK_SIZE,
    chunk_overlap: Annotated[int, typer.Option(help="chunk overlap")] = CHUNK_OVERLAP,
):
    # Initialize ray
    ray.init(runtime_env={"env_vars": {"DB_CONNECTION_STRING": DB_CONNECTION_STRING}})

    # Dataset
    ds = ray.data.from_items(
        [
            {"path": path}
            for path in Path(docs_path).rglob(f"*.{extension_type}")
            if not path.is_dir()
        ]
    )

    # Sections
    parser = parse_html_file if extension_type == "html" else parse_text_file
    sections_ds = ds.flat_map(parser)
    # TODO: do we really need to take_all()? Bring the splitter to the cluster
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
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": embedding_model},
        batch_size=EMBEDDING_BATCH_SIZE,
        num_gpus=NUM_GPUS,
        compute=ActorPoolStrategy(size=EMBEDDING_ACTORS),
    )

    # Index data
    embedded_chunks.map_batches(
        StoreResults,
        batch_size=INDEXING_BATCH_SIZE,
        num_cpus=1,
        compute=ActorPoolStrategy(size=INDEXING_ACTORS),
    ).count()

    return sections


@app.command()
def reset_index():
    with psycopg.connect(DB_CONNECTION_STRING) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document")


if __name__ == "__main__":
    app()
