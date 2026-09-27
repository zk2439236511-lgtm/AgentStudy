"""
持久化入库模块（阶段 3）。

RAG Flow Step 3 的进阶版：向量存到磁盘，索引建一次、重启还能问。
- Chroma: 本地文件向量库，进程退出后数据仍在
- documents 登记表: 记录每个 PDF 的内容哈希，内容没变就跳过 embedding
"""

import hashlib
import sqlite3
from pathlib import Path

from langchain_chroma import Chroma

from ragdemo.document_loader import load_and_chunk_pdf
from ragdemo.vector_store import EMBEDDING_BATCH_SIZE, create_embeddings

COLLECTION = "pdf_chunks"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "registry.db"


def init_registry(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the ingestion registry."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            filename    TEXT    PRIMARY KEY,
            hash        TEXT    NOT NULL,
            chunks      INTEGER NOT NULL,
            ingested_at TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def stored_hash(conn: sqlite3.Connection, filename: str) -> str | None:
    row = conn.execute(
        "SELECT hash FROM documents WHERE filename = ?", (filename,)
    ).fetchone()
    return row[0] if row else None


def load_vector_store(
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    embeddings=None,
) -> Chroma:
    """Open the on-disk Chroma index without re-embedding anything."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or create_embeddings(),
        persist_directory=str(persist_directory),
    )


def count_vectors(store: Chroma) -> int:
    return store._collection.count()


def ingest_directory(
    docs_dir: str | Path,
    conn: sqlite3.Connection,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    """
    Embed PDFs that are new or changed since the last run.

    A file is skipped when its SHA-256 matches the registry, so its existing
    vectors are reused instead of being paid for again.
    """
    result = {"ingested": [], "skipped": [], "chunks_added": 0}
    store = None

    for pdf_path in sorted(Path(docs_dir).glob("*.pdf")):
        filename = pdf_path.name
        current_hash = file_sha256(pdf_path)

        if stored_hash(conn, filename) == current_hash:
            print(f"Skip {filename}: 内容未变，沿用磁盘上的索引")
            result["skipped"].append(filename)
            continue

        if store is None:
            store = load_vector_store(persist_directory, create_embeddings())

        chunks = load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)
        # 文件内容变了：先删掉它的旧向量，避免新旧版本混在检索结果里
        store.delete(where={"source": filename})
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            store.add_documents(chunks[start : start + EMBEDDING_BATCH_SIZE])

        conn.execute(
            """INSERT INTO documents (filename, hash, chunks, ingested_at)
               VALUES (?, ?, ?, datetime('now', 'localtime'))
               ON CONFLICT(filename) DO UPDATE SET
                   hash = excluded.hash,
                   chunks = excluded.chunks,
                   ingested_at = excluded.ingested_at""",
            (filename, current_hash, len(chunks)),
        )
        conn.commit()

        result["ingested"].append(filename)
        result["chunks_added"] += len(chunks)
        print(f"Ingested {filename}: {len(chunks)} chunks")

    return result
