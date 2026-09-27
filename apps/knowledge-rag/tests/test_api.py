"""知识库 HTTP 接口的测试。

全部用替身：假向量库、假 RAG chain、临时登记表与临时上传目录——
不建索引、不打模型、不花钱。
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from ragdemo import api

client = TestClient(api.app)


class FakeStore:
    def __init__(self, count=52):
        self._collection = type("C", (), {"count": lambda self: count})()


class FakeChain:
    def __init__(self):
        self.received = None

    def invoke(self, question):
        self.received = question
        long_text = "x" * 500
        return {
            "answer": "多头注意力把 query/key/value 投影 h 次后并行计算。",
            "source_documents": [
                Document(
                    page_content=long_text,
                    metadata={"source": "sample.pdf", "page": 3, "score": 0.6789},
                )
            ],
        }


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """每个用例都换一套干净的 globals / 登记表 / 上传目录 / 假依赖。"""
    api._registry = None
    api._chain = None

    db_path = tmp_path / "registry.db"
    monkeypatch.setattr(api, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(api, "DOCS_DIR", tmp_path / "documents")
    monkeypatch.setattr(api, "load_vector_store", lambda *a, **k: FakeStore())

    yield db_path

    api._registry = None
    api._chain = None


def seed_document(db_path, filename="sample.pdf", chunks=52):
    conn = api.init_registry(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO documents (filename, hash, chunks, ingested_at)
           VALUES (?, 'deadbeef', ?, '2026-09-27 10:34:24')""",
        (filename, chunks),
    )
    conn.commit()
    conn.close()


def test_status_returns_file_and_vector_counts(isolated_state):
    seed_document(isolated_state)
    data = client.get("/kb/status").json()

    assert data == {"files": 1, "vectors": 52}


def test_list_documents_reads_registry(isolated_state):
    seed_document(isolated_state, filename="attention.pdf", chunks=30)
    data = client.get("/kb/documents").json()

    assert data == [
        {"filename": "attention.pdf", "chunks": 30, "ingested_at": "2026-09-27 10:34:24"}
    ]


def test_upload_rejects_non_pdf():
    response = client.post(
        "/kb/documents", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_upload_pdf_saves_file_and_triggers_ingest(monkeypatch, tmp_path):
    calls = {}

    def fake_ingest(docs_dir, conn, persist_directory, **kwargs):
        calls["docs_dir"] = Path(docs_dir)
        return {"ingested": ["new.pdf"], "skipped": [], "chunks_added": 7}

    monkeypatch.setattr(api, "ingest_directory", fake_ingest)

    response = client.post(
        "/kb/documents", files={"file": ("new.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )

    assert response.status_code == 201
    assert response.json() == {
        "filename": "new.pdf",
        "ingested": True,
        "skipped": False,
        "chunks_added": 7,
    }
    # 文件真的落到 documents 目录里了，不是只回了个成功
    assert (api.DOCS_DIR / "new.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_upload_strips_directory_from_filename(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_directory",
        lambda *a, **k: {"ingested": [], "skipped": ["passwd.pdf"], "chunks_added": 0},
    )
    response = client.post(
        "/kb/documents",
        files={"file": ("../../etc/passwd.pdf", b"data", "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "passwd.pdf"
    assert not (api.DOCS_DIR.parent / "passwd.pdf").exists()


def test_ask_returns_answer_with_sources(monkeypatch):
    chain = FakeChain()
    monkeypatch.setattr(api, "create_rag_chain_with_sources", lambda store: chain)

    data = client.post("/kb/ask", json={"question": "What is multi-head attention?"}).json()

    assert chain.received == "What is multi-head attention?"
    assert data["answer"].startswith("多头注意力")
    source = data["sources"][0]
    assert source["filename"] == "sample.pdf"
    assert source["page"] == 3
    # 距离 0.6789 换算成相关度：1 / (1 + 0.6789)
    assert source["relevance"] == 0.596
    assert len(source["excerpt"]) == 300


def test_ask_blank_question_returns_400(monkeypatch):
    monkeypatch.setattr(
        api, "create_rag_chain_with_sources", lambda store: FakeChain()
    )
    response = client.post("/kb/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_missing_question_returns_422():
    response = client.post("/kb/ask", json={})
    assert response.status_code == 422
