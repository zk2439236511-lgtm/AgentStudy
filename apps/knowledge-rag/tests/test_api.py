"""知识库 HTTP 接口的测试。

全部用替身：假向量库、假 RAG chain、临时登记表与临时上传目录——
不建索引、不打模型、不花钱。
"""

from pathlib import Path
import threading

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from ragdemo import api

client = TestClient(api.app)


class FakeStore:
    def __init__(self, count=52):
        self._collection = type("C", (), {"count": lambda self: count})()


class FakeChain:
    """假链路：把 create_rag_chain_with_sources 返回对象的契约（answer/sources/refused）演出来。"""

    def __init__(self, refused=False, distance=0.6789):
        self.received = None
        self.refused = refused
        self.distance = distance

    def invoke(self, question):
        self.received = question
        long_text = "x" * 500
        answer = (
            "知识库里没有与这个问题足够相关的内容，不作答。（最接近的片段距离 1.35，拒答阈值 1.00）"
            if self.refused
            else "多头注意力把 query/key/value 投影 h 次后并行计算。"
        )
        return {
            "answer": answer,
            "source_documents": [
                Document(
                    page_content=long_text,
                    metadata={"source": "sample.pdf", "page": 3, "score": self.distance},
                )
            ],
            "refused": self.refused,
        }


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """每个用例都换一套干净的 globals / 登记表 / 上传目录 / 假依赖。"""
    api._chain = None

    db_path = tmp_path / "registry.db"
    monkeypatch.setattr(api, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(api, "DOCS_DIR", tmp_path / "documents")
    monkeypatch.setattr(api, "load_vector_store", lambda *a, **k: FakeStore())

    yield db_path

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


def test_status_works_from_another_worker_thread(isolated_state):
    """真机 uvicorn 会把同步接口丢进线程池，连接不能跨线程复用。

    TestClient 始终在同一个线程里跑，所以它测不出这个 bug——
    这里连开两个线程：第一个把连接建出来，第二个必须还能用。
    """
    seed_document(isolated_state)
    errors = []

    def call_from_new_thread():
        try:
            api.get_status()
        except Exception as cause:
            errors.append(cause)

    for _ in range(2):
        thread = threading.Thread(target=call_from_new_thread)
        thread.start()
        thread.join()

    assert errors == [], f"跨线程调用失败：{errors[-1]!r}"


def test_list_documents_reads_registry(isolated_state):
    seed_document(isolated_state, filename="attention.pdf", chunks=30)
    data = client.get("/kb/documents").json()

    assert data == [
        {"filename": "attention.pdf", "chunks": 30, "ingested_at": "2026-09-27 10:34:24"}
    ]


def test_upload_accepts_txt(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_directory",
        lambda *a, **k: {"ingested": ["notes.txt"], "skipped": [], "empty": [], "chunks_added": 3},
    )

    response = client.post(
        "/kb/documents",
        files={"file": ("notes.txt", "中文笔记。".encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "notes.txt"
    assert (api.DOCS_DIR / "notes.txt").read_text(encoding="utf-8") == "中文笔记。"


def test_upload_rejects_unsupported_suffix():
    response = client.post(
        "/kb/documents", files={"file": ("resume.docx", b"data", "application/octet-stream")}
    )

    assert response.status_code == 400
    assert ".txt" in response.json()["detail"]


def test_upload_pdf_saves_file_and_triggers_ingest(monkeypatch, tmp_path):
    calls = {}

    def fake_ingest(docs_dir, conn, persist_directory, **kwargs):
        calls["docs_dir"] = Path(docs_dir)
        return {"ingested": ["new.pdf"], "skipped": [], "empty": [], "chunks_added": 7}

    monkeypatch.setattr(api, "ingest_directory", fake_ingest)

    response = client.post(
        "/kb/documents", files={"file": ("new.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )

    assert response.status_code == 201
    assert response.json() == {
        "filename": "new.pdf",
        "ingested": True,
        "skipped": False,
        "empty": False,
        "chunks_added": 7,
    }
    # 文件真的落到 documents 目录里了，不是只回了个成功
    assert (api.DOCS_DIR / "new.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_upload_scanned_pdf_is_reported_as_empty(monkeypatch):
    """一个字都没抽出来时不能伪装成入库成功。"""
    monkeypatch.setattr(
        api,
        "ingest_directory",
        lambda *a, **k: {"ingested": ["scan.pdf"], "skipped": [], "empty": ["scan.pdf"], "chunks_added": 0},
    )

    response = client.post(
        "/kb/documents", files={"file": ("scan.pdf", b"%PDF-1.4 scanned", "application/pdf")}
    )

    assert response.status_code == 201
    assert response.json()["empty"] is True
    assert response.json()["chunks_added"] == 0


def test_upload_strips_directory_from_filename(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_directory",
        lambda *a, **k: {"ingested": [], "skipped": ["passwd.pdf"], "empty": [], "chunks_added": 0},
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
    assert data["refused"] is False
    source = data["sources"][0]
    assert source["filename"] == "sample.pdf"
    assert source["page"] == 3
    # 接口原样透出 Chroma 的向量距离，不做单调变换包装
    assert source["distance"] == 0.679
    assert len(source["excerpt"]) == 300


def test_ask_surfaces_refusal_without_hiding_the_sources(monkeypatch):
    """拒答时也要把"最近的一条是什么、离得多远"透出去，用户才判断得了是没查全还是真没有。"""
    chain = FakeChain(refused=True, distance=1.35)
    monkeypatch.setattr(api, "create_rag_chain_with_sources", lambda store: chain)

    data = client.post("/kb/ask", json={"question": "明天天气怎么样"}).json()

    assert data["refused"] is True
    assert "不作答" in data["answer"]
    assert data["sources"][0]["distance"] == 1.35


def test_ask_blank_question_returns_400(monkeypatch):
    monkeypatch.setattr(
        api, "create_rag_chain_with_sources", lambda store: FakeChain()
    )
    response = client.post("/kb/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_missing_question_returns_422():
    response = client.post("/kb/ask", json={})
    assert response.status_code == 422
