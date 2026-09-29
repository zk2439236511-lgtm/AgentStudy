"""持久化入库的去重逻辑测试。

Chroma 与真实 PDF 解析都换成替身：这里只验"内容没变就跳过 embedding"这套账目逻辑，
不建库、不联网、不花钱。
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ragdemo import ingest


class FakeStore:
    """记录调用的假向量库，用来断言有没有重新写入。"""

    def __init__(self):
        self.added = []
        self.deleted = []

    def delete(self, where=None):
        self.deleted.append(where)

    def add_documents(self, docs):
        self.added.extend(docs)


@pytest.fixture
def fake_pdf(tmp_path):
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"first version content")
    return pdf


@pytest.fixture
def wired(tmp_path, monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(ingest, "load_vector_store", lambda *args, **kwargs: store)
    monkeypatch.setattr(ingest, "create_embeddings", lambda: object())
    monkeypatch.setattr(
        ingest,
        "load_and_chunk_pdf",
        lambda path, *args, **kwargs: [
            Document(
                page_content=f"chunk of {Path(path).name}",
                metadata={"source": Path(path).name},
            )
        ],
    )
    conn = ingest.init_registry(tmp_path / "registry.db")
    yield store, conn
    conn.close()


def run_ingest(fake_pdf, conn, tmp_path):
    return ingest.ingest_directory(
        fake_pdf.parent, conn, persist_directory=tmp_path / "chroma"
    )


def test_first_run_embeds_the_file(wired, fake_pdf, tmp_path):
    store, conn = wired
    result = run_ingest(fake_pdf, conn, tmp_path)

    assert result["ingested"] == ["fake.pdf"]
    assert result["skipped"] == []
    assert len(store.added) == 1


def test_second_run_skips_unchanged_file(wired, fake_pdf, tmp_path):
    store, conn = wired
    run_ingest(fake_pdf, conn, tmp_path)
    added_after_first = len(store.added)
    deleted_after_first = len(store.deleted)

    second = run_ingest(fake_pdf, conn, tmp_path)

    assert second["ingested"] == []
    assert second["skipped"] == ["fake.pdf"]
    # 没有重新 embedding，也没有动过旧向量
    assert len(store.added) == added_after_first
    assert len(store.deleted) == deleted_after_first


def test_changed_file_is_reingested_after_deleting_old_vectors(
    wired, fake_pdf, tmp_path
):
    store, conn = wired
    run_ingest(fake_pdf, conn, tmp_path)

    fake_pdf.write_bytes(b"second version content")
    third = run_ingest(fake_pdf, conn, tmp_path)

    assert third["ingested"] == ["fake.pdf"]
    assert third["chunks_added"] == 1
    # 每次入库前都会先清掉该文件的旧向量（首次入库时删的是空集）
    assert store.deleted == [{"source": "fake.pdf"}, {"source": "fake.pdf"}]
    assert len(store.added) == 2


def test_registry_stores_hash_and_chunk_count(wired, fake_pdf, tmp_path):
    store, conn = wired
    run_ingest(fake_pdf, conn, tmp_path)

    row = conn.execute(
        "SELECT hash, chunks, ingested_at FROM documents WHERE filename = ?",
        ("fake.pdf",),
    ).fetchone()

    assert row[0] == ingest.file_sha256(fake_pdf)
    assert row[1] == 1
    assert row[2]


def test_same_bytes_same_hash_different_bytes_different_hash(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"same")
    b.write_bytes(b"same")

    assert ingest.file_sha256(a) == ingest.file_sha256(b)

    b.write_bytes(b"changed")
    assert ingest.file_sha256(a) != ingest.file_sha256(b)


def test_pdf_without_text_is_flagged_empty_and_not_registered(wired, fake_pdf, tmp_path, monkeypatch):
    """扫描件一个文字都抽不出来：必须进 empty，且不写库、不进登记表。"""
    store, conn = wired
    monkeypatch.setattr(ingest, "load_and_chunk_pdf", lambda path, *a, **k: [])

    result = run_ingest(fake_pdf, conn, tmp_path)

    assert result["empty"] == ["fake.pdf"]
    assert result["ingested"] == []
    assert result["chunks_added"] == 0
    assert store.added == []
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
