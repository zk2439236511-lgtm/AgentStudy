"""
知识库 HTTP 接口（阶段 4）。

CLI 只能自己敲命令，这一层把它变成"能被网页调用的服务"：
- GET  /kb/status      索引现状（几个文件、几个向量）
- GET  /kb/documents   已入库文件列表（读登记表）
- POST /kb/documents   上传 PDF → 解析入库
- POST /kb/ask         提问 → 答案 + 来源出处
"""

import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ragdemo.ingest import (
    DEFAULT_DB_PATH,
    DEFAULT_PERSIST_DIR,
    count_vectors,
    ingest_directory,
    init_registry,
    load_vector_store,
)
from ragdemo.rag_chain import create_rag_chain_with_sources

APP_DIR = Path(__file__).resolve().parents[2]
load_dotenv(APP_DIR / ".env")

DOCS_DIR = APP_DIR / "documents"

app = FastAPI(title="个人知识库")

_registry = None
_chain = None


def get_registry():
    global _registry
    if _registry is None:
        _registry = init_registry(DEFAULT_DB_PATH)
    return _registry


def get_chain():
    """懒加载：第一次提问时才打开磁盘索引，启动服务不用等它。"""
    global _chain
    if _chain is None:
        _chain = create_rag_chain_with_sources(load_vector_store(DEFAULT_PERSIST_DIR))
    return _chain


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/kb/status")
def get_status():
    rows = get_registry().execute("SELECT COUNT(*) FROM documents").fetchone()
    return {"files": rows[0], "vectors": count_vectors(load_vector_store(DEFAULT_PERSIST_DIR))}


@app.get("/kb/documents")
def list_documents():
    rows = get_registry().execute(
        "SELECT filename, chunks, ingested_at FROM documents ORDER BY ingested_at DESC"
    ).fetchall()
    return [{"filename": r[0], "chunks": r[1], "ingested_at": r[2]} for r in rows]


@app.post("/kb/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    # 只取文件名部分，挡掉 "../../etc/passwd" 这类带路径的文件名
    safe_name = Path(file.filename or "").name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只接受 PDF 文件")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = DOCS_DIR / safe_name
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    result = ingest_directory(DOCS_DIR, get_registry(), DEFAULT_PERSIST_DIR)
    return {
        "filename": safe_name,
        "ingested": safe_name in result["ingested"],
        "skipped": safe_name in result["skipped"],
        # empty=True 表示这个 PDF 一个字都没抽出来（常见于扫描件），别当成入库成功
        "empty": safe_name in result["empty"],
        "chunks_added": result["chunks_added"],
    }


@app.post("/kb/ask")
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    result = get_chain().invoke(question)
    sources = [
        {
            "filename": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            # Chroma 的 score 是向量距离（越小越相关）。这里不做任何包装：
            # 1/(1+d) 之类只是单调变换，没有校准过，不该对外称作"相关度"
            "distance": round(float(doc.metadata.get("score", -1)), 3),
            "excerpt": doc.page_content[:300],
        }
        for doc in result["source_documents"]
    ]
    return {"answer": result["answer"], "sources": sources}
