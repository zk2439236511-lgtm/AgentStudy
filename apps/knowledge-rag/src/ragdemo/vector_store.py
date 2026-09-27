"""
Vector store module.

RAG Flow Step 3: Store
- OpenAIEmbeddings: Converts text to vector embeddings
- InMemoryVectorStore: Stores embeddings for similarity search
"""

import os

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# 阿里云百炼提供 OpenAI 兼容端点，换个 base_url 就能用同一套 SDK
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
# 百炼 embedding 单次请求最多 10 条文本，超过直接报 400
EMBEDDING_BATCH_SIZE = 10


def create_vector_store(
    chunks: list,
    model: str | None = None,
) -> InMemoryVectorStore:
    """
    Create a vector store from document chunks.

    The InMemoryVectorStore is suitable for demos and small datasets.
    For production, consider Chroma, PGVector, or Redis.

    Args:
        chunks: List of Document objects to embed and store
        model: Embedding model to use (defaults to RAG_EMBEDDING_MODEL env var)

    Returns:
        Populated InMemoryVectorStore
    """
    embeddings = OpenAIEmbeddings(
        model=model or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("RAG_BASE_URL", DEFAULT_BASE_URL),
        # LangChain 默认把文本编码成 token 数组发送，百炼只接受字符串，会报 400
        check_embedding_ctx_length=False,
    )

    vector_store = InMemoryVectorStore(embedding=embeddings)
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        vector_store.add_documents(chunks[start : start + EMBEDDING_BATCH_SIZE])
    print(f"Created vector store with {len(chunks)} documents")

    return vector_store
