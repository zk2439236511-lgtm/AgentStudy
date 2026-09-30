"""
RAG chain module using LangChain Expression Language (LCEL).

RAG Flow Step 4-5: Retrieve → Generate
- Retriever: Searches vector store for relevant chunks
- Prompt: Combines context with user question
- LLM: Generates answer based on augmented prompt

The prompt template used is:
    Answer the question based only on the following context.
    If the context doesn't contain enough information to answer, say so.

    Context:
    {context}

    Question: {question}

    Answer:
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
import os


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CHAT_MODEL = "qwen-plus"
# 检索几条由 evals/ 的网格实验决定：k=8 的 Hit@K 与 Recall@K 全面优于 k=4
DEFAULT_K = int(os.getenv("RAG_TOP_K", "8"))
# 拒答工作点，来自 evals/results/2026-09-30-chroma-threshold.md：CMRC 金标集上
# 实测 30 道构造负样本拒掉 24 道、200 道可回答题只误伤 3 道。
# 口径是 Chroma 的平方欧氏距离（越小越相关）；换向量库、换 embedding 模型都得重扫。
DEFAULT_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "1.0"))


def _build_llm(model: str | None = None) -> ChatOpenAI:
    """Build the chat model. Defaults to Bailian's qwen-plus via env override."""
    return ChatOpenAI(
        model=model or os.getenv("RAG_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("RAG_BASE_URL", DEFAULT_BASE_URL),
        temperature=1,
    )


# RAG prompt template - instructs the model to answer based on context
RAG_TEMPLATE = """Answer the question based only on the following context.
If the context doesn't contain enough information to answer, say so.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list) -> str:
    """Format retrieved documents into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def should_refuse(hits: list, max_distance: float | None) -> bool:
    """要不要在检索层就拒掉这个问题。hits 是 (Document, 分数) 列表，分数是 Chroma 距离。

    三个"放行"都是有意的：阈值关掉（内存库的分数是余弦相似度，不是距离，拿距离线去卡
    它等于乱拒）、检索退回无分数路径（口径都不知道就没法判）、以及压线（条件是「大于」）。
    取向是宁可少拒也不错拒——把本来能答的问题挡在门外更伤可用性。
    """
    if not hits:
        return True
    if max_distance is None:
        return False
    score = hits[0][1]
    if score is None:
        return False
    return float(score) > max_distance


def refusal_answer(hits: list, max_distance: float) -> str:
    """拒答文案：把"最近的一条离得多远、线画在哪"原样给用户看，不含糊。"""
    if not hits:
        return "知识库里没有检索到任何片段（索引可能是空的），不作答。"
    score = float(hits[0][1])
    return (
        f"知识库里没有与这个问题足够相关的内容，不作答。"
        f"（最接近的片段距离 {score:.2f}，拒答阈值 {max_distance:.2f}）"
    )


def create_rag_chain(
    vector_store: VectorStore,
    model: str | None = None,
    k: int = DEFAULT_K,
):
    """
    Create a RAG chain using LCEL (LangChain Expression Language).

    The chain follows this flow:
    1. question → retriever → relevant documents
    2. documents → format_docs → context string
    3. {context, question} → prompt → augmented prompt
    4. augmented prompt → LLM → response
    5. response → parser → final answer string

    Args:
        vector_store: The vector store to retrieve from
        model: OpenAI chat model to use
        k: Number of documents to retrieve

    Returns:
        LCEL chain that can be invoked with a question string
    """
    # Create retriever from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    # Create LLM (gpt-5 models only support temperature=1)
    llm = _build_llm(model)

    # Build the RAG chain using LCEL
    # The | operator chains components together
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def create_rag_chain_with_sources(
    vector_store: VectorStore,
    model: str | None = None,
    k: int = DEFAULT_K,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
):
    """
    Create a RAG chain that returns both the answer and retrieved documents.

    Useful for debugging and understanding which documents contributed to the answer.
    Documents include similarity scores in their metadata when available.

    Args:
        vector_store: The vector store to retrieve from
        model: OpenAI chat model to use
        k: Number of documents to retrieve
        max_distance: 拒答线（Chroma 距离，越小越相关）；传 None 表示不启用

    Returns:
        Callable that returns
        {"answer": str, "source_documents": list[Document], "refused": bool}
    """
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    llm = _build_llm(model)

    def retrieve_with_scores(question: str) -> list[tuple]:
        """Retrieve (Document, score) pairs; score is None when the store can't give one."""
        try:
            # Use similarity_search_with_score to get (doc, score) tuples
            return list(vector_store.similarity_search_with_score(question, k=k))
        except Exception:
            # Fallback if similarity_search_with_score not available
            return [(doc, None) for doc in vector_store.similarity_search(question, k=k)]

    def attach_scores(hits: list[tuple]) -> list:
        docs = []
        for doc, score in hits:
            if score is not None:
                doc.metadata["score"] = score
            docs.append(doc)
        return docs

    def invoke(question: str) -> dict:
        """Run the RAG chain and return answer with sources."""
        hits = retrieve_with_scores(question)
        docs = attach_scores(hits)

        if should_refuse(hits, max_distance):
            return {
                "answer": refusal_answer(hits, max_distance),
                "source_documents": docs,
                "refused": True,
            }

        # Build the prompt and get answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": format_docs(docs), "question": question})

        return {"answer": answer, "source_documents": docs, "refused": False}

    # Return a simple callable wrapper with invoke method
    class ChainWrapper:
        def invoke(self, question: str) -> dict:
            return invoke(question)

    return ChainWrapper()


def print_source_documents(docs: list[Document], max_chars: int = 200) -> None:
    """
    Print retrieved source documents for debugging.

    Args:
        docs: List of retrieved Document objects (may include similarity scores in metadata)
        max_chars: Maximum characters to show from each document
    """
    print(f"\n--- Retrieved {len(docs)} documents ---")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("score")
        content_preview = doc.page_content[:max_chars].replace("\n", " ")
        if len(doc.page_content) > max_chars:
            content_preview += "..."

        # Show similarity score if available
        score_str = f", Score: {score:.3f}" if score is not None else ""
        print(f"\n[{i}] Source: {source}, Page: {page}{score_str}")
        print(f"    {content_preview}")
    print("-----------------------------------\n")
