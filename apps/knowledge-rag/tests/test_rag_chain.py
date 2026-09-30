"""Tests for RAG chain module."""

from unittest.mock import patch
from langchain_core.documents import Document
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class FakeVectorStore:
    """只实现链路真正会调的那个方法：带分数的检索。"""

    def __init__(self, results: list[tuple[Document, float]]):
        self.results = results
        self.calls = 0

    def similarity_search_with_score(self, query: str, k: int = 4):
        self.calls += 1
        return self.results[:k]


def make_doc(text: str = "知识库里的一段内容") -> Document:
    return Document(page_content=text, metadata={"source": "笔记.txt", "page": None})


def tripwire_llm():
    """一个"不该被用到"的模型对象。

    链路是在 invoke 里才把 prompt | llm | parser 拼起来的，而 LangChain 拼接非 Runnable
    对象会当场抛错。所以拒答用例只要没炸，就等于证明了根本没去敲模型的门。
    """
    return object()


def answering_llm(content: str = "模型给的答案"):
    return FakeMessagesListChatModel(responses=[AIMessage(content=content)])


def build_chain(store, llm, max_distance):
    from ragdemo.rag_chain import create_rag_chain_with_sources

    with patch("ragdemo.rag_chain.ChatOpenAI", return_value=llm):
        return create_rag_chain_with_sources(store, k=4, max_distance=max_distance)


class TestDistanceThresholdRefusal:
    """检索层阈值拒答：阈值来自 evals/results/2026-09-30-chroma-threshold.md 的实测工作点。"""

    def test_should_refuse_when_top1_distance_is_over_the_line(self):
        from ragdemo.rag_chain import should_refuse

        assert should_refuse([(make_doc(), 1.0)], 1.0) is False  # 压线放行，条件是「大于」
        assert should_refuse([(make_doc(), 1.01)], 1.0) is True

    def test_should_refuse_when_nothing_was_retrieved(self):
        from ragdemo.rag_chain import should_refuse

        assert should_refuse([], 1.0) is True

    def test_should_not_refuse_when_threshold_is_disabled(self):
        """内存库的分数是余弦相似度不是距离，这时候必须能整体关掉，不能拿距离线去卡它。"""
        from ragdemo.rag_chain import should_refuse

        assert should_refuse([(make_doc(), 0.2)], None) is False

    def test_should_not_refuse_when_scores_are_unavailable(self):
        """检索退回无分数路径时判不了口径，取向是放行而不是全部拒答。"""
        from ragdemo.rag_chain import should_refuse

        assert should_refuse([(make_doc(), None)], 1.0) is False

    def test_refused_answer_never_reaches_the_model(self):
        store = FakeVectorStore([(make_doc(), 1.35)])
        chain = build_chain(store, tripwire_llm(), max_distance=1.0)

        result = chain.invoke("知识库里根本没有的问题")

        assert result["refused"] is True
        assert "1.35" in result["answer"]
        assert "1.00" in result["answer"]
        # 检索到的东西照样返回，用户能看见"最近的一条是什么、离得多远"
        assert len(result["source_documents"]) == 1

    def test_empty_index_refuses(self):
        chain = build_chain(FakeVectorStore([]), tripwire_llm(), max_distance=1.0)

        result = chain.invoke("任何问题")

        assert result["refused"] is True
        assert "没有检索到" in result["answer"]

    def test_close_enough_top1_still_gets_an_answer(self):
        store = FakeVectorStore([(make_doc(), 0.4)])
        chain = build_chain(store, answering_llm(), max_distance=1.0)

        result = chain.invoke("知识库里有答案的问题")

        assert result["refused"] is False
        assert result["answer"] == "模型给的答案"

    def test_disabled_threshold_answers_even_for_far_retrievals(self):
        chain = build_chain(
            FakeVectorStore([(make_doc(), 3.9)]), answering_llm(), max_distance=None
        )

        result = chain.invoke("问题")

        assert result["refused"] is False
        assert result["answer"] == "模型给的答案"

    def test_threshold_defaults_to_the_measured_operating_point(self):
        """默认值就是那根实测线，改它得走环境变量，别散着改数字。"""
        from ragdemo.rag_chain import DEFAULT_MAX_DISTANCE

        assert DEFAULT_MAX_DISTANCE == 1.0


class TestRagChain:
    """Tests for the RAG chain module."""

    def test_format_docs(self):
        """Test document formatting function."""
        from ragdemo.rag_chain import format_docs

        docs = [
            Document(page_content="First document"),
            Document(page_content="Second document"),
        ]

        result = format_docs(docs)

        assert "First document" in result
        assert "Second document" in result
        assert "\n\n" in result  # Documents should be separated

    def test_format_docs_empty_list(self):
        """Test formatting empty document list."""
        from ragdemo.rag_chain import format_docs

        result = format_docs([])

        assert result == ""

    def test_create_rag_chain_returns_runnable(self, mock_embeddings, mock_llm):
        """Test that create_rag_chain returns a runnable chain."""
        from ragdemo.rag_chain import create_rag_chain
        from ragdemo.vector_store import create_vector_store

        chunks = [
            Document(page_content="The Transformer uses attention."),
        ]

        with patch('ragdemo.vector_store.OpenAIEmbeddings', return_value=mock_embeddings):
            vector_store = create_vector_store(chunks)

        with patch('ragdemo.rag_chain.ChatOpenAI', return_value=mock_llm):
            chain = create_rag_chain(vector_store)

        assert chain is not None
        assert hasattr(chain, 'invoke')

    def test_rag_template_contains_placeholders(self):
        """Test that RAG template has required placeholders."""
        from ragdemo.rag_chain import RAG_TEMPLATE

        assert "{context}" in RAG_TEMPLATE
        assert "{question}" in RAG_TEMPLATE

    def test_create_rag_chain_with_custom_k(self, mock_embeddings, mock_llm):
        """Test creating RAG chain with custom retrieval count."""
        from ragdemo.rag_chain import create_rag_chain
        from ragdemo.vector_store import create_vector_store

        chunks = [
            Document(page_content=f"Document {i}") for i in range(10)
        ]

        with patch('ragdemo.vector_store.OpenAIEmbeddings', return_value=mock_embeddings):
            vector_store = create_vector_store(chunks)

        with patch('ragdemo.rag_chain.ChatOpenAI', return_value=mock_llm):
            chain = create_rag_chain(vector_store, k=2)

        assert chain is not None
