"""工具层的离线用例：注册、派发、失败处理，以及把检索结果排成给模型读的文本。

这里的 handler 全是假的——真检索要等索引建好（花钱在 embedding）才用得上，
而工具层的行为不该依赖它。
"""

from types import SimpleNamespace

import pytest

from agent.schema import ToolCall
from agent.tools import Tool, ToolRegistry, format_hits, make_search_tool


def echo_tool(name="ping", **overrides):
    fields = {
        "name": name,
        "description": "say it back",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "handler": lambda text: f"echo:{text}",
    }
    fields.update(overrides)
    return Tool(**fields)


def fake_document(text, source="cmrc.pdf", page=2):
    return SimpleNamespace(page_content=text, metadata={"source": source, "page": page})


class TestToolSpec:
    def test_spec_is_the_openai_compatible_shape(self):
        spec = echo_tool().spec()

        assert spec["type"] == "function"
        assert spec["function"]["name"] == "ping"
        assert spec["function"]["parameters"]["required"] == ["text"]

    def test_required_is_read_from_parameters(self):
        assert echo_tool().required() == ["text"]


class TestRegistry:
    def test_duplicate_names_are_rejected_at_assembly_time(self):
        registry = ToolRegistry([echo_tool()])

        with pytest.raises(ValueError):
            registry.register(echo_tool())

    def test_names_and_specs_agree(self):
        registry = ToolRegistry([echo_tool("ping"), echo_tool("pong")])

        assert registry.names() == {"ping", "pong"}
        assert [spec["function"]["name"] for spec in registry.specs()] == ["ping", "pong"]

    def test_empty_registry_describes_itself_without_crashing(self):
        result = ToolRegistry().call(ToolCall("c1", "anything", {}))

        assert result.ok is False
        assert "没有注册任何工具" in result.content


class TestDispatch:
    def test_known_tool_runs_and_returns_string_content(self):
        registry = ToolRegistry([echo_tool()])

        result = registry.call(ToolCall("c1", "ping", {"text": "hi"}))

        assert (result.ok, result.content) == (True, "echo:hi")

    def test_unknown_tool_becomes_model_facing_advice(self):
        registry = ToolRegistry([echo_tool()])

        result = registry.call(ToolCall("c1", "search_web", {}))

        assert result.ok is False
        assert "未知工具" in result.content and "ping" in result.content

    def test_missing_required_argument_is_named(self):
        registry = ToolRegistry([echo_tool()])

        result = registry.call(ToolCall("c1", "ping", {}))

        assert result.ok is False and "text" in result.content

    def test_unexpected_argument_is_reported_not_confused_with_a_handler_bug(self):
        registry = ToolRegistry([echo_tool()])

        result = registry.call(ToolCall("c1", "ping", {"text": "hi", "extra": 1}))

        assert result.ok is False and "不支持的参数" in result.content

    def test_handler_exception_does_not_escape_the_registry(self):
        def boom(text):
            raise RuntimeError("检索库被锁了")

        registry = ToolRegistry([echo_tool(handler=boom)])

        result = registry.call(ToolCall("c1", "ping", {"text": "hi"}))

        assert result.ok is False
        assert "RuntimeError" in result.content and "检索库被锁了" in result.content


class TestFormatHits:
    def test_no_hits_is_an_explicit_empty_statement(self):
        assert format_hits([]) == "没有检索到任何片段。"

    def test_hits_are_numbered_with_distance_and_one_based_page(self):
        hits = [(fake_document("迪韦齐斯是英格兰的一个小镇", page=2), 0.62134)]

        text = format_hits(hits)

        assert text.startswith("[1] 距离 0.6213｜cmrc.pdf 第 3 页")
        assert "迪韦齐斯" in text

    def test_long_passages_are_clipped_with_a_marker(self):
        hits = [(fake_document("结" * 700), 0.4)]

        text = format_hits(hits, limit=600)

        assert "…" in text
        assert text.count("结") == 600

    def test_missing_metadata_falls_back_instead_of_raising(self):
        document = SimpleNamespace(page_content="没有元数据", metadata=None)

        assert "未知来源" in format_hits([(document, 0.9)])


class TestSearchTool:
    def test_handler_calls_search_with_default_k(self):
        seen = []

        def search(query, k):
            seen.append((query, k))
            return [(fake_document("泰国"), 0.5)]

        tool = make_search_tool(search, default_k=8)
        content = tool.handler(query="首都")

        assert seen == [("首都", 8)]
        assert "泰国" in content

    def test_k_from_the_model_wins_over_the_default(self):
        captured = {}

        def search(query, k):
            captured["k"] = k
            return []

        make_search_tool(search, default_k=8).handler(query="x", k=3)

        assert captured["k"] == 3

    def test_zero_k_is_treated_as_missing_not_as_no_results(self):
        """钉住这个选择：`k=0` 不是一个有意义的检索请求，宁可回落到默认值。"""
        captured = {}

        def search(query, k):
            captured["k"] = k
            return []

        make_search_tool(search, default_k=8).handler(query="x", k=0)

        assert captured["k"] == 8

    def test_registry_end_to_end_through_the_tool_spec(self):
        def search(query, k):
            return [(fake_document("雷纳·库尔特·萨克斯"), 0.7)]

        registry = ToolRegistry([make_search_tool(search)])

        result = registry.call(ToolCall("c1", "search_knowledge_base", {"query": "发明者"}))

        assert result.ok is True and "[1] 距离 0.7" in result.content
