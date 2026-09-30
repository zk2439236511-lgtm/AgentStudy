"""usage.py 的离线用例：token 用量到底从哪个字段来、按游标切得对不对。

这几步一旦错，报出来的"这一轮花了多少额度"就是假的，而且假得很像真的。
真机探针已经确认百炼 compatible-mode 两处字段都有值（见 usage.py 模块注释），
这里用同样的结构造替身，不打 API。
"""

import sys
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

sys.path.insert(0, str(Path(__file__).resolve().parent))

from usage import TokenUsageCollector, extract_usage, sum_usage


def chat_result(input_tokens: int, output_tokens: int) -> LLMResult:
    """线上真实形状：AIMessage 带 usage_metadata（LangChain 标准口径）。"""
    message = AIMessage(
        content="答案",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestExtractUsage:
    def test_reads_standard_usage_metadata(self):
        usage = extract_usage(chat_result(23, 4))
        assert usage == {"input_tokens": 23, "output_tokens": 4, "total_tokens": 27}

    def test_falls_back_to_openai_token_usage(self):
        """没有 usage_metadata 时退回 llm_output["token_usage"]，字段名是 OpenAI 口径的。"""
        result = LLMResult(
            generations=[[Generation(text="答案")]],
            llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}},
        )
        assert extract_usage(result) == {"input_tokens": 100, "output_tokens": 7, "total_tokens": 107}

    def test_missing_everywhere_returns_none_not_zero(self):
        """拿不到用量宁可返回 None：记 0 会被读成"这次调用没花钱"。"""
        assert extract_usage(LLMResult(generations=[[Generation(text="答案")]])) is None
        assert extract_usage(LLMResult(generations=[])) is None


class TestCollector:
    def test_slices_usage_per_question_with_cursor(self):
        """游标切片的意义在这里：一道题期间发生了几次生成，就算在这道题头上。"""
        collector = TokenUsageCollector()
        first_cursor = collector.snapshot()
        collector.on_llm_end(chat_result(1000, 30))
        assert collector.usage_since(first_cursor) == {
            "llm_calls": 1,
            "input_tokens": 1000,
            "output_tokens": 30,
            "total_tokens": 1030,
        }

        second_cursor = collector.snapshot()
        collector.on_llm_end(chat_result(2000, 50))
        collector.on_llm_end(chat_result(500, 10))
        assert collector.usage_since(second_cursor)["llm_calls"] == 2
        assert collector.usage_since(second_cursor)["input_tokens"] == 2500
        # 第一道题的游标仍然只看得到它自己那一次
        assert collector.usage_since(first_cursor)["input_tokens"] == 3500

    def test_refusal_path_records_nothing(self):
        """阈值拦下时压根不调模型：回调不触发，这道题的用量就是全 0。"""
        collector = TokenUsageCollector()
        collector.on_llm_end(LLMResult(generations=[[Generation(text="不该发生")]]))
        assert collector.records == []
        assert collector.usage_since(0) == {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def test_sum_usage_of_empty_is_zeros(self):
        assert sum_usage([]) == {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
