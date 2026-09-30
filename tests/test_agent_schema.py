"""`agent/schema.py` 的离线用例：不调模型、不联网，专打 tool_calls 解析的各种畸形输入。

这些形状之所以值得逐条盖：Agent 出错最常见的不是"答案不好"，而是"消息结构没预期那样"——
而它只在真机第一次跑才暴露。用例把形状钉死，真跑时只需要确认"真机确实按这个形状来"。
"""

import json

import pytest

from agent.schema import (
    ACTION_FINAL,
    ACTION_INVALID,
    ACTION_TOOL,
    Decision,
    DecisionError,
    ToolCall,
    estimate_budget,
    find_unknown_tools,
    missing_required,
    over_budget,
    parse_arguments,
    parse_message,
    parse_tool_call,
    turn_input_curve,
)


def assistant_message(content=None, tool_calls=None):
    """按 OpenAI 兼容口径造一条 assistant 消息（真机那边是 message.model_dump()，形状一致）。"""
    message = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return message


def tool_call_block(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class TestParseArguments:
    def test_json_string_becomes_dict(self):
        assert parse_arguments('{"query": "格利泽581b"}') == {"query": "格利泽581b"}

    def test_empty_string_and_none_are_empty_arguments(self):
        assert parse_arguments("") == {}
        assert parse_arguments(None) == {}

    def test_dict_passes_through(self):
        assert parse_arguments({"query": "x"}) == {"query": "x"}

    def test_double_encoded_json_is_unwrapped(self):
        assert parse_arguments(json.dumps(json.dumps({"query": "x"}))) == {"query": "x"}

    @pytest.mark.parametrize("bad", ["{not json}", "['a'", "5", '"just a string"'])
    def test_anything_that_is_not_an_object_raises(self, bad):
        with pytest.raises(DecisionError):
            parse_arguments(bad)

    def test_triple_encoded_gives_up_instead_of_looping_forever(self):
        with pytest.raises(DecisionError):
            parse_arguments(json.dumps(json.dumps(json.dumps({"query": "x"}))))

    def test_non_string_non_mapping_type_raises(self):
        with pytest.raises(DecisionError):
            parse_arguments(["query", "x"])


class TestParseToolCall:
    def test_happy_path(self):
        call = parse_tool_call(tool_call_block("search_knowledge_base", '{"query": "x"}', call_id="c42"))

        assert call == ToolCall("c42", "search_knowledge_base", {"query": "x"})

    def test_missing_name_raises_because_nothing_can_be_executed(self):
        with pytest.raises(DecisionError):
            parse_tool_call(tool_call_block("", "{}"))

    def test_missing_id_degrades_to_empty_string_not_crash(self):
        block = tool_call_block("search_knowledge_base", "{}")
        del block["id"]

        assert parse_tool_call(block).call_id == ""

    def test_broken_arguments_error_surfaces_from_the_call(self):
        with pytest.raises(DecisionError):
            parse_tool_call(tool_call_block("search_knowledge_base", "{oops"))


class TestParseMessage:
    def test_tool_calls_win_over_text(self):
        message = assistant_message(content="我查一下", tool_calls=[tool_call_block("t", '{"a": 1}')])

        decision = parse_message(message)
        assert decision.action == ACTION_TOOL
        assert decision.tool_calls == (ToolCall("call_1", "t", {"a": 1}),)
        assert decision.text == "我查一下"
        assert decision.answer is None

    def test_several_calls_in_one_turn_are_all_kept(self):
        message = assistant_message(
            tool_calls=[
                tool_call_block("search_knowledge_base", '{"query": "a"}', call_id="c1"),
                tool_call_block("search_knowledge_base", '{"query": "b"}', call_id="c2"),
            ]
        )

        assert [call.arguments["query"] for call in parse_message(message).tool_calls] == ["a", "b"]

    def test_plain_text_is_a_final_answer(self):
        decision = parse_message(assistant_message(content="  泰国  "))

        assert decision == Decision(ACTION_FINAL, text="泰国")
        assert decision.answer == "泰国"

    def test_blank_content_is_not_an_answer(self):
        assert parse_message(assistant_message(content="   \n ")).action == ACTION_INVALID

    def test_empty_message_is_invalid_rather_than_final(self):
        assert parse_message({}).action == ACTION_INVALID

    def test_invalid_carries_no_answer(self):
        assert parse_message({}).answer is None


class TestToolNameAndArgumentGuards:
    def test_unknown_tool_names_are_reported_in_order(self):
        decision = Decision(
            ACTION_TOOL,
            tool_calls=(
                ToolCall("c1", "search_web", {}),
                ToolCall("c2", "search_knowledge_base", {}),
            ),
        )

        assert find_unknown_tools(decision, {"search_knowledge_base"}) == ["search_web"]

    def test_known_tools_produce_no_complaint(self):
        decision = Decision(ACTION_TOOL, tool_calls=(ToolCall("c1", "search_knowledge_base", {}),))

        assert find_unknown_tools(decision, {"search_knowledge_base"}) == []

    def test_final_answer_has_no_tools_to_check(self):
        assert find_unknown_tools(Decision(ACTION_FINAL, text="泰国"), {"search_knowledge_base"}) == []

    def test_missing_required_argument_is_named(self):
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        call = ToolCall("c1", "search_knowledge_base", {})

        assert missing_required(call, schema) == ["query"]

    def test_empty_string_counts_as_missing(self):
        schema = {"required": ["query"]}
        call = ToolCall("c1", "search_knowledge_base", {"query": ""})

        assert missing_required(call, schema) == ["query"]

    def test_tool_without_required_takes_anything(self):
        assert missing_required(ToolCall("c1", "ping", {}), {"type": "object"}) == []


class TestBudget:
    def test_curve_grows_by_growth_per_turn(self):
        assert turn_input_curve(600, 2600, 4) == [600, 3200, 5800, 8400]

    def test_zero_turns_is_a_config_error_not_a_free_pass(self):
        with pytest.raises(ValueError):
            estimate_budget(max_turns=0)

    def test_totals_sum_the_curve(self):
        estimate = estimate_budget(max_turns=6, base_tokens=600, growth_per_turn=2600, output_per_turn=150)

        assert estimate["input_tokens"] == sum(estimate["input_curve"]) == 42600
        assert estimate["output_tokens"] == 900
        assert estimate["total_tokens"] == 43500

    def test_budget_is_strictly_higher_than_turns_times_unit_price(self):
        """这条断言钉住预算算式的存在理由：`轮数 × 单次单价` 会系统性低估。"""
        estimate = estimate_budget(max_turns=6)
        naive = 6 * estimate["input_curve"][0]

        assert estimate["input_tokens"] > naive

    def test_over_budget_reports_whether_it_fits(self):
        estimate = estimate_budget(max_turns=6)

        assert over_budget(estimate, 50000) == (False, estimate["total_tokens"] - 50000)
        assert over_budget(estimate, 20000) == (True, estimate["total_tokens"] - 20000)
