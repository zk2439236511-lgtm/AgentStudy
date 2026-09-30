"""主循环的离线用例：用一串脚本化的假模型回复，把每条停止路径都单独走一遍。

模型调用是注入点，所以这里一次真调用都没有；而 `role`/`tool_call_id` 这些结构断言
之所以值得写，是因为它们错了不会报错、只会让真机的第二轮请求被 API 拒掉——
最难查的那类问题。
"""

from agent.loop import (
    STOP_ANSWERED,
    STOP_BUDGET,
    STOP_INVALID,
    STOP_MAX_TURNS,
    STOP_SCHEMA_ERROR,
    ModelReply,
    run_agent,
    summarize_run,
    with_call_ids,
)
from agent.tools import Tool, ToolRegistry


def tool_message(name, arguments, content="结果", call_id="call_1", extra=None):
    message = {"role": "assistant", "content": None}
    if extra is not None:
        message["content"] = extra
    message["tool_calls"] = [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    ]
    return message


def final_message(content):
    return {"role": "assistant", "content": content}


def scripted(replies):
    """按脚本吐回复，并记下每次收到的 messages/tools——用来断言历史确实被喂回去了。"""
    remaining = list(replies)
    seen = []

    def call_model(messages, tools):
        seen.append({"messages": [dict(m) for m in messages], "tools": list(tools)})
        return ModelReply(message=remaining.pop(0), usage={"input_tokens": 100, "output_tokens": 20})

    call_model.seen = seen
    call_model.calls = lambda: len(seen)
    return call_model


def registry_with():
    def boom():
        raise RuntimeError("库被锁了")

    return ToolRegistry(
        [
            Tool(
                name="search_knowledge_base",
                description="search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                handler=lambda query: f"答案就在这一段：{query}",
            ),
            Tool(
                name="boom",
                description="always fails",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=boom,
            ),
        ]
    )


class TestHappyPaths:
    def test_answer_without_any_tool_call_is_flagged(self):
        run = run_agent("泰国首都是哪", ToolRegistry(), scripted([final_message("曼谷")]))

        assert run.stop_reason == STOP_ANSWERED
        assert run.answer == "曼谷"
        assert run.tool_calls == 0
        assert run.answered_without_evidence is True

    def test_search_then_answer_records_two_turns(self):
        run = run_agent(
            "迪韦齐斯在哪",
            registry_with(),
            scripted([tool_message("search_knowledge_base", '{"query": "迪韦齐斯"}'), final_message("英格兰威尔特郡")]),
            max_turns=4,
        )

        assert run.stop_reason == STOP_ANSWERED
        assert run.turns_used == 2
        assert run.tool_calls == 1
        assert run.turns[0].tools[0]["name"] == "search_knowledge_base"
        assert run.turns[0].tools[0]["ok"] is True
        assert run.messages[3]["content"] == "答案就在这一段：迪韦齐斯"
        assert run.answer == "英格兰威尔特郡"

    def test_tool_output_is_fed_back_into_the_next_request(self):
        model = scripted([tool_message("search_knowledge_base", '{"query": "迪韦齐斯"}'), final_message("威尔特郡")])

        run = run_agent("迪韦齐斯在哪", registry_with(), model, max_turns=4)

        second_call = model.seen[1]
        roles = [message["role"] for message in second_call["messages"]]
        # 第二次请求时最终答案还没产生，所以历史停在 tool 上：system / user / assistant(带 tool_calls) / tool
        assert roles == ["system", "user", "assistant", "tool"]
        assert "迪韦齐斯" in second_call["messages"][3]["content"]
        assert second_call["messages"][2]["tool_calls"][0]["id"] == "call_1"  # 原始结构被保留，不是归一化后的 Decision

    def test_final_answer_is_kept_in_the_transcript(self):
        run = run_agent("泰国首都是哪", ToolRegistry(), scripted([final_message("曼谷")]))

        assert run.messages[-1] == {"role": "assistant", "content": "曼谷"}

    def test_model_sees_the_tool_specs(self):
        model = scripted([final_message("不用查")])

        run_agent("q", registry_with(), model)

        names = {spec["function"]["name"] for spec in model.seen[0]["tools"]}
        assert {"search_knowledge_base", "boom"} <= names


class TestStopConditions:
    def test_running_out_of_turns_leaves_no_answer(self):
        replies = [tool_message("search_knowledge_base", '{"query": "a"}') for _ in range(3)]
        model = scripted(replies)

        run = run_agent("查不到呢", registry_with(), model, max_turns=3)

        assert run.stop_reason == STOP_MAX_TURNS
        assert run.answer is None
        assert run.turns_used == 3
        assert model.calls() == 3

    def test_empty_turn_is_nudged_once_then_stops(self):
        run = run_agent("q", registry_with(), scripted([{"role": "assistant", "content": ""}] * 2))

        assert run.stop_reason == STOP_INVALID
        assert run.turns_used == 2
        assert run.messages[2]["content"].startswith("You returned neither")

    def test_nudge_can_recovery_a_real_answer(self):
        run = run_agent("q", registry_with(), scripted([{"role": "assistant"}, final_message("曼谷")]))

        assert run.stop_reason == STOP_ANSWERED
        assert run.answer == "曼谷"

    def test_broken_protocol_stops_immediately_without_retry(self):
        run = run_agent("q", registry_with(), scripted([tool_message("search_knowledge_base", "{坏掉的 JSON")]))

        assert run.stop_reason == STOP_SCHEMA_ERROR
        assert run.turns_used == 1
        assert run.turns[0].action == "error"

    def test_budget_is_checked_before_calling_not_after(self):
        model = scripted(
            [
                tool_message("search_knowledge_base", '{"query": "a"}'),
                tool_message("search_knowledge_base", '{"query": "b"}'),
                final_message("答案"),
            ]
        )

        run = run_agent("q", registry_with(), model, max_turns=5, token_budget=119)

        # 第一轮花掉 120（100 + 20）已经超 119，所以第二轮请求根本不该发出去
        assert run.stop_reason == STOP_BUDGET
        assert model.calls() == 1
        assert run.usage["input_tokens"] == 100
        assert run.usage["output_tokens"] == 20


class TestFailureHandlingInsideTheLoop:
    def test_unknown_tool_name_is_returned_to_the_model_and_the_loop_continues(self):
        run = run_agent(
            "q",
            registry_with(),
            scripted([tool_message("search_web", "{}"), final_message("曼谷")]),
        )

        assert run.stop_reason == STOP_ANSWERED
        assert run.turns[0].tools[0]["ok"] is False
        assert "未知工具" in run.messages[3]["content"]

    def test_failing_tool_is_recorded_and_does_not_break_the_run(self):
        run = run_agent("q", registry_with(), scripted([tool_message("boom", "{}", call_id="c9"), final_message("答")]))

        assert run.turns[0].tools[0]["name"] == "boom"
        assert run.turns[0].tools[0]["ok"] is False
        assert "库被锁了" in run.messages[3]["content"]
        assert run.stop_reason == STOP_ANSWERED

    def test_several_tool_calls_in_one_turn_all_get_answers(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search_knowledge_base", "arguments": '{"query": "a"}'}},
                {"id": "c2", "type": "function", "function": {"name": "search_knowledge_base", "arguments": '{"query": "b"}'}},
            ],
        }

        run = run_agent("q", registry_with(), scripted([message, final_message("答")]))

        tool_messages = [m for m in run.messages if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
        assert run.turns[0].tools[1]["ok"] is True


class TestCallIdRepair:
    def test_missing_ids_are_synthesised_and_consistent_on_both_sides(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "search_knowledge_base", "arguments": '{"query": "a"}'}}]}

        run = run_agent("q", registry_with(), scripted([message, final_message("答")]))

        assistant = next(m for m in run.messages if m["role"] == "assistant" and m.get("tool_calls"))
        tool = next(m for m in run.messages if m["role"] == "tool")
        assert assistant["tool_calls"][0]["id"] == "call_auto_0" == tool["tool_call_id"]

    def test_with_call_ids_does_not_mutate_the_original_message(self):
        message = {"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]}

        patched = with_call_ids(dict(message))

        assert message["tool_calls"][0].get("id") is None
        assert patched["tool_calls"][0]["id"] == "call_auto_0"

    def test_plain_message_without_tools_passes_through(self):
        assert with_call_ids({"role": "assistant", "content": "hi"}) == {"role": "assistant", "content": "hi"}


class TestSummarize:
    def test_summary_has_the_columns_the_reviewer_asked_for(self):
        run = run_agent(
            "q",
            registry_with(),
            scripted([tool_message("search_knowledge_base", '{"query": "a"}'), final_message("曼谷")]),
        )

        summary = summarize_run(run)
        assert summary["stop_reason"] == STOP_ANSWERED
        assert summary["turns"] == 2
        assert summary["tool_calls"] == 1
        assert summary["answered_without_evidence"] is False
        assert summary["input_tokens"] == 200
        assert summary["total_tokens"] == 240
        assert summary["answer_chars"] == 2
