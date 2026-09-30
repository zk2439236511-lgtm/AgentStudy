"""主循环：Decide → 执行工具 → 把结果喂回去 → 再决定，直到撞上一条停止条件。

模型调用是**注入进来**的（`call_model(messages, tools) -> ModelReply`），不是在这里 new 一个
OpenAI 客户端。理由有两条，都很具体：

1. 这是整个 Agent 唯一花钱的地方。做成参数之后，"跑满轮数""中途放弃""协议错""预算超限"
   这四条停止路径都能用一串脚本化的假消息在离线测完；真机只需要确认接线对不对
   （⑤b 的教训写在日志第 12 条：离线全绿证明不了 `config` 真能透传，得靠真机）。
2. 换成 Ollama 或别的提供商时，要改的只有适配器，循环一行不动。

`tool_calls` 的 id 会被"补"而不是原样转发：解析层允许模型不给 id（`schema.ToolCall.call_id`
可以是空串），但下一轮的 `role=tool` 消息必须靠 id 对上号，所以这里给缺 id 的调用合成一个，
并把同一个 id 写回 assistant 消息——两边不一致会让 API 直接报错。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from agent.schema import (
    ACTION_FINAL,
    ACTION_INVALID,
    ACTION_TOOL,
    DEFAULT_MAX_TURNS,
    DecisionError,
    parse_message,
)
from agent.tools import ToolRegistry

STOP_ANSWERED = "answered"
STOP_MAX_TURNS = "max_turns"
STOP_INVALID = "no_decision"
STOP_BUDGET = "token_budget_exceeded"
STOP_SCHEMA_ERROR = "schema_error"

# ⑤c 的结论直接决定这段提示词怎么写：把答案压到最短会让模型连"我为什么不答"都懒得说，
# 从而把一次正确的自拒换成一个自信的错答。所以这里明确要求它说出不成立的地方。
AGENT_SYSTEM_PROMPT = """You are a question-answering agent with access to one tool: search_knowledge_base.

Rules:
1. Before answering, call search_knowledge_base. You may call it more than once with different queries.
2. Base your answer only on what the tool returns. Bigger retrieval distance means weaker support.
3. If the retrieved passages do not support an answer, say that you cannot answer, and state briefly
   what is missing. Do not guess from your own knowledge.
4. If the question names an entity that the passages do not contain, point out the mismatch instead
   of answering about a similar entity.
5. When the evidence does support an answer, answer in one short sentence.
"""

NUDGE_MESSAGE = "You returned neither a tool call nor an answer. Either call search_knowledge_base or answer."

EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0}


@dataclass(frozen=True)
class ModelReply:
    """一次模型调用的返回：`message` 是 assistant 消息（Mapping），`usage` 由适配器折算成
    `input_tokens / output_tokens`——和 `evals/usage.py` 同一套口径，别让两处各叫各的。"""

    message: dict
    usage: dict | None = None


@dataclass(frozen=True)
class Turn:
    index: int
    action: str
    text: str | None
    tools: tuple[dict, ...]
    usage: dict


@dataclass
class AgentRun:
    """一次任务的全部状态记录：答案、为什么停、每轮做了什么、花了多少。"""

    question: str
    answer: str | None = None
    stop_reason: str | None = None
    turns: list[Turn] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"calls": 0, **EMPTY_USAGE})

    @property
    def turns_used(self) -> int:
        return len(self.turns)

    @property
    def tool_calls(self) -> int:
        return sum(len(turn.tools) for turn in self.turns)

    @property
    def answered_without_evidence(self) -> bool:
        """直接给答案却一次工具都没调——评估里最该被抓出来的行为（凭空作答）。"""
        return self.stop_reason == STOP_ANSWERED and self.tool_calls == 0


def with_call_ids(message: dict) -> dict:
    """给缺 id 的 tool_calls 补上合成 id，返回一份可以安全塞回历史的 assistant 消息。"""
    calls = message.get("tool_calls") or []
    if not calls:
        return dict(message)

    patched = []
    for position, call in enumerate(calls):
        entry = dict(call)
        entry["id"] = entry.get("id") or f"call_auto_{position}"
        entry["function"] = dict(entry.get("function") or {})
        patched.append(entry)

    copied = dict(message)
    copied["tool_calls"] = patched
    return copied


def run_agent(
    question: str,
    registry: ToolRegistry,
    call_model: Callable[[list[dict], list[dict]], ModelReply],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    token_budget: int | None = None,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
) -> AgentRun:
    """跑一个任务。返回 `AgentRun`，**不抛异常**（除了 `call_model` 自己炸）。

    `token_budget` 是"已花总量"的上限：超了就当场停。已经烧掉的退不回来，所以判定放在
    每次调用**之前**——宁可少跑一轮，也不要跑到一半才发现额度没了。
    """
    run = AgentRun(question=question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    run.messages = messages
    nudged = False

    for index in range(max_turns):
        spent = run.usage["input_tokens"] + run.usage["output_tokens"]
        if token_budget is not None and spent >= token_budget:
            run.stop_reason = STOP_BUDGET
            return run

        reply = call_model(messages, registry.specs())
        usage = reply.usage or EMPTY_USAGE
        run.usage["calls"] += 1
        run.usage["input_tokens"] += usage.get("input_tokens", 0)
        run.usage["output_tokens"] += usage.get("output_tokens", 0)

        try:
            decision = parse_message(reply.message)
        except DecisionError as exc:
            # 结构都不合法，重试也只会再花一次钱买同一条坏消息
            run.turns.append(Turn(index, "error", str(exc), (), usage))
            run.stop_reason = STOP_SCHEMA_ERROR
            return run

        if decision.action == ACTION_FINAL:
            # 最终这条也进历史：状态记录应该是一段读得通的完整对话，而不是停在工具结果上
            messages.append(with_call_ids(dict(reply.message)))
            run.turns.append(Turn(index, decision.action, decision.text, (), usage))
            run.answer = decision.text
            run.stop_reason = STOP_ANSWERED
            return run

        if decision.action == ACTION_INVALID:
            if nudged:
                run.turns.append(Turn(index, decision.action, None, (), usage))
                run.stop_reason = STOP_INVALID
                return run
            nudged = True
            run.turns.append(Turn(index, decision.action, "nudged once", (), usage))
            messages.append({"role": "user", "content": NUDGE_MESSAGE})
            continue

        # 进历史的是模型原始消息（含 tool_calls），不是归一化后的 Decision——
        # 下一轮的 role=tool 要靠原始结构里的 id 对上号
        messages.append(with_call_ids(dict(reply.message)))

        outcomes = []
        for position, call in enumerate(decision.tool_calls):
            result = registry.call(call)
            call_id = call.call_id or f"call_auto_{position}"
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result.content})
            outcomes.append({"name": result.name, "ok": result.ok, "chars": len(result.content)})

        run.turns.append(Turn(index, decision.action, decision.text, tuple(outcomes), usage))

    run.stop_reason = STOP_MAX_TURNS
    return run


def summarize_run(run: AgentRun) -> dict:
    """把一次运行折成一行可进结果表的数（评审要的是轮数与成本，不是整段 transcript）。"""
    total = run.usage["input_tokens"] + run.usage["output_tokens"]
    return {
        "stop_reason": run.stop_reason,
        "turns": run.turns_used,
        "tool_calls": run.tool_calls,
        "answered_without_evidence": run.answered_without_evidence,
        "calls": run.usage["calls"],
        "input_tokens": run.usage["input_tokens"],
        "output_tokens": run.usage["output_tokens"],
        "total_tokens": total,
        "answer_chars": len(run.answer or ""),
    }
