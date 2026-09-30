"""决定层：把一条 assistant 消息归一化成 `Decision`，再把多轮的 token 预算写成纯函数。

三件事决定了解析必须由纯函数来做，而不是顺着 SDK 的对象一路 `.attribute` 点下去：

1. `tool_calls` 是这一整层最容易出错的地方——一轮可能发多个调用、`arguments` 是 JSON
   **字符串**（可能被双重编码，也可能压根不是合法 JSON）、也可能既不发工具也不给答案。
   每一种都要能单独盖一条用例，所以这里不 import openai，只吃 Mapping：真跑时把
   `message.model_dump()` 传进来即可，解析层因此与提供商无关。
2. 错误分两类，处理方式完全不同。**协议层垃圾**（有 tool_calls 但缺 name、arguments
   解不出来）raise `DecisionError`——这种消息没法执行，重试也没意义；**模型没决定**
   （既无 tool_calls 又无文本）是合法的第三种动作 `invalid`，交给循环决定要不要再问一次。
   把两者混成一个返回值，循环就没法区分"该重试"和该停。
3. 预算算式放在这里而不是 loop 里：⑤c 的经验是"跑之前先算钱"。Agent 比单次问答危险得多，
   上下文每轮都在变长，`max_turns × 单次单价` 是**系统性低估**（第 5 轮的输入比第 1 轮大
   好几千 token），所以按 `第 i 轮输入 = base + i × growth` 逐轮累加。
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass

ACTION_TOOL = "tool"
ACTION_FINAL = "final"
ACTION_INVALID = "invalid"

DEFAULT_MAX_TURNS = 6
# base / growth 的初值借用了 RAG 那两层的实测单价（k=8 一次检索的上下文 ≈ 2.6~2.8k input），
# 它只是"跑之前有个量级"，Agent 自己真机跑过之后必须换成实测数。
DEFAULT_BASE_TOKENS = 600
DEFAULT_GROWTH_TOKENS = 2600
DEFAULT_OUTPUT_TOKENS_PER_TURN = 150


class DecisionError(ValueError):
    """消息结构不合法，没法归一化成任何可执行的决定。"""


@dataclass(frozen=True)
class ToolCall:
    """一次工具调用：`arguments` 已经是 dict（解析在 `parse_message` 里做完）。"""

    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Decision:
    """模型这一轮想干什么。`text` 是它的原始文本（工具轮也常带一句想法，留着进状态记录）。"""

    action: str
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def answer(self) -> str | None:
        return self.text if self.action == ACTION_FINAL else None


def parse_arguments(value) -> dict:
    """把 `function.arguments` 变成 dict。

    OpenAI 口径这里是 JSON 字符串；个别模型会直接给 dict，或把它双重编码成
    `'"{"query": "x"}"'`（外面还套一层引号）。空串算空参数——无参工具是合法的。
    解不出 dict 就 raise：宁可停，也不要静默丢参数照样去调工具，那种错最难查。
    """
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise DecisionError(f"arguments 类型不是字符串或 dict：{value!r}")

    text = value.strip()
    # 最多剥两层：一层是正常 JSON，第二层是双重编码。再多就说明数据坏了，别无限循环。
    for _ in range(2):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DecisionError(f"arguments 不是合法 JSON：{value!r}") from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
        if isinstance(parsed, str):
            text = parsed.strip()
            continue
        raise DecisionError(f"arguments 解出来不是对象，而是 {type(parsed).__name__}：{value!r}")
    raise DecisionError(f"arguments 被编码超过两层：{value!r}")


def parse_tool_call(raw) -> ToolCall:
    function = raw.get("function") or {}
    name = function.get("name")
    if not name:
        raise DecisionError(f"工具调用缺少 name：{raw!r}")
    return ToolCall(
        call_id=raw.get("id") or "",
        name=name,
        arguments=parse_arguments(function.get("arguments")),
    )


def parse_message(message) -> Decision:
    """一条 assistant 消息（Mapping）→ `Decision`。有 tool_calls 就优先按工具轮处理。"""
    calls = message.get("tool_calls") or []
    text = (message.get("content") or "").strip() or None
    if calls:
        return Decision(ACTION_TOOL, text=text, tool_calls=tuple(parse_tool_call(call) for call in calls))
    if text:
        return Decision(ACTION_FINAL, text=text)
    return Decision(ACTION_INVALID)


def find_unknown_tools(decision: Decision, known: set[str]) -> list[str]:
    """模型编出来的工具名（拼错、或凭空造一个 `search_web`）按出现顺序返回。

    这一层只报告不抛错：未知工具名是"内容错"不是"结构错"，标准做法是把错误当成工具的
    返回值喂回模型，让它下一轮自己改用正确的名字。
    """
    names = {call.name for call in decision.tool_calls}
    return sorted(name for name in names if name not in known)


def missing_required(call: ToolCall, schema: dict) -> list[str]:
    """按工具的 JSON schema 检查必填参数缺了哪些。

    只查 `required`，不实现完整的 JSON Schema 校验器：这里要防的是"模型不给参数"，
    参数类型对不对交给 handler 自己 raise——写半个校验器只会让人误以为它什么都管。
    """
    required = (schema or {}).get("required") or []
    return [field for field in required if call.arguments.get(field) in (None, "")]


def turn_input_curve(base_tokens: int, growth_per_turn: int, turns: int) -> list[int]:
    """第 i 轮（从 0 数起）请求体里的输入 token 估算：`base + i × growth`。

    `base` = 系统提示 + 工具 schema + 用户问题（不含任何工具结果）；
    `growth` = 每往上下文里塞一条工具结果增加的 token。
    """
    return [base_tokens + index * growth_per_turn for index in range(turns)]


def estimate_budget(
    max_turns: int = DEFAULT_MAX_TURNS,
    base_tokens: int = DEFAULT_BASE_TOKENS,
    growth_per_turn: int = DEFAULT_GROWTH_TOKENS,
    output_per_turn: int = DEFAULT_OUTPUT_TOKENS_PER_TURN,
) -> dict:
    """跑满 `max_turns` 轮的最坏情况用量，按逐轮累加而不是"轮数 × 单价"。

    最后一轮之前模型可能直接答完，所以这是上界；真机跑完要用实测的每轮输入替换
    `base/growth`，否则下一个人拿到的还是猜的数。
    """
    if max_turns <= 0:
        raise ValueError("max_turns 必须是正数，0 轮的 Agent 不是 Agent")
    curve = turn_input_curve(base_tokens, growth_per_turn, max_turns)
    inputs = sum(curve)
    outputs = output_per_turn * max_turns
    return {
        "max_turns": max_turns,
        "generations": max_turns,
        "input_tokens": inputs,
        "output_tokens": outputs,
        "total_tokens": inputs + outputs,
        "input_curve": curve,
        "mean_input_per_turn": round(inputs / max_turns, 1),
        "assumptions": {
            "base_tokens": base_tokens,
            "growth_per_turn": growth_per_turn,
            "output_per_turn": output_per_turn,
        },
    }


def over_budget(estimate: dict, available_tokens: int) -> tuple[bool, int]:
    """最坏情况估算超不超额度，返回 (是否超, 差额)。

    比"够不够跑一轮"严格：Agent 的循环一旦开跑就会连着花好几轮，中途才发现超额只能把
    已经烧掉的退不回来，所以判定用总开销，不是单次。
    """
    total = estimate["total_tokens"]
    return total > available_tokens, total - available_tokens
