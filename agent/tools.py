"""工具层：名字 → (JSON schema, handler) 的注册表，加把知识库检索包成工具。

为什么不用 LangChain 的 `@tool` / `Tool`：送进模型的 `tools` 参数只要
`{"type": "function", "function": {name, description, parameters}}` 这一段结构，
自己维护它就不必为"生成一份 JSON"引入框架；而 handler 是普通可调用对象，测试时能整个换成
假的——真检索要等索引建好（花钱在 embedding）才用得上，工具层的逻辑不该被它绑住。

工具失败**不 raise**，返回 `ok=False` 的结果文本：模型看不见返回值就学不会改用对的参数，
而"某个工具这一次没成"是 Agent 每天要处理的输入，不是程序崩溃。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from agent.schema import ToolCall, missing_required


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., str]

    def spec(self) -> dict:
        """OpenAI 兼容 `tools` 数组里的一个元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def required(self) -> list[str]:
        return list(self.parameters.get("required") or [])


@dataclass(frozen=True)
class ToolResult:
    """一次工具执行的结果。`content` 一定是字符串——tool 消息的 content 就是这个类型。"""

    name: str
    ok: bool
    content: str


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            # 同名工具送进模型会让它看到两份定义，选哪份不可预测——这种错要在装配时就炸掉
            raise ValueError(f"工具重名：{tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> set[str]:
        return set(self._tools)

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, call: ToolCall) -> ToolResult:
        """按名字派发：未知工具、缺必填参数、handler 抛错都变成 `ok=False` 的说明文本。"""
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "（当前没有注册任何工具）"
            return ToolResult(call.name, ok=False, content=f"未知工具 {call.name}。可用工具：{known}")

        absent = missing_required(call, tool.parameters)
        if absent:
            return ToolResult(
                call.name,
                ok=False,
                content=f"缺少必填参数：{', '.join(absent)}。请按 schema 重新调用。",
            )

        unexpected = sorted(set(call.arguments) - set((tool.parameters.get("properties") or {}).keys()))
        if unexpected:
            # 单独判未知参数，是为了不把 handler 内部抛的 TypeError 误标成"参数不匹配"
            return ToolResult(
                call.name,
                ok=False,
                content=f"不支持的参数：{', '.join(unexpected)}。可用参数：{sorted(tool.parameters.get('properties') or {})}",
            )

        try:
            content = tool.handler(**call.arguments)
        except Exception as exc:  # 工具内部怎么炸都不该带走整个循环
            return ToolResult(call.name, ok=False, content=f"工具执行失败：{type(exc).__name__}: {exc}")

        return ToolResult(call.name, ok=True, content=str(content))


def format_hits(hits: Sequence[tuple], limit: int = 600) -> str:
    """把 `(document, 距离)` 列表排成给模型读的编号块。

    带上距离是因为检索层那条 d ≤ 1.0 的拒答线就作用在它上面：Agent 看不到距离，
    就没法判断"这段到底像不像答案"，只能全信。
    """
    if not hits:
        return "没有检索到任何片段。"

    blocks = []
    for index, (document, distance) in enumerate(hits, start=1):
        metadata = getattr(document, "metadata", None) or {}
        source = metadata.get("source") or metadata.get("fileName") or "未知来源"
        page = metadata.get("page")
        where = f"{source} 第 {int(page) + 1} 页" if isinstance(page, int) else source
        text = (getattr(document, "page_content", "") or "").strip().replace("\n", " ")
        blocks.append(
            f"[{index}] 距离 {round(float(distance), 4)}｜{where}\n{text[:limit]}{'…' if len(text) > limit else ''}"
        )
    return "\n\n".join(blocks)


def make_search_tool(search: Callable[[str, int], Sequence[tuple]], default_k: int = 8) -> Tool:
    """把"已有的检索函数"包成 `search_knowledge_base`。

    `search(query, k)` 的契约是返回 `(document, 平方欧氏距离)` 列表——**这里不 import
    LangChain / Chroma**，所以工具层能离线测；真实接线（建索引、套 d ≤ 1.0 的拒答线）放在
    调用方，那部分要花钱。
    """

    def handler(query: str, k: int | None = None) -> str:
        return format_hits(search(query, int(k) if k else default_k))

    return Tool(
        name="search_knowledge_base",
        description=(
            "Search the knowledge base for passages relevant to the query. "
            "Returns numbered passages with their retrieval distance (smaller is closer; "
            "above 1.0 the passage is probably unrelated). Call it before answering."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or a focused sub-question."},
                "k": {"type": "integer", "description": f"How many passages to return (default {default_k})."},
            },
            "required": ["query"],
        },
        handler=handler,
    )
