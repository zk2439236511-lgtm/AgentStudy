"""零额度的 Mini Agent 演示：用脚本化的假模型把整个循环跑一遍，把 transcript 和成本摘要打出来。

为什么用假模型：真跑一次 6 轮最坏要 4 万多次 token 计费（见 `agent/schema.py` 的预算函数），
而这段演示要证明的只是"循环真的在动"——每一轮做了什么决定、工具成功还是失败、消息按什么
角色回到请求里。接真模型只需要换掉 `call_model` 这个注入点，循环一行不改。

演示故意走三条路：模型先编了一个不存在的工具（工具层回一段解释文本而不是抛错），
下一轮改用注册过的检索工具拿到证据，最后一轮作答。

用法：
    PYTHONIOENCODING=utf-8 python examples/agent_loop_demo.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.loop import ModelReply, run_agent, summarize_run
from agent.tools import Tool, ToolRegistry

EVIDENCE = (
    "[1] 距离 0.4231｜cmrc2018 语料\n"
    "迪韦齐斯（Devizes）是英格兰威尔特郡的一座集镇，位于肯尼特区，"
    "皮尤西河谷与索尔兹伯里平原之间。"
)

SCRIPT = [
    # 第 1 轮：模型编了一个没注册的工具名——工具层会把这变成一段给它看的解释
    {
        "role": "assistant",
        "content": "我先搜一下网上有没有。",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search_web", "arguments": '{"query": "迪韦齐斯"}'}}
        ],
    },
    # 第 2 轮：改用注册过的检索工具
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "search_knowledge_base", "arguments": '{"query": "迪韦齐斯 属于哪个郡"}'},
            }
        ],
    },
    # 第 3 轮：拿到证据后作答
    {"role": "assistant", "content": "英格兰威尔特郡。"},
]


def fake_model(messages: list[dict], tools: list[dict]) -> ModelReply:
    """按脚本吐消息，并假装每次调用花了 300 input / 25 output token。"""
    index = sum(1 for message in messages if message["role"] == "assistant")
    if index >= len(SCRIPT):
        return ModelReply(message={"role": "assistant", "content": "没有更多脚本了"}, usage=None)
    return ModelReply(message=SCRIPT[index], usage={"input_tokens": 300 + index * 200, "output_tokens": 25})


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="search_knowledge_base",
                description="Search the local knowledge base.",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                handler=lambda query: EVIDENCE,
            )
        ]
    )


def main() -> None:
    run = run_agent("迪韦齐斯属于英格兰的哪个郡？", build_registry(), fake_model, max_turns=5)

    print("=== 消息轨迹（角色 → 这一条是什么）===\n")
    for message in run.messages:
        if message["role"] == "tool":
            first_line = message["content"].splitlines()[0]
            print(f"  [{message['role']}] tool_call_id={message['tool_call_id']} → {first_line}")
        elif message.get("tool_calls"):
            names = [call["function"]["name"] for call in message["tool_calls"]]
            print(f"  [{message['role']}] 要调用工具 {names}（content={message['content']!r}）")
        else:
            print(f"  [{message['role']}] {(message.get('content') or '')[:60]}")

    print("\n=== 每轮的决定 ===\n")
    for turn in run.turns:
        tools = ", ".join(f"{item['name']}({'成功' if item['ok'] else '失败'} {item['chars']}字)" for item in turn.tools)
        print(f"  第 {turn.index + 1} 轮：{turn.action}｜工具 {tools or '无'}｜用量 {turn.usage}")

    print(f"\n=== 结论 ===\n  停止原因：{run.stop_reason}\n  答案：{run.answer}")
    print(f"\n=== 成本摘要 ===\n  {summarize_run(run)}")


if __name__ == "__main__":
    main()
