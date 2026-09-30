"""Mini Agent：裸写 OpenAI SDK 的 tool_calls 循环，不套 LangGraph / AgentExecutor。

模块分工（按依赖方向，schema 谁都不依赖）：
- `schema.py`：把模型消息归一化成 Decision + 循环的 token 预算算式，纯函数、离线可测
- `tools.py`：工具注册表（name / description / JSON schema / handler）
- `loop.py`：Decide → 调工具 → 把结果喂回去 → 再决定，直到答了、放弃或撞上上限
- `context.py` / `memory.py`：上下文组装与跨轮状态记录

之所以自己写：外部评审要验证的正是"最大轮数、结构化 decision、工具 schema、状态记录"
这几样能不能自己实现并测出来，交给框架黑盒就量不出来了。
"""
