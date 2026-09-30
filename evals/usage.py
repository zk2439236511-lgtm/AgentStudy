"""评测的成本计量：token 用量与每次调用的延迟。

为什么要挂回调，而不是改生产链去拿返回值：`create_rag_chain_with_sources` 里是
`prompt | llm | StrOutputParser()`，StrOutputParser 把 AIMessage 丢掉了，token 用量在链的
返回值里根本不存在。挂到 `invoke(..., config={"callbacks": [...]})` 上，生产代码一行不动，
量到的仍然是线上那条链真实发出去的请求——被测对象没有因为测量而被改造。

百炼 compatible-mode 实测（2026-09-30 一次真机探针，qwen-plus）两处都有值：
- `llm_output["token_usage"]`：OpenAI 口径 `prompt_tokens / completion_tokens / total_tokens`
- `AIMessage.usage_metadata`：LangChain 标准口径 `input_tokens / output_tokens / total_tokens`

优先读后者：它是 LangChain 的跨提供商标准字段，换个 SDK 或模型 `token_usage` 未必还在。
"""

from langchain_core.callbacks import BaseCallbackHandler


def extract_usage(response) -> dict | None:
    """从 on_llm_end 收到的 LLMResult 里取出 {input,output,total}_tokens，取不到返回 None。

    返回 None 是有意为之：拿不到用量时宁可少记一条，也不要造一个 0 出来冒充"这次调用没花钱"。
    """
    generations = getattr(response, "generations", None) or []
    message = getattr(generations[0][0], "message", None) if generations else None
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage")
    if token_usage:
        return {
            "input_tokens": int(token_usage.get("prompt_tokens") or 0),
            "output_tokens": int(token_usage.get("completion_tokens") or 0),
            "total_tokens": int(token_usage.get("total_tokens") or 0),
        }
    return None


def sum_usage(records: list[dict]) -> dict:
    """把若干次调用的用量加成一个 dict。空列表返回全 0，不返回 None——调用方要直接参与求和。"""
    return {
        "llm_calls": len(records),
        "input_tokens": sum(record["input_tokens"] for record in records),
        "output_tokens": sum(record["output_tokens"] for record in records),
        "total_tokens": sum(record["total_tokens"] for record in records),
    }


class TokenUsageCollector(BaseCallbackHandler):
    """按调用发生顺序累计 token 用量。

    `snapshot()` + `usage_since(cursor)` 用来切出"这一道题花掉多少"：现在一道题在链里只发
    1 次生成，以后 Mini Agent 的多轮循环会发多次，按游标切片两种情况都成立。
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def on_llm_end(self, response, **kwargs) -> None:
        usage = extract_usage(response)
        if usage is not None:
            self.records.append(usage)

    def snapshot(self) -> int:
        return len(self.records)

    def usage_since(self, cursor: int) -> dict:
        return sum_usage(self.records[cursor:])
