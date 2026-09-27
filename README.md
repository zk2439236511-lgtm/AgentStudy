# AgentStudy — AI Coding 学习与实践仓库

课程笔记 + 可运行工程 + 逐步构建的 Mini Agent。
学习主线见 [docs/study-notes.md](docs/study-notes.md)，实践过程见该文件末尾「实践日志」。

## 目录结构

```
├── docs/study-notes.md   # 12 课课程笔记 + 实践日志（改动/验证/踩坑/下一步）
├── apps/idea-api/        # 灵感引擎后端：FastAPI + SQLite（待办 + 灵感接口 + 千问展开）
├── apps/idea-web/        # 灵感引擎前端：Vite + React 19（磨砂玻璃 Navbar、Loading 交互）
├── apps/knowledge-rag/   # 知识库 RAG 基线：LangChain 四模块（Load→Chunk→Embed→Retrieve→Generate）
├── examples/             # 可运行示例：rag_baseline_demo.py（一条命令跑通完整 RAG）
├── agent/                # Mini Agent（规划中：loop.py / tools.py / schema.py / memory.py / context.py）
├── evals/                # RAG / Mini Agent 评测集（规划中）
└── tests/                # Mini Agent 测试（规划中）
```

## 快速运行

后端（API + 测试）：

```bash
cd apps/idea-api
pip install fastapi uvicorn pytest httpx openai
cp .env.example .env                 # 然后填入你的百炼 API-KEY（.env 已被忽略，不会上传）
uvicorn main:app --port 8000         # 启动 API，访问 http://127.0.0.1:8000
pytest tests                         # 当前 11 passed
```

前端：

```bash
cd apps/idea-web
npm install
npm run dev                    # http://127.0.0.1:5173
```

知识库 RAG（apps/knowledge-rag，基线来自 kousen/ragdemo 的 Python 版）：

```bash
cd apps/knowledge-rag
pip install -r requirements.txt
cp .env.example .env           # 填入百炼 API-KEY
pytest                         # 当前 19 passed, 5 skipped（集成用例需显式导出密钥）

python examples/rag_baseline_demo.py     # 在仓库根目录跑内存版完整 RAG
python examples/kb_persistent_demo.py    # 持久化版：首次建库，之后重启直接问
```

## 学习记录方式

小步迭代，每步四件事：**改一点 → 本地验证（pytest / 浏览器）→ git 提交 → 在实践日志追加条目**（目标 / 改动 / 验证证据 / 踩坑 / 下一步）。
记录与代码放在同一次提交里，commit 历史即学习过程。

## 当前进度

第一阶段 · 灵感引擎（LLM 应用）

- [x] 后端：GET /ideas、POST /ideas + pytest 8 通过
- [x] 接入通义千问：POST /ideas/{id}/expand（pytest 11 通过 + 真实联调）
- [ ] 前端接线后端 + SSE 流式打字机

第二阶段 · 知识库 RAG（六阶段路线）

- [x] 1. 原版 Python 跑通 + pytest（14 passed）
- [x] 2. OpenAI → 百炼 Qwen：模型与端点全部环境变量化，真实问答联调成功
- [x] 3. InMemoryVectorStore → 持久化向量库（Chroma 本地文件 + SHA-256 哈希去重，pytest 19 通过）
- [ ] 4. CLI → FastAPI + React 网页问答
- [ ] 5. 答案附引用出处（PDF 文件名 + 页码 + 原文块）
- [ ] 6. evals/ 检索评测（Recall@K、命中率）

第三阶段 · Mini Agent：把 RAG 注册成 `search_knowledge_base(query)` 工具，让模型自己决定查不查、证据够不够
