# AgentStudy — AI Coding 学习与实践仓库

课程笔记 + 可运行工程 + 逐步构建的 Mini Agent。
学习主线见 [docs/study-notes.md](docs/study-notes.md)，实践过程见该文件末尾「实践日志」。

## 目录结构

```
├── docs/study-notes.md   # 12 课课程笔记 + 实践日志（改动/验证/踩坑/下一步）
├── apps/idea-api/        # 灵感引擎后端：FastAPI + SQLite（待办 + 灵感接口）
├── apps/idea-web/        # 灵感引擎前端：Vite + React 19（磨砂玻璃 Navbar、Loading 交互）
├── agent/                # Mini Agent（规划中：loop.py / tools.py / schema.py / memory.py / context.py）
├── evals/                # Mini Agent 评估集（规划中）
├── tests/                # Mini Agent 测试（规划中）
└── examples/             # 可运行示例（规划中）
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

## 学习记录方式

小步迭代，每步四件事：**改一点 → 本地验证（pytest / 浏览器）→ git 提交 → 在实践日志追加条目**（目标 / 改动 / 验证证据 / 踩坑 / 下一步）。
记录与代码放在同一次提交里，commit 历史即学习过程。

## 当前进度

- [x] 灵感引擎后端：GET /ideas、POST /ideas + pytest 8 通过
- [x] 后端接入通义千问：POST /ideas/{id}/expand 灵感展开（pytest 11 通过 + 真实调用联调成功）
- [ ] 前端接线后端 + SSE 流式打字机
- [ ] Mini Agent：agent/ 五个文件逐个落地
