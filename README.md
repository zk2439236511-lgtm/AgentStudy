# AgentStudy — AI Coding 学习与实践仓库

课程笔记 + 可运行工程 + 逐步构建的 Mini Agent。
学习主线见 [docs/study-notes.md](docs/study-notes.md)，实践过程见该文件末尾「实践日志」。

## 目录结构

```
├── docs/study-notes.md   # 12 课课程笔记 + 实践日志（改动/验证/踩坑/下一步）
├── apps/idea-api/        # 灵感引擎后端：FastAPI + SQLite（待办 + 灵感接口 + 千问展开）
├── apps/idea-web/        # 灵感引擎前端：Vite + React 19（磨砂玻璃 Navbar、Loading 交互）
├── apps/knowledge-rag/   # 知识库 RAG：LangChain 四模块 + Chroma 持久化 + FastAPI 问答接口
├── apps/kb-web/          # 知识库前端：Vite + React 19（上传 PDF / txt / md、提问、答案 + 来源出处）
├── examples/             # 可运行示例：rag_baseline_demo.py（一条命令跑通完整 RAG）
├── agent/                # Mini Agent（规划中：loop.py / tools.py / schema.py / memory.py / context.py）
├── evals/                # 检索评测：metrics.py + 自造题网格 + CMRC2018 金标集 + Chroma 距离口径/拒答阈值实验（datasets/）
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
pytest                         # 当前 51 passed, 5 skipped（集成用例需 RUN_INTEGRATION=1）

cd ../..                       # 示例与评测脚本在仓库根目录下运行
python examples/rag_baseline_demo.py     # 内存版完整 RAG
python examples/kb_persistent_demo.py    # 持久化版：首次建库，之后重启直接问

cd apps/knowledge-rag
PYTHONPATH=src uvicorn ragdemo.api:app --port 8001   # 知识库 HTTP 接口
```

`/kb/ask` 带**检索层拒答**：最像的片段离得不够近就直接不作答（不调模型），响应里的 `refused` 告诉前端这是拒答而不是答案。
阈值走环境变量 `RAG_MAX_DISTANCE`（默认 `1.0`，Chroma 的平方欧氏距离口径），工作点的推导见下面 C 段；
`InMemoryVectorStore` 的分数是余弦相似度、不是距离，所以内存版调用点显式传 `max_distance=None` 关掉这条线。

检索评测（evals/，指标计算是纯函数、离线免费；跑评测会真实调用百炼 embedding）：

```bash
pytest evals                       # 当前 107 passed（指标 + 金标集构建 + 距离口径实验 + 答案层指标 + token 计量，全部离线）

# A. 自造题：chunk_size × k 网格，判的是"相关页有没有进前 K"
python evals/retrieval_eval.py --chunk-sizes 500,1000,2000 --ks 4,8
                               # 结果写入 evals/results/{日期}-grid.{json,md}

# B. 外部金标集（CMRC2018 人工标注，解掉自造题的数据泄漏）
#    语料不进 docs/、不建进线上知识库，只在评测脚本里建独立的内存索引；
#    原始 json 不入仓（体积大、且带 license 溯源要求），按 MANIFEST 的 URL + sha256 自行下载
curl -L -o evals/datasets/raw/cmrc2018_dev.json \
     https://raw.githubusercontent.com/ymcui/cmrc2018/master/data/cmrc2018_dev.json
python evals/datasets/build_cmrc_golden.py        # 重抽金标集，默认 100 段 / 200 问 / 30 负样本
python evals/cmrc_retrieval_eval.py --chunk-sizes 0,500 --ks 4,8
                               # 结果写入 evals/results/{日期}-cmrc.{json,md}
                               # 除检索指标外还出 top1 相似度分布 + 阈值扫描（供 C 段换算用）

# C. 距离阈值拒答：把 B 的金标集建进**真 Chroma**（临时目录），先反推距离口径再选工作点
#    会真实调用百炼 embedding；跑完把 --keep 换掉可保留库便于复查
python evals/chroma_threshold_eval.py --chunk-size 0 --k 8 \
       --thresholds 0.3,0.4,0.5,0.6,0.8,1.0,1.2 --calibrate 50
                               # 结果写入 evals/results/{日期}-chroma-threshold.{json,md}
                               # 实测结论：collection 的 space=l2（平方欧氏），且百炼向量已归一化
                               # → cos = 1 - d/2；据此换算后与内存库逐题对齐（230 题 gold_rank 差异 0/200）
                               # 工作点 d ≤ 1.0：30 道负样本拒掉 24 道，200 道可回答题只误伤 3 道

# D. 答案层评测：把 C 段建好的真 Chroma 接上线上问答链，阈值开/关各跑一遍对比
#    ⚠ 会真实调用 qwen-plus 生成（全量 230 题 ≈ 253 次生成），免费额度有限，先 --limit 冒烟
#    ⚠ 实测每次生成平均输入 ~2.6k token（k=8 的上下文就这个价），全量一趟约 54 万 input token
python evals/answer_eval.py                     # 全量：阈值开 230 题 + 阈值关探针 50 题
python evals/answer_eval.py --limit 8 --probe 4 # 冒烟：走通拒答与作答两条路径即可
                               # 加 --out-suffix tokens 写 {日期}-answer-tokens.*，别覆盖已提交的全量结果
                               # 结果写入 evals/results/{日期}-answer.{json,md}
                               # 指标：EM / char-F1 / span 命中 / 字面支撑(unigram+bigram) / 拒答分层
                               #      + 成本列：token 用量（回调实测）与每题延迟分位数
                               # 结论：span 命中 0.9077 而 char-F1 只有 0.5163 → 瓶颈是答案啰嗦不是答错；
                               #        EM 0.0974 在整句答案下没有解释力，必须与 span 命中成对看；
                               #        负样本拒答 29/30（阈值拦 24 + 模型自拒 5），与 C 段纯检索层数字一致
```

> 分数口径注意：`InMemoryVectorStore` 的分数是**余弦相似度（越大越相关）**，线上 kb-web 走 Chroma，`score` 是
> **平方欧氏距离（越小越相关）**，两套数字必须像 C 段那样先实测换算才能对齐，阈值不能直接搬。

知识库网页（先启动上面的后端接口）：

```bash
cd apps/kb-web
npm install
npm run dev                    # http://127.0.0.1:5173，/kb 请求由 Vite 代理到 8001
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
- [x] 4. CLI → FastAPI + React 网页问答（上传 PDF / txt / md、提问、Loading/错误态，浏览器端到端验证）
- [x] 5. 答案附引用出处（文件名 + 页码 + 向量距离 + 原文块，点击展开）
      · 进阶待做：答案句子与来源块逐句对应、点击跳到 PDF 具体位置
- [x] 6. evals/ 检索评测：Hit@K / Recall@K / MRR / 关键词覆盖，chunk_size × k 网格实验
      · 已把结论降级到数据能支撑的范围（k 增大天然抬高 Hit/Recall），并标注题源泄漏
      · 答案层已补简化指标（EM / char-F1 / span 命中 / 字面支撑 / 拒答正确率，见补课路线 3 与上面 D 段）；
        语义级 Correctness（让模型判分）仍未做，那是下一步

外部评审（84/100）后的补课路线：

- [x] 0. 诚实性修补：前端页码 off-by-one、`relevance` 改为原始 `distance`、题源标注、扫描件空文本不再伪装成功
- [x] 1. 知识库支持 txt / md：`SUPPORTED_SUFFIXES` 统一白名单，中文按零宽后顾在句末下刀；顺带修 `/kb/status` 跨线程复用 sqlite 连接导致的 500
- [x] 2. 外部金标集（CMRC2018 dev，人工标注）：抽取器 + 100 段 / 200 问 / 30 构造负样本，只建独立内存索引不进线上知识库，解掉自造 `expected_pages` 的数据泄漏
      · 结论要诚实：这份数据在检索层已饱和（四组配置 Hit/Recall 全 1.0），判不了 chunk_size 优劣；有区分度的是可回答题与负样本的 top1 相似度分离度（中位 0.748 vs 0.398），这是下一步阈值拒答的依据
      · 语料原始 json 不入仓，按 `evals/datasets/cmrc2018/MANIFEST.json` 的 URL + sha256 下载重建
- [x] 3. 分层指标：检索层补 Precision@K、nDCG@K，答案层 EM/F1 + 简化 Faithfulness + 拒答正确率
      · 检索层已完成（Precision@K / nDCG@K / top1 相似度分布 / 阈值扫描）；答案层也已完成，见上面 D 段与 `evals/results/2026-09-30-answer.md`
      · 注意单 gold 数据下 Precision@K 的天花板恒等于 1/K，0.25 / 0.125 是恒等式不是质量结论
      · 同一类陷阱在答案层更凶：gold 是 2~10 字的原文 span，模型答成整句时 EM 结构性趋 0（实测 0.0974），
        所以 EM 必须与 span 命中（0.9077）成对报；拒答的题不能进 EM 分母，否则"全拒答"能刷出漂亮的平均分，
        因此质量出双口径（作答子集 / 全分母按 0 计）
      · 字面支撑 unigram 已实测饱和（均值 0.9914、171/195 道等于 1.0），只有 bigram（0.8442）有区分度；
        而且这两个都只证"字在上下文里出现过"，不证事实性——凭参数知识答对的题字面支撑同样高
- [ ] 4. 距离阈值拒答 + rerank 单变量实验
      · 距离阈值拒答已完成：先在真 Chroma 上实测口径（`space=l2`、百炼向量已归一 → `cos = 1 - d/2`），再选工作点 `d ≤ 1.0`，落到 `/kb/ask` 的 `refused` 字段与前端拒答态；见 `evals/results/2026-09-30-chroma-threshold.md`
      · 已知限制：只看 top1 一根线，后 7 块不参与判定；换 embedding 模型或向量库要重扫
      · 答案层对比跑出来的意外结论：把阈值关掉，提示词那句"没有依据就说不知道"在 30 道构造负样本上自拒 29 道，
        拒答率与阈值开组同为 0.9667——所以阈值的价值在这份数据上不是"拒得更多"，而是省下 27/230 次生成调用、
        以及不看内容的可预测性；两层共同的盲区是"距离很近 + 模型敢答"（`DEV_400` top1 仅 0.5016，被硬答）
      · 待做：rerank 单变量实验（一轮只动一个变量，避免和阈值结论混在一起）
- [ ] 5. Mini Agent：裸写 OpenAI SDK `tool_calls` 循环，不套框架

第三阶段 · Mini Agent：把 RAG 注册成 `search_knowledge_base(query)` 工具，让模型自己决定查不查、证据够不够
