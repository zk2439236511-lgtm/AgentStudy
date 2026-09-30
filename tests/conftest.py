"""让根目录下的 `tests/` 能 `import agent.xxx`。

`agent/` 是仓库根里的一个包，没有装成 pip 包，所以手动把仓库根加进 `sys.path`——
和 `apps/knowledge-rag` 用 `PYTHONPATH=src` 跑测试是同一个思路，只是这里必须写进 conftest
才能同时喂给 pytest 和命令行脚本。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
