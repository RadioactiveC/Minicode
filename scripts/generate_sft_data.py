"""Generate SFT training data for MinCode's 3 coding tools via DeepSeek API.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    python scripts/generate_sft_data.py --num_batches 10 --output dataset/mincode_sft.jsonl

Each batch asks DeepSeek to generate ~30 diverse samples. With 10 batches you get ~300 samples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── MinCode 的 3 个工具定义（与 tools.py 保持一致）──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录中的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "写入内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

TOOLS_JSON = json.dumps(TOOLS, ensure_ascii=False)

# ── 种子数据：30 条高质量样本 ──

SEED_SAMPLES = [
    # ===== list_files 单工具调用 (8条) =====
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "看看当前目录有什么文件"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "."}}], ensure_ascii=False)},
            {"role": "tool", "content": "src/\ntests/\nREADME.md\npyproject.toml\n.gitignore"},
            {"role": "assistant", "content": "当前目录包含以下内容：\n- src/ 目录（源代码）\n- tests/ 目录（测试）\n- README.md（说明文档）\n- pyproject.toml（项目配置）\n- .gitignore"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "src目录下有哪些文件？"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "src"}}], ensure_ascii=False)},
            {"role": "tool", "content": "src/main.py\nsrc/utils.py\nsrc/config.py"},
            {"role": "assistant", "content": "src目录下有3个Python文件：main.py、utils.py 和 config.py。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "列出tests文件夹的内容"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "tests"}}], ensure_ascii=False)},
            {"role": "tool", "content": "tests/test_main.py\ntests/test_utils.py\ntests/conftest.py"},
            {"role": "assistant", "content": "tests文件夹下有3个文件：\n- test_main.py\n- test_utils.py\n- conftest.py（pytest配置文件）"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "Show me the project structure"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "."}}], ensure_ascii=False)},
            {"role": "tool", "content": "app/\napp/models/\napp/routes/\nrequirements.txt\nDockerfile\nREADME.md"},
            {"role": "assistant", "content": "Here's the project structure:\n- app/ - Application code with models/ and routes/ subdirectories\n- requirements.txt - Python dependencies\n- Dockerfile - Container configuration\n- README.md - Documentation"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "这个项目根目录都有啥"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {}}], ensure_ascii=False)},
            {"role": "tool", "content": "main.py\nutils/\ndata/\nconfig.yaml\nMakefile"},
            {"role": "assistant", "content": "项目根目录有：\n- main.py（入口文件）\n- utils/ 目录\n- data/ 目录\n- config.yaml（配置文件）\n- Makefile"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "帮我看看data文件夹里面放了什么数据"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "data"}}], ensure_ascii=False)},
            {"role": "tool", "content": "data/train.csv\ndata/test.csv\ndata/README.md"},
            {"role": "assistant", "content": "data文件夹里有：\n- train.csv（训练数据）\n- test.csv（测试数据）\n- README.md（数据说明文档）"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "看下config目录"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "config"}}], ensure_ascii=False)},
            {"role": "tool", "content": "path does not exist: config"},
            {"role": "assistant", "content": "config目录不存在。你可以告诉我正确的目录名，或者我帮你查看项目根目录。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "lib目录里都有什么模块？"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "lib"}}], ensure_ascii=False)},
            {"role": "tool", "content": "lib/auth.py\nlib/database.py\nlib/cache.py\nlib/__init__.py"},
            {"role": "assistant", "content": "lib目录有4个模块：\n- auth.py（认证模块）\n- database.py（数据库模块）\n- cache.py（缓存模块）\n- __init__.py（包初始化文件）"},
        ]
    },
    # ===== read_file 单工具调用 (8条) =====
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "读一下README.md"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "README.md"}}], ensure_ascii=False)},
            {"role": "tool", "content": "# MyProject\n\nA simple Python project.\n\n## Install\n\npip install -r requirements.txt"},
            {"role": "assistant", "content": "README.md的内容是一个项目说明文档，包括项目名\"MyProject\"、简介和安装方法（通过pip install -r requirements.txt安装依赖）。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "帮我看看main.py里写了什么"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "main.py"}}], ensure_ascii=False)},
            {"role": "tool", "content": "import sys\n\ndef main():\n    print(\"Hello, World!\")\n\nif __name__ == \"__main__\":\n    main()"},
            {"role": "assistant", "content": "main.py是一个简单的入口文件，定义了main()函数打印\"Hello, World!\"，并通过if __name__守卫来执行。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "看下pyproject.toml的配置"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "pyproject.toml"}}], ensure_ascii=False)},
            {"role": "tool", "content": "[project]\nname = \"myapp\"\nversion = \"0.1.0\"\nrequires-python = \">=3.10\"\ndependencies = [\"fastapi\", \"uvicorn\"]"},
            {"role": "assistant", "content": "pyproject.toml配置了项目名为myapp，版本0.1.0，要求Python>=3.10，依赖fastapi和uvicorn。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "Read the config.yaml file"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "config.yaml"}}], ensure_ascii=False)},
            {"role": "tool", "content": "database:\n  host: localhost\n  port: 5432\n  name: mydb\nserver:\n  port: 8000\n  debug: true"},
            {"role": "assistant", "content": "config.yaml contains:\n- Database config: localhost:5432, database name \"mydb\"\n- Server config: port 8000, debug mode enabled"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "打开utils.py看看有什么函数"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "src/utils.py"}}], ensure_ascii=False)},
            {"role": "tool", "content": "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b"},
            {"role": "assistant", "content": "utils.py里定义了两个函数：\n- add(a, b)：返回两数之和\n- multiply(a, b)：返回两数之积"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "帮我看看这个不存在的文件 abc.txt"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "abc.txt"}}], ensure_ascii=False)},
            {"role": "tool", "content": "file not found: abc.txt"},
            {"role": "assistant", "content": "abc.txt文件不存在。请确认文件路径是否正确。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": ".gitignore里忽略了什么？"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": ".gitignore"}}], ensure_ascii=False)},
            {"role": "tool", "content": "__pycache__/\n*.pyc\n.env\nvenv/\ndist/"},
            {"role": "assistant", "content": ".gitignore忽略了以下内容：Python缓存文件（__pycache__/、*.pyc）、环境变量文件（.env）、虚拟环境（venv/）和构建产物（dist/）。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "requirements.txt里有哪些依赖？"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "requirements.txt"}}], ensure_ascii=False)},
            {"role": "tool", "content": "flask==3.0.0\nrequests>=2.28\npydantic\nredis"},
            {"role": "assistant", "content": "项目依赖4个包：flask 3.0.0、requests 2.28+、pydantic和redis。"},
        ]
    },
    # ===== write_file 单工具调用 (6条) =====
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "帮我创建一个hello.py，里面打印hello world"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "write_file", "arguments": {"path": "hello.py", "content": "print(\"hello world\")\n"}}], ensure_ascii=False)},
            {"role": "tool", "content": "wrote 22 bytes to hello.py (overwrite)"},
            {"role": "assistant", "content": "已创建hello.py，内容是一行打印hello world的代码。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "写一个简单的README"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "write_file", "arguments": {"path": "README.md", "content": "# MyProject\n\n一个简单的Python项目。\n\n## 使用方法\n\n```bash\npython main.py\n```\n"}}], ensure_ascii=False)},
            {"role": "tool", "content": "wrote 78 bytes to README.md (overwrite)"},
            {"role": "assistant", "content": "已创建README.md，包含项目标题、简介和使用方法。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "Create a .gitignore for a Python project"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "write_file", "arguments": {"path": ".gitignore", "content": "__pycache__/\n*.pyc\n*.pyo\n.env\nvenv/\ndist/\n*.egg-info/\n.pytest_cache/\n"}}], ensure_ascii=False)},
            {"role": "tool", "content": "wrote 73 bytes to .gitignore (overwrite)"},
            {"role": "assistant", "content": "Created .gitignore with common Python exclusions: __pycache__, .env, venv/, dist/, and test cache."},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "帮我写一个简单的Flask应用到app.py"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "write_file", "arguments": {"path": "app.py", "content": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route(\"/\")\ndef index():\n    return \"Hello!\"\n\nif __name__ == \"__main__\":\n    app.run(debug=True)\n"}}], ensure_ascii=False)},
            {"role": "tool", "content": "wrote 149 bytes to app.py (overwrite)"},
            {"role": "assistant", "content": "已创建app.py，这是一个最简Flask应用，访问根路径返回\"Hello!\"，运行后在debug模式启动。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "创建一个config.json配置文件"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "write_file", "arguments": {"path": "config.json", "content": "{\n  \"host\": \"localhost\",\n  \"port\": 8080,\n  \"debug\": true\n}\n"}}], ensure_ascii=False)},
            {"role": "tool", "content": "wrote 55 bytes to config.json (overwrite)"},
            {"role": "assistant", "content": "已创建config.json，配置了host为localhost、端口8080、开启debug模式。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "帮我在tests目录下创建一个test_app.py测试文件"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "write_file", "arguments": {"path": "tests/test_app.py", "content": "def test_index():\n    assert 1 + 1 == 2\n"}}], ensure_ascii=False)},
            {"role": "tool", "content": "wrote 36 bytes to tests/test_app.py (overwrite)"},
            {"role": "assistant", "content": "已在tests目录下创建test_app.py，包含一个简单的测试用例。"},
        ]
    },
    # ===== 不需要工具的纯对话 (5条) =====
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "什么是Python的装饰器？"},
            {"role": "assistant", "content": "Python装饰器是一种用@语法修饰函数或类的特殊语法。它本质上是一个接收函数作为参数并返回新函数的高阶函数。常见用途包括日志记录、权限校验、缓存等。例如：\n\n@app.route(\"/\")\ndef index():\n    return \"hello\"\n\n这里@app.route就是一个装饰器。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "解释一下git rebase和merge的区别"},
            {"role": "assistant", "content": "merge会创建一个新的合并提交，保留完整的分支历史；rebase会把你的提交\"搬\"到目标分支的最新提交之后，让历史变成一条直线。merge更安全，rebase历史更干净。一般建议：个人分支用rebase，公共分支用merge。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么编程问题我可以帮你吗？"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "How do I use a Python list comprehension?"},
            {"role": "assistant", "content": "List comprehension is a concise way to create lists:\n\n[x * 2 for x in range(5)]  # [0, 2, 4, 6, 8]\n\nYou can add conditions:\n[x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "Python里try except怎么用"},
            {"role": "assistant", "content": "基本用法：\n\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print(\"不能除以零\")\nfinally:\n    print(\"总是执行\")\n\ntry块放可能出错的代码，except捕获特定异常，finally里的代码无论是否出错都会执行。"},
        ]
    },
    # ===== 多轮工具调用 (3条) =====
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "帮我看看项目里有什么文件，然后读一下main.py"},
            {"role": "assistant", "content": "好的，我先查看项目结构。", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "."}}], ensure_ascii=False)},
            {"role": "tool", "content": "main.py\nutils.py\nREADME.md"},
            {"role": "assistant", "content": "项目有3个文件，我来读一下main.py。", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "main.py"}}], ensure_ascii=False)},
            {"role": "tool", "content": "print(\"hello\")"},
            {"role": "assistant", "content": "项目有3个文件：main.py、utils.py和README.md。\n\nmain.py的内容很简单，只有一行print(\"hello\")。"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "先看看src目录结构，然后读一下里面的config.py"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "list_files", "arguments": {"path": "src"}}], ensure_ascii=False)},
            {"role": "tool", "content": "src/app.py\nsrc/config.py\nsrc/models.py"},
            {"role": "assistant", "content": "src下有3个文件，我来读config.py。", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "src/config.py"}}], ensure_ascii=False)},
            {"role": "tool", "content": "DB_URL = \"sqlite:///data.db\"\nDEBUG = True"},
            {"role": "assistant", "content": "src目录包含app.py、config.py和models.py。\n\nconfig.py定义了两个配置变量：\n- DB_URL：SQLite数据库路径\n- DEBUG：调试模式开启"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "你是MinCode编程助手。", "tools": TOOLS_JSON},
            {"role": "user", "content": "读一下现有的README，然后帮我重写一个更好的"},
            {"role": "assistant", "content": "好的，先看看现有的README。", "tool_calls": json.dumps([{"name": "read_file", "arguments": {"path": "README.md"}}], ensure_ascii=False)},
            {"role": "tool", "content": "# todo\nfix this later"},
            {"role": "assistant", "content": "现有README只有一行TODO，我来帮你重写。", "tool_calls": json.dumps([{"name": "write_file", "arguments": {"path": "README.md", "content": "# MyProject\n\n一个Python Web应用。\n\n## 安装\n\n```bash\npip install -r requirements.txt\n```\n\n## 运行\n\n```bash\npython main.py\n```\n"}}], ensure_ascii=False)},
            {"role": "tool", "content": "wrote 102 bytes to README.md (overwrite)"},
            {"role": "assistant", "content": "已重写README.md，新版本包含项目标题、简介、安装方法和运行说明。"},
        ]
    },
]


# ── DeepSeek 扩展 prompt ──

GENERATION_PROMPT = """为MinCode编程助手生成{batch_size}条SFT训练数据（JSONL格式，每行一个JSON）。

工具定义（所有样本的system.tools字段都用这个）：
{tools_json}

规则：
- 50%单工具调用，25%纯对话（不调工具），15%多轮调用（如list→read），10%边界情况
- 中英文混合，路径和文件内容要多样化
- assistant回复简洁

格式：
{{"conversations":[{{"role":"system","content":"你是MinCode编程助手。","tools":"<上面的工具JSON字符串>"}},{{"role":"user","content":"..."}},{{"role":"assistant","content":"","tool_calls":"[{{\\\"name\\\":\\\"read_file\\\",\\\"arguments\\\":{{\\\"path\\\":\\\"x.py\\\"}}}}]"}},{{"role":"tool","content":"文件内容"}},{{"role":"assistant","content":"总结回复"}}]}}

参考样本：
{seed_samples}

直接输出{batch_size}条JSONL，不要其他文字。"""

GENERATION_PROMPT_WRITE_FOCUS = """为MinCode编程助手生成{batch_size}条SFT训练数据（JSONL格式，每行一个JSON）。

工具定义（所有样本的system.tools字段都用这个）：
{tools_json}

**重点**：本批次侧重write_file工具调用，请确保至少60%的样本使用write_file。
write_file的场景要多样化：创建Python脚本、配置文件、文档、HTML、CSS、shell脚本、Dockerfile、yaml等。content字段要有实际完整的文件内容（不能为空）。

规则：
- 60%使用write_file工具（各种类型文件的创建场景）
- 20%多轮调用（先read/list再write，如读取现有文件后修改写入）
- 20%使用list_files或read_file
- 中英文混合，路径和文件内容要多样化
- assistant回复简洁

格式：
{{"conversations":[{{"role":"system","content":"你是MinCode编程助手。","tools":"<上面的工具JSON字符串>"}},{{"role":"user","content":"..."}},{{"role":"assistant","content":"","tool_calls":"[{{\\\"name\\\":\\\"write_file\\\",\\\"arguments\\\":{{\\\"path\\\":\\\"x.py\\\",\\\"content\\\":\\\"print('hello')\\\\n\\\"}}}}]"}},{{"role":"tool","content":"wrote 16 bytes to x.py (overwrite)"}},{{"role":"assistant","content":"总结回复"}}]}}

参考样本：
{seed_samples}

直接输出{batch_size}条JSONL，不要其他文字。"""

GENERATION_PROMPT_DIALOG_FOCUS = """为MinCode编程助手生成{batch_size}条SFT训练数据（JSONL格式，每行一个JSON）。

工具定义（所有样本的system.tools字段都用这个）：
{tools_json}

**重点**：本批次是纯对话样本，assistant不应该调用任何工具，直接回答用户的编程问题。
场景要多样化：Python语法、数据结构、算法、设计模式、Git使用、Linux命令、数据库、Web开发、调试技巧、最佳实践等。
也要包含日常问候（你好、谢谢、再见）和闲聊（天气、打招呼），这些都不应触发工具调用。

规则：
- 100%纯对话，不调用任何工具
- assistant回复简洁有价值
- 中英文混合
- 问题难度要有梯度：简单问候、基础语法、进阶概念

格式：
{{"conversations":[{{"role":"system","content":"你是MinCode编程助手。","tools":"<上面的工具JSON字符串>"}},{{"role":"user","content":"Python字典怎么遍历？"}},{{"role":"assistant","content":"可以用for循环遍历..."}}]}}

参考样本：
{seed_samples}

直接输出{batch_size}条JSONL，不要其他文字。"""


def validate_sample(sample: dict) -> bool:
    """Basic validation of a generated sample."""
    if "conversations" not in sample:
        return False
    convs = sample["conversations"]
    if not isinstance(convs, list) or len(convs) < 2:
        return False
    # Must have at least system + user
    roles = [c.get("role") for c in convs]
    if roles[0] != "system" or "user" not in roles:
        return False
    # Check tool_calls format
    for conv in convs:
        if "tool_calls" in conv and conv["tool_calls"]:
            try:
                calls = json.loads(conv["tool_calls"]) if isinstance(conv["tool_calls"], str) else conv["tool_calls"]
                if not isinstance(calls, list):
                    return False
                for call in calls:
                    if "name" not in call or call["name"] not in ("list_files", "read_file", "write_file"):
                        return False
            except (json.JSONDecodeError, TypeError):
                return False
    return True


def normalize_sample(sample: dict) -> dict:
    """Ensure tool_calls fields are JSON strings (MiniMind SFTDataset expects this)."""
    for conv in sample["conversations"]:
        if "tool_calls" in conv and not isinstance(conv["tool_calls"], str):
            conv["tool_calls"] = json.dumps(conv["tool_calls"], ensure_ascii=False)
        # Ensure tools field is string
        if "tools" in conv and not isinstance(conv["tools"], str):
            conv["tools"] = json.dumps(conv["tools"], ensure_ascii=False)
        # Ensure all expected fields exist for SFTDataset compatibility
        for field in ("content", "reasoning_content", "tools", "tool_calls"):
            if field not in conv:
                conv[field] = None
    return sample


def _compact_seed(sample: dict) -> str:
    """Compact a seed sample for the prompt — strip the bulky tools field."""
    compact = {"conversations": []}
    for msg in sample["conversations"]:
        m = {"role": msg["role"]}
        if msg.get("content"):
            m["content"] = msg["content"]
        if msg.get("tool_calls"):
            m["tool_calls"] = msg["tool_calls"]
        if msg["role"] == "system":
            m["content"] = msg.get("content", "")
            m["tools"] = "<TOOLS>"  # placeholder, described separately
        compact["conversations"].append(m)
    return json.dumps(compact, ensure_ascii=False)


def generate_batch(client: OpenAI, model: str, batch_size: int, seed_indices: list[int],
                    output_file: Path | None = None, focus: str = "balanced") -> list[dict]:
    """Call DeepSeek via streaming, parse each line as it arrives, optionally append to file."""
    picked = seed_indices[:2]
    seed_text = "\n".join(_compact_seed(SEED_SAMPLES[i]) for i in picked)

    # Select prompt based on focus
    if focus == "write":
        prompt_template = GENERATION_PROMPT_WRITE_FOCUS
    elif focus == "dialog":
        prompt_template = GENERATION_PROMPT_DIALOG_FOCUS
    else:
        prompt_template = GENERATION_PROMPT

    prompt = prompt_template.format(
        batch_size=batch_size,
        seed_samples=seed_text,
        tools_json=TOOLS_JSON,
    )

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=8192,
        stream=True,
    )

    # Accumulate streamed text, parse complete lines on the fly
    buffer = ""
    samples = []
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            buffer += delta.content
            # Try to parse complete lines from buffer
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or line.startswith("```"):
                    continue
                try:
                    sample = json.loads(line)
                    if validate_sample(sample):
                        sample = normalize_sample(sample)
                        samples.append(sample)
                        # Write immediately
                        if output_file:
                            with output_file.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        print(".", end="", flush=True)
                except json.JSONDecodeError:
                    continue

    # Handle remaining buffer (last line without trailing newline)
    buffer = buffer.strip()
    if buffer and not buffer.startswith("```"):
        try:
            sample = json.loads(buffer)
            if validate_sample(sample):
                sample = normalize_sample(sample)
                samples.append(sample)
                if output_file:
                    with output_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                print(".", end="", flush=True)
        except json.JSONDecodeError:
            pass

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate MinCode SFT data via DeepSeek API")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""),
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com",
                        help="DeepSeek API base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="Model name")
    parser.add_argument("--num_batches", type=int, default=20,
                        help="Number of generation batches")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Samples per batch (keep <=20 to fit in max_tokens)")
    parser.add_argument("--output", type=str, default="dataset/mincode_sft.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--no_seeds", action="store_true", default=False,
                        help="Do NOT include seed samples in output")
    parser.add_argument("--focus", type=str, default="balanced",
                        choices=["balanced", "write", "dialog"],
                        help="Focus area: balanced (default), write (write_file heavy), dialog (pure dialogue)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set DEEPSEEK_API_KEY or pass --api_key", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0

    # Write seed samples first
    if not args.no_seeds:
        with output_path.open("w", encoding="utf-8") as f:
            for sample in SEED_SAMPLES:
                f.write(json.dumps(normalize_sample(dict(sample)), ensure_ascii=False) + "\n")
        total_written = len(SEED_SAMPLES)
        print(f"Wrote {len(SEED_SAMPLES)} seed samples to {output_path}")
    else:
        # Append mode — don't clear existing data
        if not output_path.exists():
            output_path.write_text("")

    # Generate batches — append to file after each batch
    max_retries = 3
    for i in range(args.num_batches):
        seed_indices = random.sample(range(len(SEED_SAMPLES)), min(5, len(SEED_SAMPLES)))
        print(f"Batch {i + 1}/{args.num_batches}: generating {args.batch_size} samples...", end=" ", flush=True)

        batch = None
        for attempt in range(max_retries):
            try:
                batch = generate_batch(client, args.model, args.batch_size, seed_indices,
                                       output_file=output_path, focus=args.focus)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"retry {attempt + 1}/{max_retries} in {wait}s ({e})...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"failed after {max_retries} retries: {e}")

        if batch:
            total_written += len(batch)
            print(f" {len(batch)} samples (total: {total_written})")

        # Rate limiting
        if i < args.num_batches - 1:
            time.sleep(2)

    # Final stats
    print(f"\nDone. {total_written} samples in {output_path}")
    # Count from file
    tool_call_count = 0
    no_tool_count = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if any("tool_calls" in c and c["tool_calls"] for c in sample["conversations"]):
                tool_call_count += 1
            else:
                no_tool_count += 1
    print(f"  With tool calls: {tool_call_count}")
    print(f"  Pure dialogue:   {no_tool_count}")


if __name__ == "__main__":
    main()
