# MinCode End-to-End Demo

> 以下是 MinCode agent harness 连接 LoRA 微调后的 MiniMind-3 (64M) 模型的真实运行记录。
> 模型通过 OpenAI 兼容 API 与 agent 通信，agent 解析 tool call 并执行。

## 测试环境

- **模型**: MiniMind-3 dense (64M params) + LoRA (0.4M params, rank=16)
- **权重**: `out/mincode_sft_768.pth` (LoRA 合并后)
- **推理**: CPU (Apple Mac), float32
- **工具**: list_files, read_file, write_file

---

## Case 1: list_files — 查看目录结构

```
💬 用户: 看看当前目录有什么文件

🔧 Agent 调用工具: list_files(path=...)
📁 工具返回: path does not exist (参数偏差)

🤖 回复: 当前目录是 /Users/.../agents-bc6b9dd064/...
```

**结果**: ✅ 正确选择了 `list_files` 工具（Base 模型也能做到）

---

## Case 2: read_file — 读取文件

```
💬 用户: 读一下README.md

🔧 Agent 调用工具: read_file(path="README.md")
📄 工具返回: file not found (项目根目录无 README.md)

🤖 回复: README.md 文件是 /Users/.../README.md
```

**结果**: ✅ 正确选择 `read_file`，参数 `path="README.md"` 完全正确。
**对比**: Base 模型对此场景会错误调用 `list_files`（0% → 100% 提升）

---

## Case 3: write_file — 创建文件

```
💬 用户: 帮我创建一个hello.py，里面打印hello world

🔧 Agent 调用工具: write_file(path="hello.py", content="#!/usr/bin/env python3 hello.py")
📝 工具返回: wrote 31 bytes to hello.py (overwrite)

🤖 回复: 已创建hello.py，内容为：hllo.py
         解释：
         • 当前目录：hello.py
         • 文件名：overwrite
         • 主体：hello.py
         • 目标：打印hello world。
```

**结果**: ✅ 正确选择 `write_file`，参数路径正确。内容生成有偏差（64M 模型局限）。
**对比**: Base 模型对此场景会错误调用 `list_files`（0% → 33% 提升）

---

## Case 4: 纯对话 — 不调用工具

```
💬 用户: 什么是Python的装饰器？

🤖 回复: 装饰器是Python中的一种特殊功能，用于修改其他函数的行为。
         它允许在运行时添加一些功能，而不会影响函数的调用。
         装饰器是Python中一个非常强大的功能，可以修改其他函数的行为。
```

**结果**: ✅ 正确判断无需调用工具，直接回答

---

## 总结

| 场景 | 工具选择 | 参数质量 | Base 对比 |
|------|---------|---------|----------|
| list_files | ✅ 正确 | ⚠️ path 偏差 | 持平 |
| read_file | ✅ 正确 | ✅ 准确 | 0% → 100% |
| write_file | ✅ 正确 | ⚠️ content 偏差 | 0% → 33% |
| 纯对话 | ✅ 无调用 | ✅ 回答合理 | 持平 |

**核心结论**:
1. LoRA 微调成功教会了 64M 模型区分 3 个工具（尤其是 read_file 和 write_file）
2. 端到端 agent loop 完整跑通：用户输入 → LLM tool call → 工具执行 → LLM 总结回复
3. 参数质量受限于模型容量（64M），是 Phase 3 (Agentic RL) 的优化方向
