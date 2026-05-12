# Minicode

**从零训练一个会用工具的编程 Agent — 64M 参数，纯 CPU，SFT + RL 全流程可复现。**

Minicode 将 [MiniMind](https://github.com/jingyaogong/minimind)（64M 小语言模型）接入简化版 [ApeCode](https://github.com/ApeCodeAI/apecode) Agent 框架，构建了一个完整的 tool-calling 训练管线：数据生成 → LoRA 微调 → GRPO 强化学习，全部在消费级硬件上完成。

## 核心成果

| 阶段 | 方法 | 工具调用准确率 | 提升 |
|------|------|---------------|------|
| Base (未训练) | — | 40.0% | — |
| Phase 2: SFT | LoRA, 272 样本 | 62.5% | +22.5% |
| **Phase 3: RL** | **GRPO/CISPO, 119 样本** | **80.0%** | **+17.5%** |

> 评测基于 40 条测试用例（5 类 × 8 条），与训练数据零重叠。

## 架构

```
User Input
    │
    ▼
CLI (cli.py) ──▶ Agent Loop (agent.py)
                      │
                      ▼
              Model Adapter (model_adapter.py)
                 /          \
    MiniMind API Client    Local Inference
    (OpenAI-compat)        (Direct PyTorch)
                      │
                      ▼
              Tool Execution (tools.py)
              [list_files, read_file, write_file]
```

**模型规格**: 64M 参数 | vocab=6400 | hidden=768 | 8 layers | max_seq_len=768

### MiniMind ↔ ApeCode 通讯桥梁

MiniMind 和 ApeCode 原本是两个独立项目。Minicode 的第一个工程挑战是打通二者的通讯链路：

```
┌──── Minicode Agent 进程 ────┐          ┌──── MiniMind 服务器进程 ────┐
│                              │          │                             │
│  Agent.run()                 │          │  FastAPI (uvicorn :8998)    │
│    │                         │          │    │                        │
│    ├─ 组装 messages (dict)    │          │    ├─ apply_chat_template() │
│    ├─ tools → JSON schema    │   HTTP   │    ├─ model.generate()      │
│    ▼                         │   POST   │    ├─ parse_response()      │
│  openai.OpenAI(base_url)     │─────────▶│    │  提取 tool_calls       │
│  client.chat.completions     │◀─────────│    └─ 返回标准 JSON         │
│    │                         │   JSON   │                             │
│    ├─ 解析 tool_calls        │          └─────────────────────────────┘
│    ├─ 执行工具 → 追加结果     │
│    └─ 循环直到无 tool_call   │
└──────────────────────────────┘
```

**关键设计**：MiniMind 通过 `<tool_call>{"name":..., "arguments":...}</tool_call>` XML 标签输出工具调用，其 API 服务器将其解析为 OpenAI 兼容的结构化 `tool_calls` 对象。Minicode Agent 侧使用标准 OpenAI SDK 通信，对底层模型完全透明。

> 详细的通讯机制分析见 `docs/communication-deep-dive.md`，HTTP 桥梁的逐层拆解见 `docs/http-bridge-guide.md`。

## 项目亮点

### 1. 完整的 SFT → RL 训练管线

整个训练流程在单台 Mac CPU 上完成，无需 GPU：

- **数据生成** (`scripts/generate_sft_data.py`) — 30 条手写种子 + DeepSeek API 扩展至 272 条
- **LoRA 微调** (`scripts/train_lora.py`) — rank=16, 0.4M 可训参数 (0.61%), 5 epoch ~30min
- **GRPO 强化学习** (`scripts/train_rl.py`) — 全参数训练, 119 条手写 prompt, 3 epoch ~2h
- **自动化评测** (`scripts/eval_toolcall.py`) — 40 条测试集, 分类准确率统计

### 2. Rule-based Reward 设计

RL 阶段不依赖神经网络奖励模型，采用结构化规则打分：

| 信号 | 奖励 | 说明 |
|------|------|------|
| 工具名正确 | +2.0 | 核心信号 |
| JSON 参数有效 | +1.0 | 结构正确性 |
| 必选参数齐全 | +1.0 | 功能正确性 |
| 参数值匹配 ground truth | +1.0 | 语义正确性 |
| 正确不调用工具 | +2.0 | 抑制误触发 |
| 错误工具 / 幻觉工具名 | -1.0 / -1.5 | 惩罚混淆 |

满分 +5.0，通过 `match_args_score()` 对预测参数和 ground truth 做精确/模糊匹配。

### 3. Mock Tool Execution

RL rollout 中使用确定性 mock 替代真实工具执行，实现无副作用的多轮交互：

```python
MOCK_RESULTS = {
    "list_files": lambda args: {"files": ["README.md", "main.py", ...], "path": args["path"]},
    "read_file":  lambda args: {"content": f"# File: {args['path']}\n\nSample file content."},
    "write_file": lambda args: {"status": "ok", "bytes_written": len(args["content"])},
}
```

模型生成 `<tool_call>` → mock 执行 → 返回 `<tool_response>` → 模型继续生成，最多 3 轮。

### 4. 关键实验发现

- **数据质量 > 数据数量**：272 条高质量 SFT (62.5%) 优于 482 条含噪声数据 (47.5%)
- **RL 数据量有阈值**：39 条 RL 样本无效 (62.5% → 62.5%)，扩至 119 条后显著提升 (+17.5%)
- **增量训练不可行**：小模型上只用新数据 resume 训练会导致灾难性遗忘 (73.3% → 33.3%)
- **多轮工具调用未涌现**：单轮准确但不会自发发起第二次 tool call，需专门的多轮训练数据

## 项目结构

```
mincode/
├── src/mincode/           # Agent 框架（7 个模块）
│   ├── agent.py           #   Agent loop：LLM → tool call → execute → 循环
│   ├── model_adapter.py   #   MiniMind 适配器（API / 本地推理）
│   ├── tools.py           #   3 个内置工具定义与执行
│   ├── cli.py             #   终端 REPL 入口
│   └── ...
├── scripts/               # 训练与评测脚本
│   ├── generate_sft_data.py   # SFT 数据生成（种子 + API 扩展）
│   ├── train_lora.py          # LoRA 微调
│   ├── train_rl.py            # GRPO/CISPO 强化学习
│   └── eval_toolcall.py       # Tool-calling 评测
├── dataset/               # 训练数据（SFT 272条 + RL 119条）
├── eval/                  # 评测结果、测试集、可视化
├── docs/                  # 各阶段技术文档与日志
│   ├── progress.md            # 全项目进展记录
│   └── phase3-rl-log.md      # RL 训练详细日志
└── out/                   # 模型权重（gitignored，需自行训练）
```

## 快速开始

### 前置依赖

- Python >= 3.10
- [MiniMind](https://github.com/jingyaogong/minimind) 克隆至 `../minimind/`（sibling 目录）

```bash
# 安装依赖
pip install torch transformers

# 训练 LoRA（~30min on CPU）
python scripts/train_lora.py

# RL 训练（~2h on CPU）
python scripts/train_rl.py --from_weight mincode_v1 --output_name mincode_v1_rl2 \
    --data_path dataset/mincode_rl_combined.jsonl

# 评测
python scripts/eval_toolcall.py --weight mincode_v1_rl2
```

### 运行 Agent

```bash
# 启动 MiniMind API 服务（在 ../minimind/ 目录）
python scripts/serve_openai_api.py

# 运行 Minicode agent
python -m mincode
```

## 各阶段评测详情

### Phase 3 最优模型 (v1 RL2) 分类准确率

| 类别 | 准确率 | 说明 |
|------|--------|------|
| list_files | 7/8 (88%) | 目录查看 |
| read_file | 7/8 (88%) | 文件读取 |
| write_file | 8/8 (100%) | 文件创建/写入 |
| no_tool | 6/8 (75%) | 纯对话，不触发工具 |
| edge | 4/8 (50%) | 模糊意图、边界场景 |

### 全模型对比

```
Base (40%) ──SFT──▶ v1 (62.5%) ──RL──▶ v1_rl2 (80.0%)  ← best
                  ▶ v2 (47.5%)   ← 数据量↑但质量↓
```

## 已知局限

- **64M 模型容量有限**：参数值生成（如文件内容）质量受限
- **768 token 上下文**：复杂多轮对话容易超出预算
- **单轮工具调用**：能正确选择工具，但不会在一次对话中连续调用多个工具
- **中文偏向**：训练数据中英比 ~3:1，英文场景表现略弱

## 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) — 64M 小语言模型，提供 base 模型与 GRPO 训练框架
- [ApeCode](https://github.com/ApeCodeAI/apecode) — 终端编程 Agent，提供 harness 架构参考

## License

[Apache-2.0](LICENSE)
