# 本地模型 HTTP 桥梁指南：从 PyTorch 到 OpenAI 兼容 API

> 本文档面向只用过 OpenAI SDK 调用云端 API、但不了解服务端代码的读者。
> 目标：系统讲解如何把一个本地 PyTorch 模型包装成 OpenAI 兼容的 HTTP API，使得你可以用同样的 `openai.OpenAI(base_url=...)` 客户端来调用它。

---

## 目录

1. [整体架构：你已经熟悉的部分 vs 你不熟悉的部分](#1-整体架构)
2. [核心依赖库速览](#2-核心依赖库速览)
3. [第一层：uvicorn — HTTP 服务器](#3-第一层uvicorn--http-服务器)
4. [第二层：FastAPI — Web 框架](#4-第二层fastapi--web-框架)
5. [第三层：Pydantic — 请求/响应数据校验](#5-第三层pydantic--请求响应数据校验)
6. [第四层：apply_chat_template — 结构化消息 → 纯文本](#6-第四层apply_chat_template--结构化消息--纯文本)
7. [第五层：model.generate — 模型推理](#7-第五层modelgenerate--模型推理)
8. [第六层：parse_response — 纯文本 → 结构化响应](#8-第六层parse_response--纯文本--结构化响应)
9. [完整代码逐行解读：serve_openai_api.py](#9-完整代码逐行解读)
10. [start_minimind_server.py：为什么需要一个启动器](#10-start_minimind_serverpy为什么需要一个启动器)
11. [客户端如何使用：从 OpenAI SDK 到本地模型](#11-客户端如何使用)
12. [自己动手：最小可运行示例](#12-自己动手最小可运行示例)
13. [常见问题与踩坑记录](#13-常见问题与踩坑记录)

---

## 1. 整体架构

当你用 OpenAI SDK 调用 GPT-4 时，通信链路是：

```
你的 Python 代码
    │  openai.OpenAI(api_key="sk-...")
    │  client.chat.completions.create(messages=[...])
    ▼
[ HTTP POST ] ──────────────────────────► OpenAI 云服务器
                                           │
                                           │ (你看不到的部分)
                                           │ 接收 JSON → 推理 → 返回 JSON
                                           ▼
[ HTTP Response ] ◄──────────────────────  返回 choices[0].message
```

你只接触了**客户端**（左侧），云服务器内部发生了什么对你是黑盒。

现在我们要做的就是**自己写右侧的服务器**，让链路变成：

```
你的 Python 代码（MinCode harness）
    │  openai.OpenAI(base_url="http://localhost:8998/v1")
    │  client.chat.completions.create(messages=[...], tools=[...])
    ▼
[ HTTP POST ] ──► 本地 serve_openai_api.py（FastAPI + uvicorn）
                      │
                      │  ① Pydantic 校验 JSON → ChatRequest 对象
                      │  ② apply_chat_template() 把结构化消息渲染成纯文本
                      │  ③ tokenizer() 编码为 token IDs
                      │  ④ model.generate() 生成新 token
                      │  ⑤ tokenizer.decode() 解码回文本
                      │  ⑥ parse_response() 提取 tool_calls
                      │  ⑦ 组装 OpenAI 格式 JSON 返回
                      ▼
[ HTTP Response ] ◄── {"choices": [{"message": {"role":"assistant", ...}}]}
```

**关键洞察**：OpenAI SDK 并不关心服务器是谁——它只是发 HTTP 请求、收 JSON 响应。只要你的服务器返回的 JSON 结构正确，SDK 就能正常工作。这就是为什么我们可以用 `base_url` 指向本地服务器。

---

## 2. 核心依赖库速览

| 库                | 作用                                      | 类比                                 |
| ---------------- | --------------------------------------- | ---------------------------------- |
| **uvicorn**      | ASGI HTTP 服务器，监听端口、处理网络连接               | 相当于 Nginx/Apache，但专为 Python 异步框架设计 |
| **FastAPI**      | Web 框架，定义路由（URL → 函数的映射）、自动序列化 JSON     | 相当于 Flask，但更现代，自带类型校验              |
| **Pydantic**     | 数据模型校验，定义请求/响应的字段和类型                    | 相当于 dataclass + 自动 JSON 校验         |
| **transformers** | HuggingFace 的模型加载库，提供 tokenizer 和 model | 你可能已经熟悉                            |
| **torch**        | PyTorch，模型推理引擎                          | 你可能已经熟悉                            |

它们的关系是**逐层嵌套**的：

```
uvicorn（网络层）
  └── FastAPI（ 层）
        └── Pydantic（数据校验层）
              └── transformers + torch（模型推理层）
```

---

## 3. 第一层：uvicorn — HTTP 服务器

### 它是什么

uvicorn 是一个 **ASGI (Asynchronous Server Gateway Interface)** 服务器。你可以把它理解为一个「翻译官」：
- 它监听某个端口（比如 8998）
- 当有 HTTP 请求进来时，它把原始的 HTTP 字节流翻译成 Python 对象
- 把 Python 函数的返回值翻译回 HTTP 响应

### 在 MiniMind 中的使用

```python
# serve_openai_api.py 最后一行
uvicorn.run(app, host="0.0.0.0", port=8998)
```

这一行做了三件事：
1. **启动 HTTP 服务器**，监听所有网络接口（`0.0.0.0`）的 8998 端口
2. 把 `app`（FastAPI 应用实例）注册为请求处理器
3. **阻塞运行**——程序会停在这里持续监听，直到你 Ctrl+C

### 你需要知道的

- `host="0.0.0.0"` 表示接受所有来源的连接（如果改成 `"127.0.0.1"` 则只接受本机连接）
- `port=8998` 是自定义的端口号，OpenAI 官方用的是 443（HTTPS），我们用 8998 只是为了避免冲突
- uvicorn 本身不关心你的业务逻辑，它只负责网络 I/O

---

## 4. 第二层：FastAPI — Web 框架

### 它是什么

FastAPI 是一个 Python Web 框架，核心功能是：**定义 URL 路径和 HTTP 方法到 Python 函数的映射**（称为「路由」）。

### 最小示例

```python
from fastapi import FastAPI

app = FastAPI()  # 创建应用实例

@app.post("/v1/chat/completions")  # 定义路由：POST 请求 + URL 路径
async def chat_completions(request: ChatRequest):  # 处理函数
    # ... 处理逻辑 ...
    return {"choices": [...]}  # 返回值自动序列化为 JSON
```

### 拆解

| 元素                                  | 含义                                                              |
| ----------------------------------- | --------------------------------------------------------------- |
| `app = FastAPI()`                   | 创建一个 Web 应用实例。uvicorn 拿到的 `app` 就是它。                            |
| `@app.post("/v1/chat/completions")` | 装饰器，意思是「当收到 `POST /v1/chat/completions` 请求时，调用下面这个函数」           |
| `async def`                         | 异步函数。FastAPI 支持 async/await，但我们的模型推理是同步阻塞的（CPU 计算），所以实际上并没有异步优势 |
| `request: ChatRequest`              | 参数类型标注。FastAPI 看到类型是 Pydantic 模型后，会自动从 HTTP 请求体中解析 JSON 并校验     |
| `return {...}`                      | 返回一个 dict，FastAPI 自动将其序列化为 JSON 响应                              |

### 为什么路径是 `/v1/chat/completions`

因为 OpenAI SDK 发送请求时，会把 `base_url + "/chat/completions"` 拼接成完整 URL。当你设置 `base_url="http://localhost:8998/v1"` 时，SDK 实际发送到：

```
POST http://localhost:8998/v1/chat/completions
```

所以我们的 FastAPI 路由必须匹配这个路径。**路径必须一致，这是兼容 OpenAI SDK 的关键。**

---

## 5. 第三层：Pydantic — 请求/响应数据校验

### 它是什么

Pydantic 的 `BaseModel` 让你用 Python 类来定义「JSON 应该长什么样」。FastAPI 会自动用它来：
1. 解析请求体中的 JSON
2. 校验字段类型
3. 填充默认值
4. 如果校验失败，自动返回 422 错误

### MiniMind 中的定义

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    model: str                          # 必填，如 "minimind"
    messages: list                      # 必填，消息列表
    temperature: float = 0.7            # 可选，默认 0.7
    top_p: float = 0.92                 # 可选，默认 0.92
    max_tokens: int = 8192              # 可选，默认 8192
    stream: bool = True                 # 可选，默认 True（⚠️ 这个默认值导致了问题）
    tools: list = []                    # 可选，工具定义列表
    open_thinking: bool = False         # 可选，是否开启思考
    chat_template_kwargs: dict = None   # 可选，模板参数
```

### 当 OpenAI SDK 发来请求时

SDK 发送的 JSON 大致如下：

```json
{
  "model": "minimind",
  "messages": [
    {"role": "system", "content": "你是MinCode编程助手..."},
    {"role": "user", "content": "列出当前目录的文件"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "list_files", ...}},
    ...
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": false,
  "tool_choice": "auto"
}
```

Pydantic 会自动做以下事情：
- `model` → 填入 `"minimind"`
- `messages` → 填入消息列表
- `temperature` → 填入 `0.7`
- `stream` → 填入 `false`（覆盖默认的 `True`）
- `tool_choice` → ChatRequest 没有定义这个字段，**Pydantic 会静默忽略它**（不会报错）
- `top_p` → SDK 没发这个字段，使用默认值 `0.92`

### 与 dataclass 的区别

| 特性 | dataclass | Pydantic BaseModel |
|---|---|---|
| JSON 自动解析 | 不支持 | 自动从 JSON 构建实例 |
| 类型校验 | 不校验 | 自动校验类型，不匹配时报错 |
| 未知字段 | 需要手动处理 | 默认忽略 |
| 默认值 | 支持 | 支持 |

**一句话**：Pydantic 就是 dataclass + JSON 解析 + 类型校验。

---

## 6. 第四层：apply_chat_template — 结构化消息 → 纯文本

### 为什么需要这一步

PyTorch 模型只认识 token ID 序列（整数数组），不认识 JSON 结构。但 OpenAI SDK 发来的是结构化的 `messages` 列表。需要有人把结构化消息「扁平化」成一段纯文本，再喂给 tokenizer 编码。

这个「翻译规则」就写在 `tokenizer_config.json` 的 `chat_template` 字段中，是一段 **Jinja2 模板**。

### 调用方式

```python
new_prompt = tokenizer.apply_chat_template(
    messages,                    # 结构化消息列表
    tokenize=False,              # 返回文本而非 token IDs
    add_generation_prompt=True,  # 在末尾添加 "<|im_start|>assistant\n"
    tools=tools or None,         # 工具定义（可选）
    open_thinking=open_thinking  # 是否开启思考模式
)
```

### 转换示例

**输入**（结构化 JSON）：

```json
{
  "messages": [
    {"role": "system", "content": "你是MinCode编程助手。工作目录:/home/user"},
    {"role": "user", "content": "列出当前目录的文件"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "list_files", "description": "列出目录中的文件", ...}}
  ]
}
```

**输出**（纯文本）：

```
<|im_start|>system
你是MinCode编程助手。工作目录:/home/user

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "list_files", "description": "列出目录中的文件", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
列出当前目录的文件<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### 关键细节

1. **系统消息被合并了**：当有 `tools` 时，模板会把 system content 和工具说明合并到一个 `<|im_start|>system` 块中
2. **工具定义被 JSON 序列化**后嵌入到 `<tools>` XML 标签中
3. **工具调用指令**写死在模板里："For each function call, return a json object..."
4. **`add_generation_prompt=True`** 在末尾插入 `<|im_start|>assistant\n<think>\n\n</think>\n\n`，告诉模型「现在该你说话了」
5. **`<think>` 标签**总是会插入——即使不开 thinking 模式，也会插入空的 `<think>\n\n</think>`

### Jinja2 模板语法简介

如果你好奇模板本身的语法，这是其核心逻辑的伪代码：

```python
# 伪代码，对应 tokenizer_config.json 中的 chat_template
if tools:
    print("<|im_start|>system")
    if messages[0].role == "system":
        print(messages[0].content)
    print("# Tools\n\nYou may call one or more functions...")
    print("<tools>")
    for tool in tools:
        print(json.dumps(tool))
    print("</tools>")
    print("调用指令...")
    print("<|im_end|>")

for message in messages:
    if message.role == "user":
        print(f"<|im_start|>user\n{message.content}<|im_end|>")
    elif message.role == "assistant":
        print(f"<|im_start|>assistant\n<think>...</think>\n{message.content}")
        if message.tool_calls:
            for call in message.tool_calls:
                print(f"<tool_call>\n{json.dumps(call)}\n</tool_call>")
        print("<|im_end|>")
    elif message.role == "tool":
        print(f"<tool_response>\n{message.content}\n</tool_response>")

if add_generation_prompt:
    print("<|im_start|>assistant\n<think>\n\n</think>\n\n")
```

---

## 7. 第五层：model.generate — 模型推理

### 完整流程

```python
# 步骤 1：把纯文本编码为 token IDs
inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
# inputs["input_ids"] 形如 tensor([[1, 356, 2847, 19, ...]])
# inputs["attention_mask"] 形如 tensor([[1, 1, 1, 1, ...]])

# 步骤 2：模型生成
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_ids"],            # 输入 token IDs
        max_length=inputs["input_ids"].shape[1] + request.max_tokens,  # 最大总长度
        do_sample=True,                 # 使用采样（而非 greedy）
        temperature=request.temperature,
        top_p=request.top_p,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,  # 遇到 <|im_end|> 停止
    )

# 步骤 3：只取新生成的部分，解码回文本
answer = tokenizer.decode(
    generated_ids[0][inputs["input_ids"].shape[1]:],  # 截掉输入部分
    skip_special_tokens=True
)
```

### 关键参数解释

| 参数 | 含义 |
|---|---|
| `max_length` | 输入 + 输出的最大 token 总数。如果输入有 500 个 token，max_tokens=4096，则模型最多生成到 4596 个 token |
| `do_sample=True` | 随机采样。如果 False 则每次选概率最高的 token（greedy），输出确定性 |
| `temperature` | 控制随机性。越低越确定，越高越随机 |
| `top_p` | 核采样。只从累计概率前 p 的 token 中采样 |
| `eos_token_id` | 遇到这个 token 就停止生成。MiniMind 的 eos 是 `<|im_end|>`（ID=2） |
| `pad_token_id` | 填充 token ID，batch 推理时用于对齐序列长度 |

### generate() 的输出

`generated_ids` 是一个 tensor，包含**输入 + 新生成**的全部 token IDs。比如：

```
输入:  [1, 356, 2847, 19, ...]  (500 个 token)
输出:  [1, 356, 2847, 19, ..., 4521, 78, 2]  (500 + 120 个 token)
                                 ↑ 新生成部分 ↑ eos
```

所以用 `generated_ids[0][inputs["input_ids"].shape[1]:]` 截掉前 500 个输入 token，只保留新生成的部分。

---

## 8. 第六层：parse_response — 纯文本 → 结构化响应

### 为什么需要这一步

模型生成的是**纯文本**，可能长这样：

```
我来帮你查看当前目录的文件。
<tool_call>
{"name": "list_files", "arguments": {"path": "."}}
</tool_call>
```

但 OpenAI SDK 期望收到的是**结构化 JSON**：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "我来帮你查看当前目录的文件。",
      "tool_calls": [{
        "id": "call_1715000000_0",
        "type": "function",
        "function": {
          "name": "list_files",
          "arguments": "{\"path\": \".\"}"
        }
      }]
    }
  }]
}
```

`parse_response()` 就是做这个转换的。

### 代码解析

```python
def parse_response(text):
    # ── 1. 提取 <think>...</think> 思考内容 ──
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    # ── 2. 提取所有 <tool_call>...</tool_call> ──
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())  # 解析 JSON
            tool_calls.append({
                "id": f"call_{int(time.time())}_{i}",  # 生成唯一 ID
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)
                }
            })
        except Exception:
            pass  # JSON 解析失败则跳过

    # ── 3. 从文本中移除 tool_call 标签，剩下的就是 content ──
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)

    return text.strip(), reasoning_content, tool_calls or None
```

### 返回值

```python
content, reasoning_content, tool_calls = parse_response(answer)
# content = "我来帮你查看当前目录的文件。"
# reasoning_content = None (或思考过程文本)
# tool_calls = [{"id": "call_...", "type": "function", "function": {...}}] 或 None
```

---

## 9. 完整代码逐行解读

下面是 `serve_openai_api.py` 中非流式（`stream=False`）分支的完整逻辑，也是 MinCode 实际使用的路径：

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # ─── 分支判断：流式 vs 非流式 ───
        if request.stream:
            # ... 流式处理（MinCode 不使用）...
        else:
            # ─── 步骤 1：结构化消息 → 纯文本 ───
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )[-request.max_tokens:]
            #   ↑ ⚠️ 截断：只保留末尾 max_tokens 个字符
            #   这是一个 BUG：它截的是字符数而非 token 数
            #   如果 max_tokens 太小，会把工具定义截掉

            # ─── 步骤 2：文本 → token IDs ───
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

            # ─── 步骤 3：模型推理 ───
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                # ─── 步骤 4：解码新生成的 token ───
                answer = tokenizer.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

            # ─── 步骤 5：纯文本 → 结构化数据 ───
            content, reasoning_content, tool_calls = parse_response(answer)

            # ─── 步骤 6：组装 OpenAI 格式响应 ───
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 数据流可视化

```
OpenAI SDK 发送的 JSON
    │
    ▼ (FastAPI 自动解析)
ChatRequest 对象
    │
    │  request.messages, request.tools
    ▼
apply_chat_template()
    │
    │  "<|im_start|>system\n你是MinCode...<tools>...</tools>...<|im_start|>user\n..."
    ▼
tokenizer()
    │
    │  tensor([[1, 356, 2847, 19, ...]])
    ▼
model.generate()
    │
    │  tensor([[..., 4521, 78, 21, 3456, 22, 2]])   ← 21=<tool_call>, 22=</tool_call>
    ▼
tokenizer.decode()
    │
    │  "我来帮你查看...\n<tool_call>\n{\"name\":\"list_files\"...}\n</tool_call>"
    ▼
parse_response()
    │
    │  content = "我来帮你查看..."
    │  tool_calls = [{"id":"call_...", "function":{"name":"list_files",...}}]
    ▼
组装 OpenAI 格式 JSON → HTTP 响应 → OpenAI SDK 解析为 Python 对象
```

---

## 10. start_minimind_server.py：为什么需要一个启动器

### 问题

MiniMind 原始的 `serve_openai_api.py` 有几个问题导致不能直接用于 MinCode：

1. **`.half()` 在 CPU 上会崩溃**：原始代码 `model.half().eval().to(device)` 使用 float16，但 CPU 不支持 float16 矩阵乘法
2. **`__main__` 守卫**：模型加载在 `if __name__ == "__main__":` 块里，无法被外部 import
3. **依赖 argparse**：直接运行时会解析命令行参数，import 时会与调用方的参数冲突
4. **路径依赖**：使用相对路径 `../model`、`../out`，必须从特定目录执行

### 解决方案：importlib 动态加载

`start_minimind_server.py` 的策略是：**自己加载模型，然后把模型注入到 serve_openai_api 模块的全局变量中**。

```python
# ── 1. 自己加载模型（可以控制 float32/float16）──
tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
model = model.float().eval()  # float32 for CPU ← 关键修改

# ── 2. 用 importlib 动态加载 serve_openai_api.py ──
import importlib.util

# 创建模块规格（不执行）
spec = importlib.util.spec_from_file_location(
    "serve_openai_api",                                    # 模块名（随意取）
    os.path.join(MINIMIND_DIR, "scripts", "serve_openai_api.py"),  # 文件路径
)
# 创建空模块对象
server_module = importlib.util.module_from_spec(spec)

# ── 3. 预注入全局变量 ──
server_module.model = model        # 注入我们自己加载的模型
server_module.tokenizer = tokenizer
server_module.device = device

# ── 4. 绕过 __main__ 守卫 ──
server_module.__name__ = "serve_openai_api"  # 不是 "__main__"，所以 if 块不会执行

# ── 5. 执行模块代码 ──
spec.loader.exec_module(server_module)
# 这会执行 serve_openai_api.py 的顶层代码：
#   - import 语句
#   - app = FastAPI()
#   - class ChatRequest(BaseModel): ...
#   - def parse_response(...): ...
#   - @app.post("/v1/chat/completions") ...
# 但 NOT 执行 if __name__ == "__main__": 块

# ── 6. 再次注入（exec_module 可能覆盖了全局变量）──
server_module.model = model
server_module.tokenizer = tokenizer
server_module.device = device

# ── 7. 启动服务器 ──
uvicorn.run(server_module.app, host="0.0.0.0", port=args.port)
```

### 为什么要注入两次

`exec_module()` 执行模块的顶层代码时，如果模块里有 `model = None` 之类的赋值（虽然 serve_openai_api.py 没有，但为了安全），会覆盖我们预注入的值。所以执行完之后再注入一次确保一定生效。

### importlib 关键概念

```python
# 普通 import（这在 serve_openai_api.py 上不可用，因为它不在 Python 包路径中）
import serve_openai_api  # ModuleNotFoundError!

# importlib 动态加载（可以加载任意路径的 .py 文件）
spec = importlib.util.spec_from_file_location("模块名", "文件路径")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # 执行文件中的代码
# 之后 module.app, module.parse_response 等都可以访问
```

---

## 11. 客户端如何使用

了解了服务端之后，再看客户端就很简单了。MinCode 的 `model_adapter.py` 所做的事情：

```python
from openai import OpenAI

# 创建客户端，指向本地服务器
client = OpenAI(
    api_key="minimind",                    # 任意字符串，MiniMind 不校验
    base_url="http://localhost:8998/v1",   # 指向本地服务器
    timeout=120,
)

# 发送请求——和调用 GPT-4 完全一样
response = client.chat.completions.create(
    model="minimind",
    messages=[
        {"role": "system", "content": "你是MinCode编程助手..."},
        {"role": "user", "content": "列出当前目录的文件"},
    ],
    tools=[
        {"type": "function", "function": {"name": "list_files", ...}},
    ],
    tool_choice="auto",
    temperature=0.7,
    max_tokens=4096,
    stream=False,      # ⚠️ 必须显式设为 False
)

# 读取响应
message = response.choices[0].message
print(message.content)      # "我来帮你查看当前目录..."
print(message.tool_calls)   # [ChatCompletionMessageToolCall(...)]
```

### ⚠️ stream=False 的重要性

MiniMind 的 `ChatRequest` 定义了 `stream: bool = True`（默认值）。如果 SDK 不发送 `stream` 字段，服务端会默认走流式分支，返回 SSE (Server-Sent Events) 格式的响应，而 SDK 期望的是普通 JSON，导致解析失败。

**必须显式传 `stream=False`**。

---

## 12. 自己动手：最小可运行示例

如果你想从零开始理解整个流程，下面是一个**最小化**的 HTTP 桥梁示例（约 50 行），不依赖 MiniMind 的复杂代码：

```python
"""minimal_server.py — 最小 OpenAI 兼容 API 服务器"""
import json
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 1. 加载模型 ──
MODEL_NAME = "jingyaogong/minimind-3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = model.float().eval()

# ── 2. 定义请求格式 ──
class ChatRequest(BaseModel):
    model: str = "minimind"
    messages: list
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    tools: list = []

# ── 3. 创建 FastAPI 应用 ──
app = FastAPI()

# ── 4. 定义路由 ──
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 结构化消息 → 纯文本
    prompt = tokenizer.apply_chat_template(
        request.messages, tokenize=False,
        add_generation_prompt=True,
        tools=request.tools or None,
    )
    # 纯文本 → token IDs
    inputs = tokenizer(prompt, return_tensors="pt")
    # 模型推理
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 解码新生成的 token
    answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # 返回 OpenAI 格式
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
    }

# ── 5. 启动服务器 ──
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8998)
```

运行 `python minimal_server.py` 后，你就可以用 OpenAI SDK 连接了：

```python
from openai import OpenAI
client = OpenAI(api_key="x", base_url="http://localhost:8998/v1")
r = client.chat.completions.create(model="minimind", messages=[{"role":"user","content":"你好"}], stream=False)
print(r.choices[0].message.content)
```

> 这个最小示例省略了 tool_call 解析（parse_response）、流式传输等功能。MiniMind 的完整版在此基础上增加了这些能力。

---

## 13. 常见问题与踩坑记录

### Q1：为什么 `base_url` 末尾要有 `/v1`？

OpenAI SDK 会在 `base_url` 后面拼接 `/chat/completions`。所以：
- `base_url="http://localhost:8998/v1"` → 请求发送到 `http://localhost:8998/v1/chat/completions` ✅
- `base_url="http://localhost:8998"` → 请求发送到 `http://localhost:8998/chat/completions` ❌（路径不匹配）

### Q2：`api_key` 可以随便填吗？

MiniMind 的 FastAPI 服务器没有做认证校验，所以 api_key 可以是任意字符串。但 OpenAI SDK **要求** api_key 参数不能为空，所以我们填一个占位符 `"minimind"`。

### Q3：CPU 上 `.half()` 为什么会报错？

`.half()` 把模型权重从 float32 转为 float16。GPU 对 float16 有硬件级支持（Tensor Core），但 CPU 的矩阵乘法 (`addmm`) 不支持 float16。错误信息：`"addmm_impl_cpu_" not implemented for 'Half'`。

解决方案：在 CPU 上使用 `.float()`（float32）。

### Q4：`[-request.max_tokens:]` 截断是什么意思？

```python
new_prompt = tokenizer.apply_chat_template(...)[-request.max_tokens:]
```

这是对**字符串**（不是 token）做切片，取最后 `max_tokens` 个**字符**。这是 MiniMind 的一个粗糙实现——它用 max_tokens 同时控制：
1. 提示词截断（取最后 N 个字符）
2. 生成长度限制（最多生成 N 个 token）

这导致一个两难困境：
- max_tokens 太小 → 提示词被截掉开头，丢失系统提示和工具定义
- max_tokens 太大 → CPU 推理尝试生成太多 token，耗时过长

我们在 MinCode 中使用 `max_tokens=4096` 作为折中。

### Q5：返回的 JSON 必须哪些字段？

OpenAI SDK 最少需要：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ]
}
```

可选但建议包含的字段：`id`, `object`, `created`, `model`, `finish_reason`, `tool_calls`。

### Q6：如何支持 tool_calls 返回？

在返回 JSON 的 `message` 中添加 `tool_calls` 字段即可。SDK 会自动将其解析为 `ChatCompletionMessageToolCall` 对象：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": {
          "name": "list_files",
          "arguments": "{\"path\": \".\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

注意 `arguments` 必须是**字符串**（JSON 序列化后的），不是 dict。

### Q7：为什么不直接在 Python 里加载模型调用，非要走 HTTP？

走 HTTP 有两个好处：
1. **解耦**：模型服务和 agent harness 是独立进程，可以分别重启、升级
2. **协议标准化**：用 OpenAI 兼容 API 意味着你的 harness 可以无缝切换到 GPT-4、Claude 或其他任何兼容的模型

但如果你只在本地用，直接调用也完全可以。MinCode 的 `ChatModel` Protocol 就是为此设计的——你可以写一个 `LocalMiniMindClient` 直接加载模型，不走 HTTP，只要实现 `complete()` 方法即可。

---

## 附录：技术栈总结图

```
┌─────────────────────────────────────────────────────────────────┐
│                        MinCode Harness                          │
│                                                                 │
│  agent.py ──► model_adapter.py ──► OpenAI SDK                   │
│                                       │                         │
│                                       │  HTTP POST              │
│                                       │  Content-Type: json     │
│                                       ▼                         │
├─────────────────────────── 网络边界 ─────────────────────────────┤
│                                       │                         │
│                    uvicorn (HTTP 服务器)                         │
│                         │                                       │
│                    FastAPI (路由框架)                             │
│                         │                                       │
│              Pydantic (JSON → ChatRequest)                       │
│                         │                                       │
│            apply_chat_template (消息 → 文本)                     │
│                         │                                       │
│              tokenizer (文本 → token IDs)                        │
│                         │                                       │
│            model.generate (token IDs → token IDs)                │
│                         │                                       │
│              tokenizer.decode (token IDs → 文本)                 │
│                         │                                       │
│            parse_response (文本 → 结构化数据)                     │
│                         │                                       │
│              FastAPI (dict → JSON 响应)                          │
│                         │                                       │
│                    uvicorn (HTTP 响应)                           │
│                                                                 │
│                   MiniMind API Server                            │
└─────────────────────────────────────────────────────────────────┘
```
