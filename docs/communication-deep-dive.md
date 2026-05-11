# MinCode 与 MiniMind 模型通讯机制深度解析

本文档逐层拆解 MinCode Agent 与 MiniMind 模型之间的完整通讯链路，从用户敲下回车到模型返回结果，每一步发生了什么、数据格式如何变换、由哪行代码负责。

---

## 0. 全景图

```
┌─────────────────────── MinCode 进程 ───────────────────────┐    ┌──── MiniMind 服务器进程 ────┐
│                                                             │    │                              │
│  用户输入                                                    │    │  FastAPI (uvicorn :8998)     │
│    │                                                        │    │    │                         │
│    ▼                                                        │    │    ▼                         │
│  Agent.run()                                                │    │  chat_completions()          │
│    │                                                        │    │    │                         │
│    ├─ 组装 messages (OpenAI 格式的 dict 列表)                 │    │    ├─ apply_chat_template()  │
│    │                                                        │    │    │   → 把 messages + tools  │
│    ├─ tools.as_openai_tools() → 6 个工具的 JSON schema        │    │    │     渲染成一段纯文本      │
│    │                                                        │    │    │                         │
│    ▼                                                        │    │    ├─ tokenizer() → token ids │
│  MiniMindClient.complete()                                  │    │    │                         │
│    │                                                        │    │    ├─ model.generate()       │
│    ├─ OpenAI SDK 序列化为 HTTP JSON body                     │    │    │   → 自回归生成 token     │
│    │                                                        │    │    │                         │
│    ▼                                                        │    │    ├─ tokenizer.decode()     │
│  ┌──────────────────┐       HTTP POST                       │    │    │   → 生成的纯文本         │
│  │ openai.OpenAI()  │ ──────/v1/chat/completions ──────────►│    │    │                         │
│  │  client.chat.    │                                       │    │    ├─ parse_response()       │
│  │  completions.    │       HTTP 200 JSON                   │    │    │   → 从文本中提取:        │
│  │  create()        │ ◄────────────────────────────────────-│    │    │     content              │
│  └──────────────────┘                                       │    │    │     reasoning_content    │
│    │                                                        │    │    │     tool_calls           │
│    ├─ _message_to_dict() → 纯 dict                          │    │    │                         │
│    │                                                        │    │    └─ 返回标准 JSON          │
│    ▼                                                        │    │                              │
│  Agent 解析 tool_calls → 执行工具 → 追加结果 → 再次调用模型    │    └──────────────────────────────┘
│    │
│    ▼
│  返回最终文本给用户
└─────────────────────────────────────────────────────────────┘
```

下面逐层展开。

---

## 1. 第一层：Agent 组装请求

**代码位置：`src/mincode/agent.py` 第 67-76 行**

当用户输入 `"列出当前目录的文件"` 时，`Agent.run()` 做了两件事：

```python
# (1) 把用户输入追加到 messages 历史
self.messages.append({"role": "user", "content": user_input})

# (2) 调用模型，传入完整的 messages 历史 + 工具定义
assistant = self.model.complete(
    messages=self.messages,          # ← 所有历史消息
    tools=self.tools.as_openai_tools(),  # ← 6 个工具的 JSON schema
)
```

此时 `self.messages` 的内容（首轮对话）：

```python
[
    {"role": "system", "content": "你是MinCode编程助手。工作目录:/path/to/mincode"},
    {"role": "user", "content": "列出当前目录的文件"},
]
```

`self.tools.as_openai_tools()` 返回的内容（6 个工具，以 `list_files` 为例）：

```python
[
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录中的文件和子目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径"}
                },
                "required": []
            }
        }
    },
    # ... 另外 5 个工具: read_file, grep_files, write_file, replace_in_file, exec_command
]
```

**代码位置：`src/mincode/tools.py` 第 68-79 行** — `as_openai_tools()` 方法把内部的 `Tool` dataclass 转成 OpenAI function-calling 标准格式。

这两个 Python 对象（`messages` 和 `tools`）被传给 `self.model.complete()`，也就是 `MiniMindClient.complete()`。

---

## 2. 第二层：MiniMindClient 发送 HTTP 请求

**代码位置：`src/mincode/model_adapter.py` 第 74-113 行**

`MiniMindClient.complete()` 把 Python dict 通过 OpenAI SDK 发送为 HTTP POST 请求：

```python
def complete(self, *, messages, tools):
    kwargs = {
        "model": self.model,           # "minimind"
        "messages": messages,           # 上一步的 messages 列表
        "temperature": self.temperature, # 0.3
        "max_tokens": self.max_tokens,   # 4096
        "stream": False,                # 关键：必须显式关闭流式
    }
    if tools:
        kwargs["tools"] = tools         # 工具定义列表
        kwargs["tool_choice"] = "auto"  # 让模型自己决定是否调工具

    response = self._client.chat.completions.create(**kwargs)
    #          ↑ 这一行实际发出了 HTTP 请求
```

### 2.1 OpenAI SDK 做了什么

`self._client` 是在 `__post_init__` 中创建的 `openai.OpenAI` 实例：

```python
# model_adapter.py 第 62-72 行
def __post_init__(self):
    from openai import OpenAI
    self._client = OpenAI(
        api_key=self.api_key,      # "minimind"（占位符，MiniMind不校验）
        base_url=self.base_url,    # "http://localhost:8998/v1"
        timeout=self.timeout,      # 120 秒
    )
```

当调用 `self._client.chat.completions.create(**kwargs)` 时，OpenAI SDK 内部：

1. 将 `kwargs` 序列化为 JSON
2. 发送 `POST http://localhost:8998/v1/chat/completions`
3. 请求头：`Content-Type: application/json`
4. 等待响应，反序列化 JSON 为 SDK 的 `ChatCompletion` 对象

实际发出的 HTTP 请求体：

```json
{
    "model": "minimind",
    "messages": [
        {"role": "system", "content": "你是MinCode编程助手。工作目录:/path/to/mincode"},
        {"role": "user", "content": "列出当前目录的文件"}
    ],
    "tools": [
        {"type": "function", "function": {"name": "list_files", ...}},
        {"type": "function", "function": {"name": "read_file", ...}},
        ...
    ],
    "tool_choice": "auto",
    "temperature": 0.3,
    "max_tokens": 4096,
    "stream": false
}
```

---

## 3. 第三层：MiniMind 服务器接收请求

**代码位置：`minimind/scripts/serve_openai_api.py` 第 171-227 行**

MiniMind 这边是一个 FastAPI 应用，接收到请求后进入 `chat_completions()` 函数。

### 3.1 请求解析

FastAPI + Pydantic 自动将 JSON body 解析为 `ChatRequest` 对象：

```python
# serve_openai_api.py 第 50-59 行
class ChatRequest(BaseModel):
    model: str
    messages: list            # ← 原样接收 messages
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192    # ← 默认 8192，我们传了 4096
    stream: bool = True       # ← 默认 True！我们传了 False
    tools: list = []          # ← 原样接收工具定义
    open_thinking: bool = False
```

因为我们传了 `stream: false`，所以走非流式分支（第 186 行的 `else`）。

### 3.2 Chat Template 渲染

这是通讯机制中最关键的格式变换。MiniMind 的 tokenizer 内嵌了一段 Jinja2 模板，负责将结构化的 `messages` + `tools` 渲染成模型能理解的纯文本。

```python
# serve_openai_api.py 第 187-193 行
new_prompt = tokenizer.apply_chat_template(
    request.messages,               # 我们传入的 messages 列表
    tokenize=False,                 # 返回字符串而非 token ids
    add_generation_prompt=True,     # 在末尾添加 "<|im_start|>assistant\n"
    tools=request.tools or None,    # 我们传入的工具定义
    open_thinking=...               # 是否开启思考模式
)[-request.max_tokens:]            # 按字符数截断（用 max_tokens 值）
```

这个 Jinja2 模板（定义在 `model/tokenizer_config.json` 中）做了以下事情：

**输入**：结构化的 messages 和 tools

**输出**（实际渲染结果）：

```
<|im_start|>system
你是MinCode编程助手。工作目录:/path/to/mincode

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "list_files", "description": "列出目录中的文件和子目录", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径"}}, "required": []}}}
{"type": "function", "function": {"name": "read_file", ...}}
...（其余工具）
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

模板做了这些事：
1. 如果有 `tools`，将 system message 和工具定义合并成一个大的 system block
2. 工具定义被包裹在 `<tools>...</tools>` XML 标签中，每个工具一行 JSON
3. 在工具定义之后，插入了调用格式说明："For each function call, return a json object... within `<tool_call></tool_call>` XML tags"
4. 用户消息用 `<|im_start|>user\n...<|im_end|>` 包裹
5. 末尾添加 `<|im_start|>assistant\n<think>\n\n</think>\n\n` 作为生成起点（空的思考块 + 等待模型续写）

这段纯文本就是模型实际看到的输入。模型在训练时（SFT 阶段）见过大量这种格式的数据，所以它"知道"看到 `<tools>` 就意味着可以输出 `<tool_call>` 标签。

### 3.3 分词 (Tokenization)

```python
# serve_openai_api.py 第 194 行
inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
```

将上面的纯文本字符串转成 token id 张量。MiniMind 使用 BPE + ByteLevel tokenizer，vocab=6400，其中有专门的特殊 token：

| Token | ID | 用途 |
|-------|-----|------|
| `<\|im_start\|>` | 1 | 消息块开始 |
| `<\|im_end\|>` | 2 | 消息块结束 |
| `<tool_call>` | 21 | 工具调用开始 |
| `</tool_call>` | 22 | 工具调用结束 |
| `<tool_response>` | 23 | 工具响应开始 |
| `</tool_response>` | 24 | 工具响应结束 |
| `<think>` | 25 | 思考内容开始 |
| `</think>` | 26 | 思考内容结束 |

这些特殊 token 是在训练 tokenizer 时就预留好的（ID 0-35 全部是特殊 token），不会被 BPE 拆分，模型可以一个 token 就输出一个完整的 `<tool_call>` 标记。

### 3.4 模型推理 (Generation)

```python
# serve_openai_api.py 第 195-205 行
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_ids"],                                     # prompt tokens
        max_length=inputs["input_ids"].shape[1] + request.max_tokens,  # 最大长度
        do_sample=True,
        temperature=request.temperature,       # 0.3
        top_p=request.top_p,                  # 0.92
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,  # 0
        eos_token_id=tokenizer.eos_token_id,  # 2 (<|im_end|>)
    )
```

模型（MiniMindForCausalLM，64M 参数的 decoder-only transformer）从 prompt 末尾开始，一个 token 一个 token 地自回归生成，直到输出 `<|im_end|>`（EOS, token id=2）或达到 max_length。

模型输出的 token 序列解码后可能是：

```
<tool_call>
{"name": "list_files", "arguments": {"path": "."}}
</tool_call>
```

### 3.5 解码与后处理

```python
# serve_openai_api.py 第 206 行
answer = tokenizer.decode(
    generated_ids[0][inputs["input_ids"].shape[1]:],  # 只取新生成的部分
    skip_special_tokens=True                           # 去掉 <|im_end|> 等
)
```

此时 `answer` 是一个纯文本字符串，可能包含 `<think>...</think>` 和 `<tool_call>...</tool_call>` 标签。

### 3.6 响应解析 (parse_response)

**代码位置：`serve_openai_api.py` 第 83-102 行**

`parse_response()` 用正则表达式从生成的纯文本中提取结构化信息：

```python
def parse_response(text):
    # 1. 提取思考内容
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    # 2. 提取工具调用
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        call = json.loads(m.strip())
        tool_calls.append({
            "id": f"call_{int(time.time())}_{i}",  # 生成唯一 ID
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": json.dumps(call.get("arguments", {}))
            }
        })

    # 3. 从文本中去掉已提取的 <tool_call> 标签，剩下的是纯文本 content
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)

    return text.strip(), reasoning_content, tool_calls or None
```

这个函数完成了**从模型的自由文本输出到结构化 API 响应**的关键转换：

| 模型原始输出 | 解析后 |
|-------------|--------|
| `<think>让我查看文件列表</think>` | `reasoning_content = "让我查看文件列表"` |
| `<tool_call>{"name": "list_files", "arguments": {"path": "."}}</tool_call>` | `tool_calls = [{"id": "call_xxx_0", "type": "function", "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"}}]` |
| 剩余的普通文本 | `content = "..."` |

### 3.7 构造 HTTP 响应

```python
# serve_openai_api.py 第 207-225 行
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
```

返回给 MinCode 的 HTTP 响应 JSON：

```json
{
    "id": "chatcmpl-1778487296",
    "object": "chat.completion",
    "created": 1778487296,
    "model": "minimind",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1778487296_0",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": "{\"path\": \".\"}"
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }
    ]
}
```

---

## 4. 第四层：MinCode 接收响应

### 4.1 SDK 反序列化

OpenAI SDK 将上面的 JSON 反序列化为 Python 对象：

```
response                              → ChatCompletion 对象
  .choices[0]                         → Choice 对象
    .message                          → ChatCompletionMessage 对象
      .role                           → "assistant"
      .content                        → ""
      .tool_calls                     → [ChatCompletionMessageToolCall 对象]
        [0].id                        → "call_1778487296_0"
        [0].type                      → "function"
        [0].function                  → Function 对象
          .name                       → "list_files"
          .arguments                  → '{"path": "."}'
```

### 4.2 _message_to_dict() 转换

**代码位置：`src/mincode/model_adapter.py` 第 13-38 行**

SDK 的对象是带类型的 Pydantic model，但 Agent 内部统一用纯 dict。`_message_to_dict()` 做转换：

```python
def _message_to_dict(message):
    # 基础字段
    result = {"role": "assistant", "content": message.content or ""}

    # 提取 reasoning_content（MiniMind 的 <think> 标签解析结果）
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content

    # 提取 tool_calls — 从 SDK 对象转为纯 dict
    raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
    if raw_tool_calls:
        calls = []
        for item in raw_tool_calls:
            function = getattr(item, "function", None)
            calls.append({
                "id": str(getattr(item, "id", "")),         # "call_1778487296_0"
                "type": "function",
                "function": {
                    "name": str(getattr(function, "name", "")),       # "list_files"
                    "arguments": str(getattr(function, "arguments", "{}")),  # '{"path": "."}'
                },
            })
        result["tool_calls"] = calls

    return result
```

为什么用 `getattr` 而不是直接访问属性？因为 OpenAI SDK 不同版本的对象结构可能不同，`getattr` 配合默认值能确保兼容性。

转换后的纯 dict：

```python
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_1778487296_0",
            "type": "function",
            "function": {
                "name": "list_files",
                "arguments": "{\"path\": \".\"}"
            }
        }
    ]
}
```

---

## 5. 第五层：Agent 处理响应并执行工具

**代码位置：`src/mincode/agent.py` 第 84-116 行**

Agent 拿到上面的 dict 后：

```python
# 第 84 行：检查是否有 tool_calls
tool_calls = assistant.get("tool_calls") or []

# 第 86-95 行：把 assistant 消息存入历史
record = {
    "role": "assistant",
    "content": assistant.get("content"),     # ""
    "tool_calls": tool_calls,                # [{"id": ..., "function": {...}}]
}
self.messages.append(record)

# 第 98-99 行：如果没有 tool_calls，直接返回文本
if not tool_calls:
    return _coerce_text(assistant.get("content"))

# 第 101-116 行：有 tool_calls，逐个执行
for call in tool_calls:
    call_id = str(call.get("id", ""))                # "call_1778487296_0"
    function = call.get("function") or {}
    name = str(function.get("name", ""))              # "list_files"
    arguments = str(function.get("arguments", "{}"))  # '{"path": "."}'

    result = self.tools.execute(name, arguments)
    # ToolRegistry.execute() 做了：
    #   1. 按 name 查找注册的 Tool
    #   2. json.loads(arguments) 解析参数
    #   3. 如果是 mutating 工具且未 auto_approve，询问用户
    #   4. 调用 tool.handler(context, parsed_args)
    #   5. 返回结果字符串

    # 把工具执行结果追加到 messages 历史
    self.messages.append({
        "role": "tool",
        "tool_call_id": call_id,               # 对应上面的 call id
        "content": result,                      # 工具返回的字符串
    })
```

执行 `list_files` 工具后，`result` 可能是：

```
CLAUDE.md
pyproject.toml
docs/
scripts/
skills/
src/
```

---

## 6. 第六层：带工具结果的第二轮请求

Agent 在执行完所有 tool_calls 后，回到循环顶部（`agent.py` 第 71 行），再次调用 `self.model.complete()`。

此时 `self.messages` 已变为：

```python
[
    {"role": "system", "content": "你是MinCode编程助手。工作目录:..."},
    {"role": "user", "content": "列出当前目录的文件"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_xxx_0", "type": "function",
         "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_xxx_0",
     "content": "CLAUDE.md\npyproject.toml\ndocs/\nscripts/\nskills/\nsrc/"},
]
```

这一轮的通讯流程与第 2-4 层完全相同，只是 messages 更长了。

在 MiniMind 服务器端，`apply_chat_template()` 会将这些消息（含工具调用和工具响应）渲染为：

```
<|im_start|>system
你是MinCode编程助手。工作目录:...
（工具定义 <tools>...</tools>）
<|im_end|>
<|im_start|>user
列出当前目录的文件<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "list_files", "arguments": {"path": "."}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
CLAUDE.md
pyproject.toml
docs/
scripts/
skills/
src/
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

```

注意 chat template 的两个关键设计：
- **assistant 的 tool_call** 被渲染成 `<tool_call>...</tool_call>` 标签，嵌入在 assistant 消息内
- **tool 的 role** 被渲染成 `<|im_start|>user` + `<tool_response>...</tool_response>`，也就是说工具结果被当作一种特殊的 user 消息

模型看到工具响应后，这一轮通常会直接输出总结性的文本（不再调用工具），例如：

```
当前目录包含以下文件和目录：
- CLAUDE.md
- pyproject.toml
- docs/
- scripts/
- skills/
- src/
```

此时 `parse_response()` 不会提取到 `tool_calls`（因为没有 `<tool_call>` 标签），`finish_reason` 为 `"stop"`。Agent 收到后发现 `tool_calls` 为空，直接返回文本内容给用户，循环结束。

---

## 7. 数据格式变换总结

一次完整的工具调用涉及 **5 次格式变换**：

```
[Agent 内部 Python dict]
        │
        │  (1) OpenAI SDK 序列化
        ▼
[HTTP JSON body]  ──── POST /v1/chat/completions ────►  [MiniMind 服务器]
                                                              │
                                                              │  (2) apply_chat_template
                                                              ▼
                                                        [纯文本 prompt]
                                                              │
                                                              │  (3) tokenizer 编码
                                                              ▼
                                                        [token id 张量]
                                                              │
                                                              │  model.generate() → tokenizer.decode()
                                                              ▼
                                                        [模型输出纯文本，含 <tool_call> 标签]
                                                              │
                                                              │  (4) parse_response 正则提取
                                                              ▼
                                                        [结构化 JSON 响应]
        [HTTP JSON response]  ◄──────────────────────────────-┘
        │
        │  (5) _message_to_dict 转换
        ▼
[Agent 内部 Python dict]
```

| 阶段 | 格式 | 负责者 | 代码位置 |
|------|------|--------|----------|
| (1) Python dict → HTTP JSON | OpenAI SDK 内部 | `openai` 库 | SDK 内部 |
| (2) JSON messages → 纯文本 prompt | Jinja2 chat template | `tokenizer_config.json` 中的模板 | `serve_openai_api.py:187-193` |
| (3) 纯文本 → token ids → 生成 → 纯文本 | tokenizer + model | transformers | `serve_openai_api.py:194-206` |
| (4) 纯文本（含标签）→ 结构化 JSON | 正则表达式 | `parse_response()` | `serve_openai_api.py:83-102` |
| (5) SDK 对象 → Python dict | `_message_to_dict()` | model_adapter.py | `model_adapter.py:13-38` |

---

## 8. 协议兼容性解析

为什么 MinCode 能无缝对接 MiniMind？因为通讯双方在三个层面上对齐了协议：

### 8.1 HTTP API 层

MiniMind 的 `serve_openai_api.py` 实现了 OpenAI Chat Completions API 的子集：

| OpenAI API 规范 | MiniMind 实现 | 状态 |
|-----------------|--------------|------|
| `POST /v1/chat/completions` | `@app.post("/v1/chat/completions")` | 兼容 |
| `messages` 列表（system/user/assistant/tool） | `ChatRequest.messages: list` | 兼容 |
| `tools` 工具定义 | `ChatRequest.tools: list` | 兼容 |
| `tool_choice` | 忽略（由 chat template 决定） | 可用 |
| `stream: false` 返回完整 JSON | 非流式分支 | 兼容 |
| `response.choices[0].message.tool_calls` | `parse_response()` 提取后注入 | 兼容 |
| `finish_reason: "tool_calls"` / `"stop"` | 根据是否有 tool_calls 决定 | 兼容 |

### 8.2 消息格式层

Agent 内部、HTTP 传输、服务器接收全部使用 OpenAI Chat Completions 消息格式：

```python
# system message
{"role": "system", "content": "..."}

# user message
{"role": "user", "content": "..."}

# assistant message (with tool calls)
{"role": "assistant", "content": "...", "tool_calls": [
    {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
]}

# tool result message
{"role": "tool", "tool_call_id": "...", "content": "..."}
```

### 8.3 工具定义格式层

```python
# OpenAI function calling 标准格式
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "...",
        "parameters": { ... JSON Schema ... }
    }
}
```

MiniMind 的 chat template 直接将这个 JSON 原样嵌入 `<tools>` 标签中，所以任何符合 OpenAI 格式的工具定义都能被 MiniMind 接受。

---

## 9. 一次完整请求的时序图

```
时间线 ──────────────────────────────────────────────────────────────────►

用户          Agent (agent.py)           MiniMindClient          MiniMind Server
 │                │                        │                        │
 │  "列出文件"    │                        │                        │
 │ ──────────────►│                        │                        │
 │                │                        │                        │
 │                │ messages=[sys,user]     │                        │
 │                │ tools=[6个工具schema]   │                        │
 │                │ ──────────────────────► │                        │
 │                │                        │                        │
 │                │                        │  POST /v1/chat/...     │
 │                │                        │ ──────────────────────►│
 │                │                        │                        │
 │                │                        │                        │ apply_chat_template()
 │                │                        │                        │ tokenizer.encode()
 │                │                        │                        │ model.generate()
 │                │                        │                        │ tokenizer.decode()
 │                │                        │                        │ parse_response()
 │                │                        │                        │
 │                │                        │  200 OK {tool_calls}   │
 │                │                        │ ◄──────────────────────│
 │                │                        │                        │
 │                │  dict{tool_calls=[...]} │                        │
 │                │ ◄──────────────────────-│                        │
 │                │                        │                        │
 │                │ 执行 list_files 工具    │                        │
 │                │ 结果: "CLAUDE.md\n..."  │                        │
 │                │                        │                        │
 │                │ messages=[sys,user,     │                        │
 │                │   assistant+tc, tool]   │                        │
 │                │ ──────────────────────► │                        │
 │                │                        │  POST /v1/chat/...     │
 │                │                        │ ──────────────────────►│
 │                │                        │                        │
 │                │                        │                        │ （第二轮推理）
 │                │                        │                        │
 │                │                        │  200 OK {content}      │
 │                │                        │ ◄──────────────────────│
 │                │                        │                        │
 │                │  dict{content="..."}   │                        │
 │                │ ◄─────────────────────-│                        │
 │                │                        │                        │
 │  "当前目录有..."│                       │                        │
 │ ◄──────────────│                        │                        │
```

---

## 10. 关键设计取舍

### 10.1 为什么用 HTTP API 而不是直接加载模型

方案 A（当前）：Agent 进程 ←HTTP→ MiniMind API 服务器进程
方案 B（备选）：Agent 进程直接 import 并加载 PyTorch 模型

选择方案 A 的原因：
- **零代码改动**：MiniMind 自带了 `serve_openai_api.py`，直接用
- **进程隔离**：模型推理的内存和 CPU 占用不影响 Agent 的交互体验
- **协议标准**：用了 OpenAI 格式，将来可以无缝切换到任何兼容的模型服务器
- **调试方便**：可以用 curl 独立测试 API，排查问题时能隔离是 Agent 端还是模型端的问题

代价是多了一次 HTTP 序列化/反序列化和网络延迟，但对于 localhost 通讯来说可以忽略。

### 10.2 为什么 chat template 是核心

MiniMind 模型本身**不理解**结构化的 messages 和 tools。它只是一个接受 token 序列、输出 token 序列的 transformer。Chat template 是让模型"理解"对话和工具的关键桥梁：

- 没有 chat template → 模型看到的是一堆 JSON dict，完全不知道怎么回复
- 有了 chat template → 模型看到的是它训练时见过的格式化文本，能正确续写

这也解释了为什么模型能输出正确的 `<tool_call>` 格式但选错工具名——**格式是 chat template 教的，但工具知识要靠训练数据**。

### 10.3 为什么 parse_response 用正则而不是结构化解析

因为 LLM 的输出本质上是**自由文本**。模型不保证输出合法的 JSON 或完美的标签嵌套。正则表达式的容错性比严格的 XML/JSON 解析器好得多：

- `<tool_call>` 前后可能有空白或换行 → 正则可以处理
- 模型可能在 `<tool_call>` 前输出一些文本 → 正则能跳过
- JSON 可能有轻微格式问题 → 用 `try/except` 包裹 `json.loads`

---

## 附录：完整的 Token 级别示例

用 MiniMind 的 tokenizer 对第一轮请求进行 token 级别编码：

```
Token ID  →  Token 文本
────────────────────────
   1      →  <|im_start|>
  ...     →  system\n你是MinCode编程助手。...
           →  # Tools\n\nYou may call...
           →  <tools>\n{"type": "function", ...}\n</tools>
           →  \nFor each function call...
           →  <tool_call>\n{"name": ...}\n</tool_call>
   2      →  <|im_end|>
   1      →  <|im_start|>
  ...     →  user\n列出当前目录的文件
   2      →  <|im_end|>
   1      →  <|im_start|>
  ...     →  assistant\n
  25      →  <think>                ← 特殊 token，单个 ID
  ...     →  \n\n
  26      →  </think>               ← 特殊 token，单个 ID
  ...     →  \n\n
           →  （模型从这里开始生成）
```

模型生成的 token 序列（理想情况）：

```
  21      →  <tool_call>            ← 一个 token
  ...     →  \n{"name": "list_files", "arguments": {"path": "."}}
  22      →  </tool_call>           ← 一个 token
   2      →  <|im_end|>             ← EOS，生成结束
```
