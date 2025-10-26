# Function Calling 完整流程详解

## Q: 是大模型返回的 function calling 的 json 吗？整个流程是什么？

### 核心概念澄清

**是的，Function Calling 的 JSON 是由大模型返回的**。但这里有个重要的理解：

- **大模型不会真正执行函数**
- **大模型只是决定要调用什么函数，以及传递什么参数**
- **真正的函数执行是由我们的代码来完成的**

### 完整流程图解

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   1. 发送请求    │    │   2. 模型决策    │    │   3. 解析执行    │    │   4. 继续对话    │
│                │    │                │    │                │    │                │
│ 用户: "分析数据"  │───▶│ 模型: "需要执行   │───▶│ 代码: 真正执行    │───▶│ 模型: "根据结果   │
│ + 工具定义      │    │ execute_code"   │    │ Python代码      │    │ 继续分析"       │
│                │    │ + 参数JSON      │    │                │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 详细流程步骤

#### 步骤 1: 构造请求 (我们的代码 → LLM)

```python
# 我们发送给 LLM 的请求
request = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "system", 
            "content": "你是一个数学建模专家"
        },
        {
            "role": "user", 
            "content": "请分析这个数据集"
        }
    ],
    "tools": [  # 关键：我们告诉模型有哪些工具可用
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "执行Python代码",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string", 
                            "description": "要执行的Python代码"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ],
    "tool_choice": "auto"  # 让模型自主决定是否使用工具
}
```

#### 步骤 2: 模型决策 (LLM → 我们的代码)

**模型返回的 JSON 响应**：
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,  // 注意：当有工具调用时，content 通常为 null
                "tool_calls": [  // 关键：这是模型的决策
                    {
                        "id": "call_xyz789",
                        "type": "function", 
                        "function": {
                            "name": "execute_code",  // 模型决定调用这个函数
                            "arguments": "{\"code\": \"import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df.head())\"}"  // 模型生成的参数
                        }
                    }
                ]
            }
        }
    ]
}
```

**重要理解**：
- 模型**只是生成了一个"调用指令"**
- 模型**没有真正执行任何代码**
- 模型**根据上下文智能决定**要调用什么函数以及传递什么参数

#### 步骤 3: 我们解析并执行 (我们的代码)

```python
# MathModelAgent 中的处理逻辑
async def handle_llm_response(self, response):
    # 1. 检查是否有工具调用
    if (hasattr(response.choices[0].message, "tool_calls") 
        and response.choices[0].message.tool_calls):
        
        # 2. 获取工具调用信息
        tool_call = response.choices[0].message.tool_calls[0]
        tool_id = tool_call.id  # "call_xyz789"
        function_name = tool_call.function.name  # "execute_code"
        
        # 3. 解析参数 (这是模型生成的JSON字符串)
        arguments_json = tool_call.function.arguments  
        # '{"code": "import pandas as pd\\ndf = pd.read_csv(\'data.csv\')\\nprint(df.head())"}'
        
        arguments = json.loads(arguments_json)
        code = arguments["code"]
        
        # 4. 我们的代码真正执行函数
        if function_name == "execute_code":
            # 这里才是真正的执行！
            result, error_occurred, error_msg = await self.code_interpreter.execute_code(code)
            
            # 5. 构造工具响应消息
            tool_response = {
                "role": "tool",
                "tool_call_id": tool_id,  # 必须匹配
                "name": function_name,
                "content": result if not error_occurred else error_msg
            }
            
            return tool_response
```

#### 步骤 4: 继续对话 (我们的代码 → LLM)

```python
# 将工具执行结果加入对话历史
self.chat_history.extend([
    # 添加模型的工具调用消息
    {
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {
                "id": "call_xyz789",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": '{"code": "import pandas as pd\\ndf = pd.read_csv(\'data.csv\')\\nprint(df.head())"}'
                }
            }
        ]
    },
    # 添加工具执行结果
    {
        "role": "tool", 
        "tool_call_id": "call_xyz789",
        "name": "execute_code",
        "content": "   name  age  salary\n0  Alice   25   50000\n1    Bob   30   60000\n..."
    }
])

# 再次调用模型，让它基于结果继续对话
next_response = await self.model.chat(history=self.chat_history)
```

**模型基于结果的回复**：
```json
{
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "数据已成功加载！我看到这是一个包含员工信息的数据集，有姓名、年龄和薪资三个字段。接下来我来进行详细分析..."
            }
        }
    ]
}
```

### 关键技术细节

#### 1. 工具调用的消息配对

**每个工具调用都必须有对应的响应**：
```python
# 工具调用消息 (来自模型)
{
    "role": "assistant",
    "tool_calls": [{"id": "call_123", "function": {...}}]
}

# 工具响应消息 (我们构造)  
{
    "role": "tool",
    "tool_call_id": "call_123",  # 必须匹配上面的 id
    "name": "execute_code",
    "content": "执行结果"
}
```

#### 2. 参数解析的错误处理

```python
try:
    # 模型生成的参数可能有格式错误
    arguments = json.loads(tool_call.function.arguments)
    code = arguments["code"]
except json.JSONDecodeError as e:
    # 处理模型生成的无效 JSON
    logger.error(f"解析工具参数失败: {e}")
    # 可以要求模型重新生成
except KeyError as e:
    # 处理缺少必要参数
    logger.error(f"缺少必要参数: {e}")
```

#### 3. 多轮工具调用

```python
# 模型可能连续调用多个工具
while True:
    response = await self.model.chat(history=self.chat_history, tools=tools)
    
    if response.choices[0].message.tool_calls:
        # 执行工具并添加结果到历史
        tool_result = await self.execute_tool(response.choices[0].message.tool_calls[0])
        self.chat_history.append(response.choices[0].message.model_dump())
        self.chat_history.append(tool_result)
        continue  # 继续下一轮
    else:
        # 没有工具调用，对话结束
        return response.choices[0].message.content
```

### 具体代码示例 (MathModelAgent)

```python
# app/core/agents/coder_agent.py
class CoderAgent(Agent):
    async def run(self, prompt: str) -> CoderToWriter:
        while True:
            # 1. 发送请求给模型 (包含工具定义)
            response = await self.model.chat(
                history=self.chat_history,
                tools=coder_tools,  # 工具定义
                tool_choice="auto"
            )
            
            # 2. 检查模型是否决定使用工具
            if (hasattr(response.choices[0].message, "tool_calls") 
                and response.choices[0].message.tool_calls):
                
                tool_call = response.choices[0].message.tool_calls[0]
                tool_id = tool_call.id
                
                # 3. 根据模型的决策执行相应工具
                if tool_call.function.name == "execute_code":
                    # 解析模型生成的参数
                    code = json.loads(tool_call.function.arguments)["code"]
                    
                    # 添加模型的工具调用到历史
                    await self.append_chat_history(
                        response.choices[0].message.model_dump()
                    )
                    
                    # 4. 真正执行代码 (这里才是实际执行)
                    text_to_gpt, error_occurred, error_message = await self.code_interpreter.execute_code(code)
                    
                    # 5. 构造工具响应并添加到历史
                    tool_response = {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": "execute_code", 
                        "content": text_to_gpt if not error_occurred else error_message
                    }
                    await self.append_chat_history(tool_response)
                    
                    if error_occurred:
                        # 如果出错，让模型反思并重试
                        retry_count += 1
                        reflection_prompt = get_reflection_prompt(error_message, code)
                        await self.append_chat_history({
                            "role": "user",
                            "content": reflection_prompt
                        })
                        continue
                    else:
                        # 成功执行，继续下一轮对话
                        continue
            else:
                # 6. 模型认为任务完成，没有更多工具调用
                return CoderToWriter(
                    coder_response=response.choices[0].message.content,
                    created_images=await self.code_interpreter.get_created_images()
                )
```

### 常见误解澄清

#### ❌ 误解 1: "模型会直接执行代码"
**正确理解**: 模型只会生成"执行代码的指令"，真正的执行由我们的 `code_interpreter` 完成。

#### ❌ 误解 2: "Function Calling 是一个特殊的模型能力"  
**正确理解**: Function Calling 只是一种特殊的文本生成格式，模型学会了生成符合工具调用规范的 JSON。

#### ❌ 误解 3: "工具调用是异步的"
**正确理解**: 工具调用在对话层面是同步的，模型等待工具执行结果后再继续。

#### ❌ 误解 4: "所有模型都支持 Function Calling"
**正确理解**: 只有专门训练过的模型才支持，如 GPT-4、Claude-3.5、Gemini 等。

### 调试技巧

#### 1. 查看模型的原始响应
```python
logger.info(f"模型原始响应: {response}")
logger.info(f"工具调用: {response.choices[0].message.tool_calls}")
```

#### 2. 验证参数解析
```python
try:
    arguments = json.loads(tool_call.function.arguments)
    logger.info(f"解析后的参数: {arguments}")
except Exception as e:
    logger.error(f"参数解析失败: {e}")
    logger.error(f"原始参数字符串: {tool_call.function.arguments}")
```

#### 3. 检查消息历史
```python
logger.info(f"当前对话历史长度: {len(self.chat_history)}")
for i, msg in enumerate(self.chat_history[-5:]):  # 最后5条消息
    logger.info(f"消息 {i}: {msg['role']} - {msg.get('content', 'tool_calls')[:100]}")
```

### 总结

Function Calling 的本质是：

1. **我们定义**可用的工具和参数格式
2. **模型决策**是否需要使用工具以及如何使用
3. **模型生成**符合格式的 JSON 指令
4. **我们解析**这些指令并真正执行函数
5. **我们构造**工具执行结果的消息
6. **模型基于**执行结果继续对话

整个过程中，**模型负责决策和理解，我们负责执行和控制**。这种分工让 AI 系统既具备了智能决策能力，又保持了确定性的执行控制。