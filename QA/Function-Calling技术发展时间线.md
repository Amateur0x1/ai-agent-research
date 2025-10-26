# Function Calling 技术发展时间线

## Q: 什么时候开始有了 Function Calling 这个规则？各大模型什么时候开始支持的？

根据网络搜索的结果，我为你整理了 Function Calling 技术的完整发展时间线：

### 🚀 OpenAI - 领先者 (2023年6月)

#### **2023年6月13日** - 划时代的开始
- **OpenAI 发布 GPT-4-0613 和 GPT-3.5-turbo-0613**
- **首次引入 Function Calling 能力**
- 这是业界**第一个**支持 Function Calling 的大型语言模型 API

**关键特性**：
```python
# 2023年6月13日首次支持的格式
{
    "model": "gpt-4-0613",
    "messages": [...],
    "functions": [  # 注意：最初是 functions 参数
        {
            "name": "execute_code",
            "description": "执行Python代码",
            "parameters": {...}
        }
    ],
    "function_call": "auto"  # 最初是 function_call
}
```

#### **2023年12月** - API 标准化
- **废弃 `functions` 和 `function_call` 参数**
- **引入标准化的 `tools` 和 `tool_choice` 参数**
- 这个变更奠定了现代 Function Calling 的标准格式

```python
# 新的标准格式 (2023年12月后)
{
    "model": "gpt-4",
    "messages": [...],
    "tools": [  # 新的标准参数
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "parameters": {...}
            }
        }
    ],
    "tool_choice": "auto"  # 新的标准参数
}
```

#### **2024年6月** - Structured Outputs
- **推出 Structured Outputs 功能**
- 进一步增强了 Function Calling 的可靠性和结构化输出能力

### 🎯 Anthropic Claude - 快速跟进 (2024年)

#### **2024年3月** - Claude 3 发布
- **Claude 3 家族发布**（Haiku、Sonnet、Opus）
- **计划支持 Tool Use（Function Calling）**
- 开始内测和开发者预览

#### **2024年5月31日** - 正式发布
- **Claude Tool Use 正式 GA（General Availability）**
- 支持通过 Anthropic API、Amazon Bedrock、Google Cloud Vertex AI 使用
- 比 OpenAI 晚了将近一年

**Claude 的格式**：
```python
# Claude 的 tools 格式
{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [...],
    "tools": [
        {
            "name": "execute_code",  # 无需 type 和 function 包装
            "description": "执行Python代码",
            "input_schema": {...}  # 使用 input_schema 而非 parameters
        }
    ]
}
```

### 🌟 Google Gemini - 渐进发展 (2024-2025年)

#### **2023年12月** - Gemini 1.0 发布
- **Gemini 1.0 发布**
- 初期版本可能包含基础的 Function Calling 支持

#### **2024年** - 功能完善
- **Gemini 1.5 系列发布**
- Function Calling 功能逐步完善和稳定

#### **2024年12月** - Gemini 2.0 时代
- **Gemini 2.0 Flash Experimental 发布**
- **"为智能体时代而生"**
- 内置 Google Search、代码执行、Function Calling 等工具
- 支持多轮推理和高级智能体能力

#### **2025年** - 当前状态
- **Gemini 2.5 系列**
- 在代码生成和 Function Calling 方面持续改进
- 支持免费层级（每天1500次请求）

**Gemini 的格式**：
```python
# Gemini 的 tools 格式
{
    "model": "gemini-2.0-flash-exp",
    "contents": [...],  # 使用 contents 而非 messages
    "tools": [
        {
            "function_declarations": [  # 使用 function_declarations
                {
                    "name": "execute_code",
                    "description": "执行Python代码",
                    "parameters": {...}
                }
            ]
        }
    ]
}
```

### 📊 发展时间线总览

```
2023年6月    2023年12月    2024年3月    2024年5月    2024年12月    2025年
    |           |           |           |           |           |
OpenAI首发    API标准化    Claude 3    Claude GA   Gemini 2.0  Gemini 2.5
Function     tools参数    发布计划    正式发布    智能体时代   持续改进
Calling      标准化                             
```

### 🔧 技术标准演进

#### 第一代 (2023年6月-12月)
- **OpenAI 专有格式**
- `functions` 和 `function_call` 参数
- 单一厂商支持

#### 第二代 (2023年12月-2024年)
- **标准化格式确立**
- `tools` 和 `tool_choice` 参数
- 多厂商开始采用类似格式

#### 第三代 (2024年-现在)
- **格式趋同但有差异**
- 各厂商保持兼容性的同时优化自己的实现
- LiteLLM 等统一接口工具出现

### 🌍 行业影响

#### **为什么这个功能如此重要？**

1. **突破了 LLM 的边界**
   - 从纯文本生成到能够调用外部工具
   - 使 AI 从"聊天机器人"进化为"智能助手"

2. **催生了 AI Agent 生态**
   - 使复杂的多步骤任务自动化成为可能
   - 推动了 Agent 框架的发展（LangChain、AutoGen 等）

3. **改变了应用开发模式**
   - 开发者可以轻松让 AI 调用任何 API
   - 降低了 AI 应用的开发门槛

### 🔍 技术实现对比

| 厂商 | 首次支持时间 | 参数名称 | 格式特点 | 优势 |
|------|-------------|----------|----------|------|
| **OpenAI** | 2023年6月 | `tools` | 标准 OpenAPI 格式 | 🥇 最早支持，生态最完善 |
| **Anthropic** | 2024年5月 | `tools` | 简化的 schema 格式 | 🎯 格式简洁，易于使用 |
| **Google** | 2024年 | `tools` | function_declarations 格式 | 🚀 内置工具丰富 |

### 💡 为什么你之前不知道？

这个功能相对较新，而且：

1. **技术门槛**：需要理解 JSON Schema 和 API 调用
2. **文档分散**：各厂商文档位置不同
3. **快速变化**：API 格式在快速演进中
4. **框架屏蔽**：很多人通过 LangChain 等框架使用，不直接接触原始 API

### 🛠️ MathModelAgent 的选择

MathModelAgent 选择**直接使用原生 API**而不是框架，原因：

1. **时间优势**：2024年开发时，Function Calling 已经成熟
2. **控制权**：完全控制工具调用逻辑
3. **性能**：避免框架开销
4. **标准化**：基于已经稳定的 API 标准

### 🔮 未来趋势

1. **格式进一步标准化**：可能出现统一的工业标准
2. **功能更加强大**：支持更复杂的工具调用模式
3. **性能持续优化**：更快的响应速度和更高的准确性
4. **生态更加丰富**：更多预构建的工具和集成

---

**总结**：Function Calling 是一个相对新的技术，从 2023年6月 OpenAI 首次引入到现在，不到两年时间。各大厂商快速跟进，现在已经成为现代 LLM 的标准能力。MathModelAgent 正是基于这个成熟的技术栈构建的，这就是为什么它能够如此优雅地实现多智能体工具调用！