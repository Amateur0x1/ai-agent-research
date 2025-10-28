# 第一层：LLM 基础设施与适配层 - 技术实现详解

## 概述

LLM 基础设施与适配层是 AI Agent 架构的基石，负责统一不同大语言模型的接口差异，确保 Agent 系统具有模型无关性。本文将通过 **MathModelAgent**、**OpenAI Agents SDK**、**ADK (Agent Development Kit)** 和 **Vercel AI SDK** 的真实代码实现，深入分析这一层的设计理念和技术实现。

## 核心职责

1. **统一接口抽象** - 屏蔽不同 LLM 提供商的 API 差异
2. **标准化协议** - 基于 OpenAI Function Calling 标准
3. **多模型管理** - 为不同 Agent 分配专门的模型
4. **模型无关性** - 支持 100+ 模型提供商的无缝切换

---

## 一、MathModelAgent：自主实现的LLM适配层

### 1.1 核心LLM封装类

MathModelAgent 采用自主实现的方式，通过 LiteLLM 提供统一接口：

```python
# MathModelAgent/backend/app/core/llm/llm.py
from litellm import acompletion
import litellm

class LLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        task_id: str,
    ):
        self.api_key = api_key
        self.model = model          # 如: gpt-4o, claude-3.5-sonnet
        self.base_url = base_url    # 自定义API端点
        self.chat_count = 0
        self.max_tokens: int | None = None
        self.task_id = task_id

    async def chat(
        self,
        history: list = None,
        tools: list = None,
        tool_choice: str = None,
        max_retries: int = 8,
        retry_delay: float = 1.0,
        top_p: float | None = None,
        agent_name: AgentType = AgentType.SYSTEM,
        sub_title: str | None = None,
    ) -> str:
        # 验证和修复工具调用完整性
        if history:
            history = self._validate_and_fix_tool_calls(history)

        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "messages": history,
            "stream": False,
            "top_p": top_p,
            "metadata": {"agent_name": agent_name},
        }

        # 支持 Function Calling
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        if self.base_url:
            kwargs["base_url"] = self.base_url

        # 启用JSON Schema验证
        litellm.enable_json_schema_validation = True

        # 指数退避重试机制
        for attempt in range(max_retries):
            try:
                response = await acompletion(**kwargs)
                # 处理响应和实时消息推送
                await self.send_message(response, agent_name, sub_title)
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
```

**设计亮点：**
- **完全控制权**: 不依赖外部框架，获得高度定制化能力
- **工具调用完整性验证**: 自动检测并修复 tool_calls 与 tool 响应的配对关系
- **智能重试机制**: 指数退避 + 错误反思
- **实时消息推送**: 集成 Redis 进行实时状态同步

### 1.2 工厂模式：多模型专业化分工

```python
# MathModelAgent/backend/app/core/llm/llm_factory.py
class LLMFactory:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def get_all_llms(self) -> tuple[LLM, LLM, LLM, LLM]:
        coordinator_llm = LLM(
            api_key=settings.COORDINATOR_API_KEY,
            model=settings.COORDINATOR_MODEL,        # 如: gpt-4o-mini
            base_url=settings.COORDINATOR_BASE_URL,
            task_id=self.task_id,
        )

        modeler_llm = LLM(
            api_key=settings.MODELER_API_KEY,
            model=settings.MODELER_MODEL,            # 如: o1-preview
            base_url=settings.MODELER_BASE_URL,
            task_id=self.task_id,
        )

        coder_llm = LLM(
            api_key=settings.CODER_API_KEY,
            model=settings.CODER_MODEL,              # 如: claude-3.5-sonnet
            base_url=settings.CODER_BASE_URL,
            task_id=self.task_id,
        )

        writer_llm = LLM(
            api_key=settings.WRITER_API_KEY,
            model=settings.WRITER_MODEL,             # 如: gemini-2.0-flash
            base_url=settings.WRITER_BASE_URL,
            task_id=self.task_id,
        )

        return coordinator_llm, modeler_llm, coder_llm, writer_llm
```

**专业化分工理念：**
- **协调者 (Coordinator)**: 使用快速、便宜的模型进行任务分解
- **建模者 (Modeler)**: 使用推理能力强的模型进行数学建模
- **编码者 (Coder)**: 使用代码能力强的模型进行编程实现
- **写作者 (Writer)**: 使用文本生成能力强的模型撰写论文

### 1.3 OpenAI Function Calling 标准实现

```python
# MathModelAgent/backend/app/core/functions.py
coder_tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "执行Python代码并返回结果",
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
]

writer_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "搜索相关学术论文",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

---

## 二、OpenAI Agents SDK：企业级LLM适配层

### 2.1 LiteLLM 模型适配器

OpenAI Agents SDK 提供了更完善的 LiteLLM 集成：

```python
# openai-agents-python-main/src/agents/extensions/models/litellm_model.py
class LitellmModel(Model):
    """支持任何通过 LiteLLM 访问的模型"""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs
    ) -> ModelResponse:
        # 转换输入格式
        converted_messages = Converter.items_to_messages(input)

        if system_instructions:
            converted_messages.insert(0, {
                "content": system_instructions,
                "role": "system",
            })

        # 转换工具定义
        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

        # 转换交接（Handoff）为工具
        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        # 调用 LiteLLM
        ret = await litellm.acompletion(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools or None,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            frequency_penalty=model_settings.frequency_penalty,
            presence_penalty=model_settings.presence_penalty,
            max_tokens=model_settings.max_tokens,
            tool_choice=self._remove_not_given(tool_choice),
            response_format=self._remove_not_given(response_format),
            parallel_tool_calls=parallel_tool_calls,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 转换响应格式
        items = Converter.message_to_output_items(
            LitellmConverter.convert_message_to_openai(ret.choices[0].message)
        )

        return ModelResponse(
            output=items,
            usage=usage,
            response_id=None,
        )
```

**设计亮点：**
- **标准化接口**: 实现了统一的 Model 接口规范
- **格式转换器**: 自动处理不同模型间的消息格式差异
- **Handoff 支持**: 内置 Agent 间的控制权转移机制
- **追踪集成**: 内置 OpenTelemetry 追踪支持

### 2.2 使用示例

```python
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

# 使用 Claude
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=LitellmModel(
        model="anthropic/claude-3-5-sonnet-20240620",
        api_key="your-anthropic-key"
    ),
    tools=[get_weather],
)

result = await Runner.run(agent, "What's the weather in Tokyo?")
```

---

## 三、ADK (Agent Development Kit)：Google 的多模型方案

### 3.1 LiteLLM 集成架构

ADK 提供了最复杂的 LiteLLM 集成，支持多种内容类型：

```python
# adk-python/src/google/adk/models/lite_llm.py
class LiteLlm(BaseLlm):
    """Google ADK 的 LiteLLM 包装器"""

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        # 警告：使用 Gemini 通过 LiteLLM 的性能损失
        _warn_gemini_via_litellm(model)
        self._additional_args = kwargs

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # 转换 ADK 格式到 LiteLLM 格式
        messages, tools, response_format, generation_params = (
            _get_completion_inputs(llm_request)
        )

        completion_args = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
        }
        completion_args.update(self._additional_args)

        if generation_params:
            completion_args.update(generation_params)

        if stream:
            # 流式处理
            async for part in await self.llm_client.acompletion(**completion_args):
                for chunk, finish_reason in _model_response_to_chunk(part):
                    if isinstance(chunk, FunctionChunk):
                        # 处理函数调用块
                        yield _message_to_generate_content_response(...)
                    elif isinstance(chunk, TextChunk):
                        # 处理文本块
                        yield _message_to_generate_content_response(...)
        else:
            response = await self.llm_client.acompletion(**completion_args)
            yield _model_response_to_generate_content_response(response)
```

### 3.2 多模态内容支持

```python
def _get_content(parts: Iterable[types.Part]) -> Union[OpenAIMessageContent, str]:
    """转换多模态内容"""
    content_objects = []
    for part in parts:
        if part.text:
            if len(parts) == 1:
                return part.text
            content_objects.append({"type": "text", "text": part.text})
        elif part.inline_data and part.inline_data.data:
            base64_string = base64.b64encode(part.inline_data.data).decode("utf-8")
            data_uri = f"data:{part.inline_data.mime_type};base64,{base64_string}"

            if part.inline_data.mime_type.startswith("image"):
                content_objects.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri, "format": part.inline_data.mime_type},
                })
            elif part.inline_data.mime_type.startswith("video"):
                content_objects.append({
                    "type": "video_url",
                    "video_url": {"url": data_uri, "format": part.inline_data.mime_type},
                })
            # ... 支持更多媒体类型
    return content_objects
```

**设计亮点：**
- **多模态支持**: 原生支持图片、视频、音频、PDF 等多种内容类型
- **性能警告机制**: 智能检测并警告 Gemini 通过 LiteLLM 的性能损失
- **流式处理**: 复杂的流式响应聚合和函数调用处理

---

## 四、Vercel AI SDK：Provider 抽象模式

### 4.1 Provider 接口设计

Vercel AI SDK 采用了最优雅的 Provider 抽象模式：

```typescript
// ai/packages/openai/src/openai-provider.ts
export interface OpenAIProvider extends ProviderV3 {
  (modelId: OpenAIResponsesModelId): LanguageModelV3;

  // 不同 API 的专门方法
  chat(modelId: OpenAIChatModelId): LanguageModelV3;
  responses(modelId: OpenAIResponsesModelId): LanguageModelV3;
  completion(modelId: OpenAICompletionModelId): LanguageModelV3;

  // 多模态支持
  embedding(modelId: OpenAIEmbeddingModelId): EmbeddingModelV3<string>;
  image(modelId: OpenAIImageModelId): ImageModelV3;
  speech(modelId: OpenAISpeechModelId): SpeechModelV2;
  transcription(modelId: OpenAITranscriptionModelId): TranscriptionModelV2;
}

export function createOpenAI(
  options: OpenAIProviderSettings = {},
): OpenAIProvider {
  const baseURL = withoutTrailingSlash(options.baseURL) ?? 'https://api.openai.com/v1';
  const providerName = options.name ?? 'openai';

  // 工厂方法模式
  const createChatModel = (modelId: OpenAIChatModelId) =>
    new OpenAIChatLanguageModel(modelId, {
      provider: `${providerName}.chat`,
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createResponsesModel = (modelId: OpenAIResponsesModelId) => {
    return new OpenAIResponsesLanguageModel(modelId, {
      provider: `${providerName}.responses`,
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      fetch: options.fetch,
      fileIdPrefixes: ['file-'],
    });
  };

  // 主要的 provider 函数
  const provider = function (modelId: OpenAIResponsesModelId) {
    return createResponsesModel(modelId);
  };

  // 附加专门的创建方法
  provider.chat = createChatModel;
  provider.responses = createResponsesModel;
  provider.completion = createCompletionModel;
  provider.embedding = createEmbeddingModel;
  // ...

  return provider as OpenAIProvider;
}

// 默认实例
export const openai = createOpenAI();
```

### 4.2 使用示例

```typescript
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';

// 方式一：使用默认 Responses API
const result1 = await generateText({
  model: openai('gpt-4o'),
  prompt: 'Explain quantum computing',
});

// 方式二：明确指定 Chat API
const result2 = await generateText({
  model: openai.chat('gpt-4o'),
  prompt: 'Explain quantum computing',
});

// 方式三：多模态使用
const result3 = await generateText({
  model: openai('gpt-4o'),
  messages: [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'What is in this image?' },
        { type: 'image', image: imageUrl },
      ],
    },
  ],
});
```

**设计亮点：**
- **函数重载**: 同一个 provider 函数支持多种调用方式
- **API 专门化**: 为不同的 OpenAI API（Chat, Responses, Completion）提供专门的方法
- **TypeScript 类型安全**: 完善的类型定义确保编译时安全
- **多模态原生支持**: 统一的接口处理文本、图片、音频等

---

## 五、架构模式对比分析

### 5.1 设计模式对比

| 框架 | 架构模式 | 优势 | 适用场景 |
|------|----------|------|----------|
| **MathModelAgent** | 自主实现 + 工厂模式 | 完全控制权、高度定制化 | 专业化应用、深度优化 |
| **OpenAI Agents SDK** | 适配器模式 + 转换器 | 标准化、企业级功能 | 企业应用、多团队协作 |
| **ADK** | 包装器模式 + 多模态 | Google 生态集成、性能优化 | Google 云环境、多模态应用 |
| **Vercel AI SDK** | Provider 抽象 + 工厂方法 | 类型安全、开发体验优秀 | 全栈应用、快速开发 |

### 5.2 LiteLLM 集成方式对比

```python
# MathModelAgent：直接调用
response = await litellm.achat(
    model=self.model,
    messages=history,
    tools=tools,
    api_key=self.api_key,
    base_url=self.base_url
)

# OpenAI Agents SDK：包装调用
ret = await litellm.acompletion(
    model=self.model,
    messages=converted_messages,
    tools=converted_tools,
    # ... 更多参数
)

# ADK：高级包装
response = await self.llm_client.acompletion(
    model=self.model,
    messages=messages,
    tools=tools,
    response_format=response_format,
    **completion_args
)
```

### 5.3 Function Calling 标准化程度

1. **MathModelAgent**: 100% 遵循 OpenAI 标准，但自主处理
2. **OpenAI Agents SDK**: 提供转换器，支持多种格式
3. **ADK**: 从 Google 格式转换为 OpenAI 格式
4. **Vercel AI SDK**: TypeScript 类型化的工具定义

---

## 六、最佳实践与设计原则

### 6.1 统一接口设计原则

1. **模型无关性**: 通过抽象层屏蔽具体模型实现差异
2. **标准协议**: 采用 OpenAI Function Calling 作为业界标准
3. **错误处理**: 智能重试 + 降级机制
4. **可观测性**: 内置日志、追踪、监控支持

### 6.2 多模型管理策略

```python
# 专业化分工示例
AGENT_MODEL_MAPPING = {
    "coordinator": "gpt-4o-mini",          # 快速、便宜
    "modeler": "o1-preview",               # 推理能力强
    "coder": "claude-3.5-sonnet",         # 代码能力强
    "writer": "gemini-2.0-flash",         # 文本生成能力强
}
```

### 6.3 性能优化建议

1. **连接池管理**: 复用 HTTP 连接
2. **并发控制**: 避免过多并发请求导致限流
3. **缓存策略**: 相同输入的响应缓存
4. **流式处理**: 长响应的实时输出

---

## 总结

LLM 基础设施与适配层是 AI Agent 架构的核心基础，不同框架采用了不同的设计模式：

- **MathModelAgent** 通过自主实现获得最大控制权和定制化能力
- **OpenAI Agents SDK** 提供企业级的标准化解决方案
- **ADK** 专注于 Google 生态的深度集成和多模态支持
- **Vercel AI SDK** 追求最佳的开发者体验和类型安全

无论采用哪种方案，关键是要实现：**统一接口、标准协议、模型无关性**，为上层的 Agent 系统提供稳定可靠的 LLM 服务。这一层的设计质量直接决定了整个 Agent 系统的可扩展性和可维护性。
