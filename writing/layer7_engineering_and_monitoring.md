# 第七层：工程化与容错监控层 - 技术实现详解

## 概述

工程化与容错监控层是确保 AI Agent 系统在生产环境中稳定运行的最后一道防线。该层关注三个核心能力：

1. **智能容错** - 错误检测、智能反思、自动重试与降级
2. **可观测性** - 完整的追踪、监控与性能分析
3. **故障恢复** - 检查点回退、时间旅行与状态重放

本文通过 **MathModelAgent**（智能重试）、**ADK**（OpenTelemetry 追踪）、**OpenAI Agents SDK**（自动追踪）、**LangGraph**（时间旅行）的真实代码，展示生产级 Agent 系统的工程化实践。

## 核心职责

1. **错误处理与重试** - 智能识别错误并自动修复
2. **执行追踪** - 记录完整的执行轨迹
3. **性能监控** - Token 使用、延迟、成本分析
4. **故障恢复** - 从任意检查点恢复执行
5. **可观测性** - 集成外部监控平台

---

## 一、智能容错：错误反思与自动重试

### 1.1 MathModelAgent：CoderAgent 的智能重试机制

CoderAgent 实现了最有代表性的智能容错模式：**错误检测 → 反思分析 → 重新生成代码**。

```python
# MathModelAgent/backend/app/core/agents/coder_agent.py
class CoderAgent(Agent):
    def __init__(
        self,
        task_id: str,
        model: LLM,
        work_dir: str,
        max_chat_turns: int = settings.MAX_CHAT_TURNS,
        max_retries: int = settings.MAX_RETRIES,  # 最大反思次数
        code_interpreter: BaseCodeInterpreter = None,
    ):
        super().__init__(task_id, model, max_chat_turns)
        self.work_dir = work_dir
        self.max_retries = max_retries
        self.code_interpreter = code_interpreter
    
    async def run(self, prompt: str, subtask_title: str) -> CoderToWriter:
        retry_count = 0
        last_error_message = None
        
        while retry_count < self.max_retries:
            # 调用 LLM 生成代码
            response = await self.model.chat(
                history=self.chat_history,
                tools=coder_tools,
                tool_choice="auto"
            )
            
            # 检测工具调用
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call.id
                    code = json.loads(tool_call.function.arguments)["code"]
                    
                    # 执行代码
                    (
                        text_to_gpt,
                        error_occurred,
                        error_message,
                    ) = await self.code_interpreter.execute_code(code)
                    
                    if error_occurred:
                        # 记录错误工具响应
                        await self.append_chat_history({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": "execute_code",
                            "content": error_message,
                        })
                        
                        # 增加重试计数
                        logger.warning(f"代码执行错误: {error_message}")
                        retry_count += 1
                        logger.info(f"当前尝试次数: {retry_count} / {self.max_retries}")
                        
                        # === 核心：生成反思提示 ===
                        reflection_prompt = get_reflection_prompt(error_message, code)
                        
                        await redis_manager.publish_message(
                            self.task_id,
                            SystemMessage(content="代码手反思纠正错误", type="error"),
                        )
                        
                        # 将反思提示添加到对话历史
                        await self.append_chat_history({
                            "role": "user",
                            "content": reflection_prompt
                        })
                        
                        # 继续下一次循环，让 LLM 重新生成代码
                        continue
                    else:
                        # 成功执行
                        await self.append_chat_history({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": "execute_code",
                            "content": text_to_gpt,
                        })
                        return CoderToWriter(code_response=text_to_gpt, ...)
```

**设计亮点：**
- **智能反思**：不是简单重试，而是通过 LLM 分析错误原因
- **上下文累积**：错误信息保留在对话历史中，帮助 LLM 学习
- **分层重试**：Agent 级别的重试次数独立于 LLM 调用的重试

### 1.2 反思提示的设计

```python
# MathModelAgent/backend/app/core/prompts.py
def get_reflection_prompt(error_message, code) -> str:
    return f"""The code execution encountered an error:
{error_message}

Please analyze the error, identify the cause, and provide a corrected version of the code. 
Consider:
1. Syntax errors
2. Missing imports
3. Incorrect variable names or types
4. File path issues
5. Any other potential issues
6. If a task repeatedly fails to complete, try breaking down the code, changing your approach, or simplifying the model.
7. Don't ask user any thing about how to do and next to do, just do it by yourself.

Previous code:
{code}

Please provide an explanation of what went wrong and Remember call the function tools to retry
"""
```

**关键要素：**
- **错误上下文**：包含完整的错误消息和失败代码
- **引导思考**：提示 LLM 从多个角度分析问题
- **行动指令**：明确要求生成修正代码并重新调用工具

### 1.3 LLM 层的指数退避重试

```python
# MathModelAgent/backend/app/core/llm/llm.py
class LLM:
    async def chat(
        self,
        history: list = None,
        tools: list = None,
        max_retries: int = 8,      # 最大重试次数
        retry_delay: float = 1.0,  # 初始重试延迟
        **kwargs
    ) -> str:
        # 验证工具调用完整性
        if history:
            history = self._validate_and_fix_tool_calls(history)
        
        # 指数退避重试
        for attempt in range(max_retries):
            try:
                response = await acompletion(**kwargs)
                
                if not response or not hasattr(response, "choices"):
                    raise ValueError("无效的API响应")
                
                self.chat_count += 1
                await self.send_message(response, agent_name, sub_title)
                return response
                
            except Exception as e:
                logger.error(f"第{attempt + 1}次重试: {str(e)}")
                
                if attempt < max_retries - 1:
                    # 指数退避：延迟时间随重试次数增加
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                # 所有重试都失败，抛出异常
                logger.debug(f"请求参数: {kwargs}")
                raise
```

**设计优势：**
- **双层防护**：Agent 级重试（智能）+ LLM 调用重试（网络）
- **指数退避**：避免对 API 造成压力
- **完整性验证**：自动检测并修复工具调用序列

### 1.4 JSON 解析的容错重试

```python
# MathModelAgent/backend/app/core/agents/coordinator_agent.py
class CoordinatorAgent(Agent):
    async def run(self, ques_all: str) -> CoordinatorToModeler:
        max_retries = 3
        attempt = 0
        
        while attempt <= max_retries:
            try:
                response = await self.model.chat(
                    history=self.chat_history,
                    agent_name=self.__class__.__name__,
                )
                
                json_str = response.choices[0].message.content
                
                # 清理 JSON 字符串
                json_str = json_str.replace("```json", "").replace("```", "").strip()
                json_str = re.sub(r"[\x00-\x1F\x7F]", "", json_str)
                
                if not json_str:
                    raise ValueError("返回的 JSON 字符串为空")
                
                questions = json.loads(json_str)
                ques_count = questions["ques_count"]
                
                logger.info(f"questions:{questions}")
                return CoordinatorToModeler(questions=questions, ques_count=ques_count)
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                attempt += 1
                logger.warning(f"解析失败 (尝试 {attempt}/{max_retries}): {str(e)}")
                
                if attempt > max_retries:
                    logger.error(f"超过最大重试次数，放弃解析")
                    raise RuntimeError(f"无法解析模型响应: {str(e)}")
                
                # 添加错误反馈提示
                error_prompt = f"⚠️ 上次响应格式错误: {str(e)}。请严格输出JSON格式"
                await self.append_chat_history({
                    "role": "system",
                    "content": self.system_prompt + "\n" + error_prompt
                })
        
        raise RuntimeError("意外的流程终止")
```

**关键策略：**
- **主动清理**：去除 Markdown 代码块标记
- **即时反馈**：将解析错误反馈给 LLM
- **明确边界**：超过重试次数后明确失败

---

## 二、可观测性：OpenTelemetry 追踪（ADK）

### 2.1 ADK 的自动追踪架构

ADK 内置了完整的 OpenTelemetry 追踪，自动记录智能体执行、工具调用、LLM 请求的详细信息。

```python
# adk-python/src/google/adk/agents/base_agent.py
class BaseAgent(BaseModel):
    async def run_async(
        self,
        parent_context: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """智能体运行入口，自动包装追踪"""
        
        async def _run_with_trace() -> AsyncGenerator[Event, None]:
            # 创建追踪 Span
            with tracer.start_as_current_span(f'invoke_agent {self.name}') as span:
                ctx = self._create_invocation_context(parent_context)
                
                # 记录智能体调用信息
                tracing.trace_agent_invocation(span, self, ctx)
                
                # 前置回调
                if event := await self.__handle_before_agent_callback(ctx):
                    yield event
                
                if ctx.end_invocation:
                    return
                
                # 执行智能体逻辑
                async with Aclosing(self._run_async_impl(ctx)) as agen:
                    async for event in agen:
                        yield event
                
                if ctx.end_invocation:
                    return
                
                # 后置回调
                if event := await self.__handle_after_agent_callback(ctx):
                    yield event
        
        async with Aclosing(_run_with_trace()) as agen:
            async for event in agen:
                yield event
```

### 2.2 智能体调用追踪

```python
# adk-python/src/google/adk/telemetry/tracing.py
def trace_agent_invocation(
    span: trace.Span, agent: BaseAgent, ctx: InvocationContext
) -> None:
    """记录智能体调用的详细信息"""
    
    # 必需属性
    span.set_attribute('gen_ai.operation.name', 'invoke_agent')
    
    # 智能体信息
    span.set_attribute('gen_ai.agent.description', agent.description)
    span.set_attribute('gen_ai.agent.name', agent.name)
    
    # 会话信息
    span.set_attribute('gen_ai.conversation.id', ctx.session.id)
```

### 2.3 工具调用追踪

```python
# adk-python/src/google/adk/telemetry/tracing.py
def trace_tool_call(
    tool: BaseTool,
    args: dict[str, Any],
    function_response_event: Event,
):
    """记录工具调用的详细信息"""
    span = trace.get_current_span()
    
    # 工具元信息
    span.set_attribute('gen_ai.operation.name', 'execute_tool')
    span.set_attribute('gen_ai.tool.description', tool.description)
    span.set_attribute('gen_ai.tool.name', tool.name)
    span.set_attribute('gen_ai.tool.type', tool.__class__.__name__)
    
    # 工具调用参数
    span.set_attribute(
        'gcp.vertex.agent.tool_call_args',
        _safe_json_serialize(args),
    )
    
    # 工具响应
    tool_call_id = '<not specified>'
    tool_response = '<not specified>'
    
    if function_response_event.content and function_response_event.content.parts:
        response_parts = function_response_event.content.parts
        function_response = response_parts[0].function_response
        
        if function_response:
            if function_response.id:
                tool_call_id = function_response.id
            if function_response.response:
                tool_response = function_response.response
    
    span.set_attribute('gen_ai.tool.call_id', tool_call_id)
    
    if not isinstance(tool_response, dict):
        tool_response = {'result': tool_response}
    
    span.set_attribute(
        'gcp.vertex.agent.tool_response',
        _safe_json_serialize(tool_response),
    )
```

### 2.4 LLM 调用追踪

```python
# adk-python/src/google/adk/telemetry/tracing.py
def trace_call_llm(
    invocation_context: InvocationContext,
    event_id: str,
    llm_request: LlmRequest,
    llm_response: LlmResponse,
):
    """记录 LLM 调用的完整信息"""
    span = trace.get_current_span()
    
    # 标准 GenAI 属性
    span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
    span.set_attribute('gen_ai.request.model', llm_request.model)
    
    # 上下文信息
    span.set_attribute('gcp.vertex.agent.invocation_id', invocation_context.invocation_id)
    span.set_attribute('gcp.vertex.agent.session_id', invocation_context.session.id)
    span.set_attribute('gcp.vertex.agent.event_id', event_id)
    
    # 请求参数
    span.set_attribute(
        'gcp.vertex.agent.llm_request',
        _safe_json_serialize(_build_llm_request_for_trace(llm_request)),
    )
    
    # 配置参数
    if llm_request.config:
        if llm_request.config.top_p:
            span.set_attribute('gen_ai.request.top_p', llm_request.config.top_p)
        if llm_request.config.max_output_tokens:
            span.set_attribute('gen_ai.request.max_tokens', llm_request.config.max_output_tokens)
    
    # 响应信息
    try:
        llm_response_json = llm_response.model_dump_json(exclude_none=True)
    except Exception:
        llm_response_json = '<not serializable>'
    
    span.set_attribute('gcp.vertex.agent.llm_response', llm_response_json)
    
    # Token 使用统计
    if llm_response.usage_metadata:
        span.set_attribute('gen_ai.usage.input_tokens', llm_response.usage_metadata.prompt_tokens)
        span.set_attribute('gen_ai.usage.output_tokens', llm_response.usage_metadata.candidates_tokens)
```

**追踪体系优势：**
- **完整覆盖**：Agent、Tool、LLM 三层全覆盖
- **标准化**：遵循 OpenTelemetry GenAI 语义约定
- **可扩展**：支持自定义导出器（Logfire、AgentOps 等）
- **零侵入**：自动追踪，无需手动埋点

---

## 三、OpenAI Agents SDK：自动追踪与 Span 管理

### 3.1 默认追踪机制

OpenAI Agents SDK 内置了全面的追踪系统，默认开启：

```python
# openai-agents-python-main/docs/tracing.md（摘要）
# 默认追踪的内容：
# 1. 整个 Runner.{run, run_sync, run_streamed}() 被包装在 trace() 中
# 2. 每次 Agent 执行被包装在 agent_span() 中
# 3. LLM 生成被包装在 generation_span() 中
# 4. 函数工具调用被包装在 function_span() 中
# 5. Guardrails 被包装在 guardrail_span() 中
# 6. Handoffs 被包装在 handoff_span() 中
# 7. 音频转录被包装在 transcription_span() 中
# 8. TTS 被包装在 speech_span() 中
```

### 3.2 Trace 与 Span 的层次结构

```python
# openai-agents-python-main/src/agents/tracing/traces.py
class Trace(abc.ABC):
    """代表一个完整的端到端工作流
    
    包含相关的 Span 和元数据
    """
    
    # 工作流名称
    workflow_name: str
    
    # 唯一追踪 ID（格式：trace_<32_alphanumeric>）
    trace_id: str
    
    # 可选分组 ID（如聊天线程 ID）
    group_id: str | None
    
    # 元数据
    metadata: dict[str, Any] | None

# Span 表示有开始和结束时间的操作
# - started_at 和 ended_at 时间戳
# - trace_id：所属的 trace
# - parent_id：父 Span（如果有）
# - span_data：Span 的详细信息
```

### 3.3 自定义追踪

```python
# 基本用法
with trace("Order Processing") as t:
    validation_result = await Runner.run(validator, order_data)
    if validation_result.approved:
        await Runner.run(processor, order_data)

# 带元数据和分组
with trace(
    "Customer Service",
    group_id="chat_123",
    metadata={"customer": "user_456"}
) as t:
    result = await Runner.run(support_agent, query)
```

### 3.4 追踪配置

```python
# 全局禁用追踪
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# 单次运行禁用追踪
RunConfig(tracing_disabled=True)
```

**追踪特性：**
- **零配置**：默认自动追踪，无需额外设置
- **层次化**：Trace → Span 的清晰层次
- **云集成**：直接推送到 OpenAI Traces 仪表板
- **灵活控制**：支持全局或单次禁用

---

## 四、LangGraph：时间旅行与检查点回退

### 4.1 时间旅行概念

LangGraph 的时间旅行功能允许从任意检查点恢复执行，支持三种用例：

1. **理解推理**：分析成功结果的决策步骤
2. **调试错误**：定位错误发生的位置和原因
3. **探索替代方案**：修改状态后重新执行

```python
# langgraph/docs/docs/concepts/time-travel.md（摘要）
# 时间旅行流程：
# 1. 运行图并保存检查点
# 2. 使用 get_state_history() 获取历史检查点
# 3. （可选）使用 update_state() 修改状态
# 4. 使用 checkpoint_id 从特定检查点恢复执行
```

### 4.2 获取检查点历史

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# 配置检查点
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 首次运行
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"question": "..."}, config)

# 获取执行历史
history = list(graph.get_state_history(config))

for state in history:
    print(f"Step: {state.values}")
    print(f"Next: {state.next}")
    print(f"Checkpoint ID: {state.config['configurable']['checkpoint_id']}")
    print("---")
```

### 4.3 修改状态并重新执行

```python
# 1. 获取特定检查点的状态
checkpoint_config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "某个历史检查点的 ID"
    }
}

current_state = graph.get_state(checkpoint_config)
print(f"Current state: {current_state.values}")

# 2. 修改状态
graph.update_state(
    checkpoint_config,
    {"documents": filtered_documents}  # 修改后的状态
)

# 3. 从修改后的检查点恢复执行
# 传入 None 表示从当前状态继续
result = graph.invoke(None, checkpoint_config)
```

### 4.4 容错恢复

```python
# langgraph/libs/langgraph/tests/test_pregel.py（摘要）
def test_checkpoint_recovery():
    """测试检查点容错恢复"""
    
    class State(TypedDict):
        steps: Annotated[list[str], operator.add]
        attempt: int  # 追踪尝试次数
    
    def failing_node(state: State):
        # 第一次尝试失败，重试时成功
        if state["attempt"] == 1:
            raise RuntimeError("Simulated failure")
        return {"steps": ["node1"]}
    
    builder = StateGraph(State)
    builder.add_node("node1", failing_node)
    builder.add_node("node2", second_node)
    
    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    
    # 第一次尝试失败
    with pytest.raises(RuntimeError):
        graph.invoke(
            {"steps": ["start"], "attempt": 1},
            config
        )
    
    # 验证检查点状态
    state = graph.get_state(config)
    assert state.values == {"steps": ["start"], "attempt": 1}
    assert state.next == ("node1",)  # 应该重试失败的节点
    assert "RuntimeError('Simulated failure')" in state.tasks[0].error
    
    # 重试并更新尝试计数
    result = graph.invoke({"steps": [], "attempt": 2}, config)
    assert result == {"steps": ["start", "node1", "node2"], "attempt": 2}
    
    # 验证检查点历史记录了两次尝试
    history = list(graph.get_state_history(config))
    assert len(history) >= 2
    
    # 验证错误被记录在检查点中
    failed_checkpoint = next(c for c in history if c.tasks and c.tasks[0].error)
    assert "RuntimeError('Simulated failure')" in failed_checkpoint.tasks[0].error
```

**容错特性：**
- **自动保存失败状态**：节点失败时保存检查点
- **保留成功写入**：并行节点中成功的部分不会重新执行
- **错误信息保留**：检查点包含完整的错误堆栈
- **灵活恢复**：可选择从失败点重试或修改状态后恢复

### 4.5 Pending Writes 机制

```markdown
# langgraph/docs/docs/concepts/persistence.md（摘要）

当某个节点在超级步骤中间失败时，LangGraph 会保存该超级步骤中
其他成功完成的节点的待写入（pending writes）。

这样，当我们从该超级步骤恢复执行时，就不需要重新运行那些
已经成功的节点。
```

---

## 五、工程化最佳实践

### 5.1 错误处理策略对比

| 策略 | MathModelAgent | ADK | LangGraph | OpenAI SDK |
|------|----------------|-----|-----------|------------|
| **Agent 级重试** | ✅ 智能反思 | ⚠️ 手动实现 | ✅ 检查点恢复 | ✅ 自动重试 |
| **LLM 级重试** | ✅ 指数退避 | ✅ 内置 | ✅ 内置 | ✅ 内置 |
| **工具级重试** | ✅ 代码执行 | ✅ 确认门 | ⚠️ 手动实现 | ⚠️ 手动实现 |
| **错误反思** | ✅ Prompt 引导 | ❌ | ❌ | ❌ |
| **状态恢复** | ⚠️ Redis 消息 | ⚠️ Session | ✅ 检查点 | ⚠️ 手动实现 |

### 5.2 追踪与监控对比

| 特性 | ADK | OpenAI SDK | LangGraph | MathModelAgent |
|------|-----|------------|-----------|----------------|
| **自动追踪** | ✅ OTel | ✅ 内置 | ⚠️ 需配置 | ❌ |
| **标准协议** | ✅ OTel | ✅ OpenAI | ⚠️ 自定义 | ❌ |
| **云集成** | ✅ Vertex AI | ✅ OpenAI | ✅ LangSmith | ❌ |
| **自定义导出** | ✅ Exporter | ✅ Processor | ✅ Callbacks | ⚠️ 手动 |
| **可视化** | ✅ GCP Console | ✅ Traces UI | ✅ Studio | ⚠️ 自建 |

### 5.3 容错恢复能力对比

| 能力 | LangGraph | ADK | OpenAI SDK | MathModelAgent |
|------|-----------|-----|------------|----------------|
| **检查点** | ✅ 自动 | ✅ Session | ⚠️ 手动 | ⚠️ Redis |
| **时间旅行** | ✅ 内置 | ❌ | ❌ | ❌ |
| **状态修改** | ✅ update_state | ⚠️ 手动 | ⚠️ 手动 | ❌ |
| **并行容错** | ✅ Pending Writes | ❌ | ❌ | ❌ |
| **错误诊断** | ✅ 任务错误 | ✅ Span 错误 | ✅ Span 错误 | ⚠️ 日志 |

### 5.4 组合策略建议

**1. 小型项目/快速原型**
```python
# 使用 OpenAI Agents SDK 的默认追踪
from agents import Agent, Runner
from agents.tracing import trace

with trace("My Workflow"):
    result = Runner.run(agent, input)
```

**2. 企业级应用**
```python
# ADK + OpenTelemetry + 自定义导出器
from google.adk.agents import LlmAgent
from google.adk.telemetry import setup_otel

# 配置 OTel 导出到 Logfire/AgentOps
setup_otel(exporters=[LogfireExporter(), AgentOpsExporter()])

agent = LlmAgent(...)
result = await agent.run_async(ctx)
```

**3. 需要容错与回退**
```python
# LangGraph + 检查点 + 时间旅行
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(db_url)
graph = workflow.compile(checkpointer=checkpointer)

try:
    result = graph.invoke(input, config)
except Exception as e:
    # 获取失败前的检查点
    history = list(graph.get_state_history(config))
    last_good_checkpoint = history[1]  # 失败前的状态
    
    # 修改状态并重试
    graph.update_state(last_good_checkpoint.config, modified_state)
    result = graph.invoke(None, last_good_checkpoint.config)
```

**4. 需要智能容错**
```python
# MathModelAgent 模式 + 自定义反思
async def smart_retry_agent(prompt: str, max_retries: int = 3):
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            result = await execute_with_tools(prompt)
            return result
        except ToolExecutionError as e:
            retry_count += 1
            
            # 生成反思提示
            reflection = f"""
            The previous attempt failed with error: {e}
            
            Please analyze the error and try a different approach.
            Consider: {get_error_hints(e)}
            """
            
            # 将反思添加到历史
            prompt = reflection
```

---

## 六、监控指标与告警

### 6.1 关键指标

**执行性能：**
- 平均响应时间
- P50/P95/P99 延迟
- 超时率

**资源消耗：**
- Token 使用量（输入/输出）
- API 调用次数
- 成本统计

**可靠性：**
- 成功率
- 重试率
- 错误类型分布

**业务指标：**
- 任务完成率
- 用户满意度
- 工具调用成功率

### 6.2 告警策略

```python
# 示例：基于 OTel 的告警
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

# 创建指标
error_counter = meter.create_counter(
    "agent_errors_total",
    description="Total number of agent errors",
)

retry_counter = meter.create_counter(
    "agent_retries_total",
    description="Total number of agent retries",
)

latency_histogram = meter.create_histogram(
    "agent_latency_seconds",
    description="Agent execution latency",
)

# 记录指标
async def monitored_agent_run(agent, input):
    start_time = time.time()
    
    try:
        result = await agent.run(input)
        latency_histogram.record(time.time() - start_time)
        return result
    except Exception as e:
        error_counter.add(1, {"error_type": type(e).__name__})
        raise
```

### 6.3 日志策略

**分级日志：**
```python
# DEBUG: 详细的调试信息
logger.debug(f"LLM request: {request}")

# INFO: 关键执行步骤
logger.info(f"Agent started: {agent.name}")

# WARNING: 重试和降级
logger.warning(f"Retry {attempt}/{max_retries}: {error}")

# ERROR: 执行失败
logger.error(f"Agent failed: {error}", exc_info=True)
```

**结构化日志：**
```python
logger.info(
    "Agent execution completed",
    extra={
        "agent_name": agent.name,
        "task_id": task_id,
        "duration_ms": duration,
        "token_count": token_count,
        "retry_count": retry_count,
    }
)
```

---

## 七、总结

工程化与容错监控层是生产级 Agent 系统的**质量保障**：

**核心能力矩阵：**

| 能力 | 实现方案 | 优先级 |
|------|---------|--------|
| **智能容错** | MathModelAgent 反思机制 | 🔴 高 |
| **自动追踪** | ADK OTel / OpenAI SDK | 🔴 高 |
| **检查点恢复** | LangGraph 时间旅行 | 🟡 中 |
| **性能监控** | OTel + Prometheus | 🔴 高 |
| **成本追踪** | Token 统计 + 告警 | 🟡 中 |
| **日志分析** | 结构化日志 + ELK | 🔴 高 |

**实施路线图：**

1. **第一阶段（MVP）**：
   - 基础日志记录
   - 简单重试机制
   - 错误捕获与上报

2. **第二阶段（生产）**：
   - OpenTelemetry 追踪
   - 检查点持久化
   - 性能监控仪表板

3. **第三阶段（优化）**：
   - 智能错误反思
   - 时间旅行调试
   - 自动化告警与修复

通过系统化的工程实践，可以构建**可观测、可恢复、可优化**的生产级 Agent 系统，确保在复杂场景下的稳定性和可靠性。

