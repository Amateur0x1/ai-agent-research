# 第二层：工具系统与执行层 - 技术实现详解

## 概述

工具系统与执行层是 AI Agent 能力延伸的核心，负责将大语言模型的"决策能力"转化为"实际行动"。本层的关键理念是：**LLM 负责决策，系统负责执行**。本文将通过 **MathModelAgent**、**OpenAI Agents SDK**、**ADK** 和 **Vercel AI SDK** 的真实代码实现，深入分析工具调用的本质、执行器设计模式和安全机制。

## 核心职责

1. **Function Calling 协议** - 基于 OpenAI 标准的工具调用机制
2. **工具执行器设计** - 安全、可靠的工具执行架构
3. **代码执行隔离** - 本地 Jupyter 与云端沙盒的双模式实现
4. **人在回路 (HITL)** - 高风险操作的人工确认机制
5. **错误处理与重试** - 智能反思与自动修复

---

## 一、Function Calling 的本质与架构

### 1.1 核心理念：分离决策与执行

Function Calling 的本质是**决策与执行的分离**：

```python
# 工具调用架构流程
LLM 模型                Agent                    工具执行器
    ↓                    ↓                         ↓
发起工具调用  →  解析工具调用参数  →  执行具体工具逻辑
    ↓                    ↓                         ↓
tool_calls        tool_call.function     实际执行结果
    ↓                    ↓                         ↓
返回结果      ←  构造tool响应消息   ←    返回执行结果
```

**关键洞察：**
- **LLM 不执行函数** - 模型只负责决策和理解
- **系统执行函数** - 真正的执行由我们的代码完成
- **标准协议** - 基于 OpenAI Function Calling 标准确保兼容性

### 1.2 MathModelAgent：自主实现的工具调用系统

#### 工具定义 (Function Schema)

```python
# MathModelAgent/backend/app/core/functions.py

# CoderAgent 的代码执行工具
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

# WriterAgent 的文献搜索工具
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

#### 工具调用检测与解析

```python
# MathModelAgent/backend/app/core/agents/coder_agent.py
class CoderAgent(Agent):
    async def run(self, prompt: str) -> CoderToWriter:
        while True:
            # 1. 调用 LLM，传入工具定义
            response = await self.model.chat(
                history=self.chat_history,
                tools=coder_tools,           # 传入工具定义
                tool_choice="auto"           # 让模型自主决定是否使用工具
            )

            # 2. 检测是否有工具调用
            if (hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls):

                # 3. 解析工具调用
                tool_call = response.choices[0].message.tool_calls[0]
                tool_id = tool_call.id

                if tool_call.function.name == "execute_code":
                    # 4. 解析参数
                    code = json.loads(tool_call.function.arguments)["code"]

                    # 5. 添加助手响应到历史
                    await self.append_chat_history(
                        response.choices[0].message.model_dump()
                    )

                    # 6. 执行工具
                    result = await self.code_interpreter.execute_code(code)

                    # 7. 构造工具响应消息
                    await self.append_chat_history({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": "execute_code",
                        "content": result
                    })

                    continue  # 继续对话循环
            else:
                # 没有工具调用，任务完成
                return CoderToWriter(response.choices[0].message.content)
```

**设计亮点：**
- **完全自主控制**: 不依赖外部工具框架，获得高度定制化能力
- **异步执行**: 支持长时间运行的工具调用
- **错误智能重试**: 基于错误信息的智能反思机制
- **实时状态反馈**: 通过 Redis 向前端推送执行状态

---

## 二、工具执行器设计模式

### 2.1 抽象基类设计

#### MathModelAgent 的执行器抽象

```python
# MathModelAgent/backend/app/tools/base_interpreter.py
class BaseCodeInterpreter(abc.ABC):
    def __init__(
        self,
        task_id: str,
        work_dir: str,
        notebook_serializer: NotebookSerializer,
    ):
        self.task_id = task_id
        self.work_dir = work_dir
        self.notebook_serializer = notebook_serializer
        self.section_output: dict[str, dict[str, list[str]]] = {}
        self.last_created_images = set()

    @abc.abstractmethod
    async def initialize(self):
        """初始化解释器，必要时上传文件、启动内核等"""
        ...

    @abc.abstractmethod
    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        """执行一段代码，返回 (输出文本, 是否出错, 错误信息)"""
        ...

    @abc.abstractmethod
    async def cleanup(self):
        """清理资源，比如关闭沙箱或内核"""
        ...

    @abc.abstractmethod
    async def get_created_images(self, section: str) -> list[str]:
        """获取当前 section 创建的图片列表"""
        ...

    async def _push_to_websocket(self, content_to_display: list[OutputItem] | None):
        """实时推送执行结果到前端"""
        agent_msg = InterpreterMessage(output=content_to_display)
        await redis_manager.publish_message(self.task_id, agent_msg)
```

#### ADK 的执行器抽象

```python
# adk-python/src/google/adk/code_executors/base_code_executor.py
class BaseCodeExecutor(BaseModel):
    """Google ADK 的代码执行器基类"""

    optimize_data_file: bool = False
    """是否自动处理数据文件"""

    stateful: bool = False
    """是否保持状态"""

    error_retry_attempts: int = 2
    """连续执行错误的重试次数"""

    code_block_delimiters: List[tuple[str, str]] = [
        ('```tool_code\n', '\n```'),
        ('```python\n', '\n```'),
    ]
    """代码块定界符"""

    execution_result_delimiters: tuple[str, str] = ('```tool_output\n', '\n```')
    """执行结果定界符"""

    @abc.abstractmethod
    def execute_code(
        self,
        invocation_context: InvocationContext,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult:
        """执行代码并返回结果"""
        pass
```

### 2.2 双模式代码执行器实现

#### 本地执行器：基于 Jupyter Kernel

```python
# MathModelAgent/backend/app/tools/local_interpreter.py
class LocalCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, task_id: str, work_dir: str, notebook_serializer: NotebookSerializer):
        super().__init__(task_id, work_dir, notebook_serializer)
        self.km, self.kc = None, None
        self.interrupt_signal = False

    async def initialize(self):
        """初始化本地 Jupyter 内核"""
        logger.info("初始化本地内核")
        self.km, self.kc = jupyter_client.manager.start_new_kernel(
            kernel_name="python3"
        )
        self._pre_execute_code()

    def _pre_execute_code(self):
        """执行初始化代码，设置工作目录"""
        init_code = (
            f"import os\n"
            f"work_dir = r'{self.work_dir}'\n"
            f"os.makedirs(work_dir, exist_ok=True)\n"
            f"os.chdir(work_dir)\n"
            f"print('当前工作目录:', os.getcwd())\n"
        )
        self.execute_code_(init_code)

    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        """执行代码并返回结果"""
        logger.info(f"执行代码: {code}")

        # 添加代码到 notebook
        self.notebook_serializer.add_code_cell_to_notebook(code)

        text_to_gpt: list[str] = []
        content_to_display: list[OutputItem] | None = []
        error_occurred: bool = False
        error_message: str = ""

        # 推送执行状态
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content="开始执行代码"),
        )

        # 执行 Python 代码
        execution = self.execute_code_(code)

        # 处理执行结果
        for mark, out_str in execution:
            if mark == "stdout":
                text_to_gpt.append(f"输出: {out_str}")
                content_to_display.append(StdOutModel(content=out_str))
            elif mark == "stderr":
                text_to_gpt.append(f"错误: {out_str}")
                content_to_display.append(StdErrModel(content=out_str))
                error_occurred = True
                error_message = out_str
            elif mark == "result":
                text_to_gpt.append(f"结果: {out_str}")
                content_to_display.append(ResultModel(content=out_str))

        # 推送结果到前端
        await self._push_to_websocket(content_to_display)

        return "\n".join(text_to_gpt), error_occurred, error_message
```

**本地执行器优势：**
- ✅ **完全控制权**: 无需依赖外部服务
- ✅ **性能优异**: 无网络延迟
- ✅ **成本为零**: 无额外费用
- ✅ **持久化**: 代码保存为 .ipynb 文件

#### 云端执行器：基于 E2B 沙盒

```python
# MathModelAgent/backend/app/tools/e2b_interpreter.py
class E2BCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, task_id: str, work_dir: str, notebook_serializer: NotebookSerializer):
        super().__init__(task_id, work_dir, notebook_serializer)
        self.sbx = None

    @classmethod
    async def create(cls, task_id: str, work_dir: str, notebook_serializer: NotebookSerializer):
        """异步工厂方法创建实例"""
        instance = cls(task_id, work_dir, notebook_serializer)
        return instance

    async def initialize(self, timeout: int = 3000):
        """异步初始化沙箱环境"""
        try:
            self.sbx = await AsyncSandbox.create(
                api_key=settings.E2B_API_KEY,
                timeout=timeout
            )
            logger.info("沙箱环境初始化成功")
            await self._pre_execute_code()
            await self._upload_all_files()
        except Exception as e:
            logger.error(f"初始化沙箱环境失败: {str(e)}")
            raise

    async def _upload_all_files(self):
        """上传工作目录中的所有文件到沙箱"""
        files = [f for f in os.listdir(self.work_dir) if f.endswith((".csv", ".xlsx"))]

        for file in files:
            file_path = os.path.join(self.work_dir, file)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    content = f.read()
                    await self.sbx.files.write(f"/home/user/{file}", content)
                    logger.info(f"成功上传文件到沙箱: {file}")

    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        """在云端沙盒中执行代码"""
        logger.info(f"在沙盒中执行代码: {code}")

        try:
            # 1. 在沙盒中执行代码
            execution = await self.sbx.run_code(code, language="python")

            text_to_gpt: list[str] = []
            content_to_display: list[OutputItem] = []
            error_occurred = False
            error_message = ""

            # 2. 处理执行结果
            if execution.error:
                error_occurred = True
                error_message = execution.error.message
                text_to_gpt.append(f"错误: {error_message}")
                content_to_display.append(ErrorModel(content=error_message))
            else:
                # 处理正常输出
                for result in execution.results:
                    if result.text:
                        text_to_gpt.append(f"输出: {result.text}")
                        content_to_display.append(StdOutModel(content=result.text))

                    if result.png:
                        # 处理图片输出
                        image_filename = await self._save_image_from_base64(
                            result.png, section="execution"
                        )
                        text_to_gpt.append(f"生成图片: {image_filename}")
                        content_to_display.append(
                            ResultModel(content=f"![生成图片]({image_filename})")
                        )

            # 3. 推送结果
            await self._push_to_websocket(content_to_display)

            return "\n".join(text_to_gpt), error_occurred, error_message

        except Exception as e:
            logger.error(f"沙盒执行代码失败: {str(e)}")
            return "", True, str(e)
```

**云端执行器优势：**
- ✅ **环境隔离**: 完全隔离的执行环境
- ✅ **无需配置**: 无需本地环境配置
- ✅ **多语言支持**: 支持多种编程语言
- ✅ **可扩展性**: 支持大规模并发执行

### 2.3 工厂模式：动态选择执行器

```python
# MathModelAgent/backend/app/tools/create_interpreter.py
async def create_interpreter(
    kind: Literal["remote", "local"] = "local",
    task_id: str,
    work_dir: str,
    notebook_serializer: NotebookSerializer,
    timeout: int = 3000,
) -> BaseCodeInterpreter:
    """工厂方法：根据配置动态创建代码执行器"""

    # 如果没有 E2B API Key，强制使用本地模式
    if not settings.E2B_API_KEY:
        kind = "local"
        logger.warning("未配置 E2B_API_KEY，使用本地代码执行器")

    if kind == "remote":
        logger.info("创建云端代码执行器 (E2B)")
        interpreter = await E2BCodeInterpreter.create(
            task_id=task_id,
            work_dir=work_dir,
            notebook_serializer=notebook_serializer
        )
        await interpreter.initialize(timeout=timeout)
        return interpreter
    else:
        logger.info("创建本地代码执行器 (Jupyter)")
        interpreter = LocalCodeInterpreter(
            task_id=task_id,
            work_dir=work_dir,
            notebook_serializer=notebook_serializer
        )
        await interpreter.initialize()
        return interpreter
```

---

## 三、OpenAI Agents SDK：企业级工具系统

### 3.1 装饰器模式的工具定义

```python
# openai-agents-python-main/src/agents/tool.py
@dataclass
class FunctionTool:
    """函数工具的核心类"""

    name: str
    """工具名称，向 LLM 显示"""

    description: str
    """工具描述，向 LLM 显示"""

    params_json_schema: dict[str, Any]
    """参数的 JSON Schema"""

    on_invoke_tool: Callable[[ToolContext[Any], str], Awaitable[Any]]
    """工具调用的执行函数"""

    strict_json_schema: bool = True
    """是否使用严格的 JSON Schema"""

    is_enabled: bool | Callable = True
    """工具是否启用"""

    tool_input_guardrails: list[ToolInputGuardrail[Any]] | None = None
    """输入护栏"""

    tool_output_guardrails: list[ToolOutputGuardrail[Any]] | None = None
    """输出护栏"""

# 装饰器实现
def function_tool(
    func: ToolFunction[...] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    strict_json_schema: bool = True,
    **kwargs
) -> FunctionTool | Callable[[ToolFunction[...]], FunctionTool]:
    """
    将 Python 函数转换为 FunctionTool 的装饰器

    支持两种使用方式：
    1. @function_tool (无括号)
    2. @function_tool(...) (有参数)
    """

    if func is not None:
        # 无括号使用：@function_tool
        return _create_function_tool(func)
    else:
        # 有参数使用：@function_tool(...)
        def decorator(real_func: ToolFunction[...]) -> FunctionTool:
            return _create_function_tool(real_func)
        return decorator

def _create_function_tool(the_func: ToolFunction[...]) -> FunctionTool:
    """创建 FunctionTool 的核心逻辑"""

    # 1. 解析函数签名
    signature = inspect.signature(the_func)

    # 2. 生成 JSON Schema
    schema = function_schema(
        the_func,
        strict=strict_json_schema,
        docstring_style=DocstringStyle.AUTO
    )

    # 3. 创建调用包装器
    async def on_invoke_tool(context: ToolContext[Any], args_json: str) -> Any:
        try:
            args = json.loads(args_json)
            return await _invoke_function_with_context(the_func, context, args)
        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            raise

    return FunctionTool(
        name=name_override or the_func.__name__,
        description=description_override or _extract_description(the_func),
        params_json_schema=schema,
        on_invoke_tool=on_invoke_tool,
        strict_json_schema=strict_json_schema,
        **kwargs
    )
```

### 3.2 使用示例：优雅的工具定义

```python
# openai-agents-python-main/examples/tools/function_tool_example.py
from agents import Agent, Runner, function_tool

@function_tool
async def get_weather(location: str) -> str:
    """获取指定地点的天气信息

    Args:
        location: 地点名称，如"北京"或"New York"

    Returns:
        天气描述字符串
    """
    # 模拟天气 API 调用
    return f"{location}今天天气晴朗，温度 25°C"

@function_tool(
    name_override="search_documents",
    description_override="在文档库中搜索相关信息"
)
async def document_search(query: str, max_results: int = 5) -> list[str]:
    """搜索文档"""
    # 模拟文档搜索
    return [f"文档 {i}: 关于 {query} 的内容" for i in range(max_results)]

# 创建 Agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant with access to tools.",
    model="gpt-4o",
    tools=[get_weather, document_search]
)

# 运行对话
result = await Runner.run(agent, "北京今天天气怎么样？")
```

### 3.3 内置代码执行器

```python
# openai-agents-python-main/examples/tools/code_interpreter.py
from agents import Agent, CodeInterpreterTool, Runner

agent = Agent(
    name="Code interpreter",
    model="gpt-4.1",
    instructions="You love doing math.",
    tools=[
        CodeInterpreterTool(
            tool_config={
                "type": "code_interpreter",
                "container": {"type": "auto"}
            },
        )
    ],
)

# 流式执行
result = Runner.run_streamed(agent, "What is the square root of 273 * 312821 plus 1782?")
async for event in result.stream_events():
    if (event.type == "run_item_stream_event"
        and event.item.type == "tool_call_item"
        and event.item.raw_item.type == "code_interpreter_call"):

        print(f"Code interpreter code:\n```\n{event.item.raw_item.code}\n```\n")
```

**设计亮点：**
- ✅ **类型安全**: TypeScript 风格的类型注解支持
- ✅ **自动 Schema 生成**: 从函数签名自动生成 JSON Schema
- ✅ **护栏机制**: 内置输入输出验证
- ✅ **上下文注入**: 自动注入运行上下文

---

## 四、人在回路 (HITL) 确认机制

### 4.1 ADK 的工具确认流程

ADK 提供了生产级的 HITL 确认机制，用于高风险工具操作：

```python
# adk-python/tests/unittests/runners/test_run_tool_confirmation.py

# 配置需要确认的工具
agent = Agent(
    model=Gemini(model="gemini-2.5-pro"),
    instructions="You are a helpful assistant.",
    tools=[
        ToolConfig(
            function_declarations=[dangerous_tool],
            tool_confirmation_config=ToolConfirmationConfig(
                confirm_by_default=True,  # 默认需要确认
                require_human_approval=True  # 需要人工审批
            )
        )
    ]
)

# HITL 确认流程示例
async def handle_tool_confirmation(agent_response):
    """处理工具确认请求"""

    if agent_response.needs_confirmation():
        # 1. 显示确认界面
        confirmation_request = agent_response.get_confirmation_request()
        print(f"工具调用需要确认: {confirmation_request.tool_name}")
        print(f"参数: {confirmation_request.parameters}")
        print(f"风险等级: {confirmation_request.risk_level}")

        # 2. 等待用户确认
        user_decision = await get_user_confirmation()

        # 3. 提交确认结果
        if user_decision.approved:
            await agent_response.approve_tool_call(
                user_decision.modification  # 可选的参数修改
            )
        else:
            await agent_response.reject_tool_call(
                reason=user_decision.rejection_reason
            )
```

### 4.2 Vercel AI SDK 的确认机制

```typescript
// ai/content/cookbook/01-next/75-human-in-the-loop.mdx

import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

async function runWithHITL() {
  // 1. 初始调用
  const result = await generateText({
    model: openai('gpt-4o'),
    messages: [{ role: 'user', content: 'Delete all files in /home/user' }],
    tools: {
      deleteFiles: {
        description: 'Delete files from filesystem',
        parameters: z.object({
          path: z.string().describe('Directory path'),
        }),
      },
    },
  });

  // 2. 检查是否有工具调用
  if (result.toolCalls && result.toolCalls.length > 0) {
    const toolCall = result.toolCalls[0];

    // 3. 风险评估
    if (isHighRiskOperation(toolCall)) {
      // 4. 请求人工确认
      const userConfirmation = await requestUserConfirmation({
        operation: toolCall.toolName,
        parameters: toolCall.args,
        riskLevel: 'HIGH'
      });

      if (!userConfirmation.approved) {
        return { error: 'Operation cancelled by user' };
      }
    }

    // 5. 执行确认后的工具调用
    const toolResult = await executeToolCall(toolCall);

    // 6. 继续对话
    const finalResult = await generateText({
      model: openai('gpt-4o'),
      messages: [
        ...result.messages,
        {
          role: 'assistant',
          content: result.text,
          tool_calls: result.toolCalls,
        },
        {
          role: 'tool',
          tool_call_id: toolCall.toolCallId,
          content: toolResult,
        },
      ],
    });

    return finalResult;
  }
}

function isHighRiskOperation(toolCall: any): boolean {
  const highRiskPatterns = [
    /delete.*files?/i,
    /rm\s+-rf/i,
    /DROP\s+TABLE/i,
    /sudo/i
  ];

  return highRiskPatterns.some(pattern =>
    pattern.test(toolCall.toolName) ||
    pattern.test(JSON.stringify(toolCall.args))
  );
}
```

**HITL 设计原则：**
- 🔒 **风险分级**: 根据操作风险自动分级
- 👤 **人工审批**: 高风险操作必须人工确认
- 📝 **审计日志**: 记录所有确认决策
- ⚡ **实时通知**: 立即通知相关人员

---

## 五、错误处理与智能重试

### 5.1 MathModelAgent 的智能反思机制

```python
# MathModelAgent/backend/app/core/agents/coder_agent.py
class CoderAgent(Agent):
    async def run(self, prompt: str) -> CoderToWriter:
        retry_count = 0
        max_retries = settings.MAX_RETRIES

        while retry_count < max_retries:
            try:
                response = await self.model.chat(
                    history=self.chat_history,
                    tools=coder_tools,
                    tool_choice="auto"
                )

                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]

                    if tool_call.function.name == "execute_code":
                        code = json.loads(tool_call.function.arguments)["code"]

                        # 执行代码
                        result, error_occurred, error_message = await self.code_interpreter.execute_code(code)

                        if error_occurred:
                            # 智能反思：基于错误信息生成修正提示
                            retry_count += 1
                            reflection_prompt = self._generate_reflection_prompt(
                                error_message, code, retry_count
                            )

                            await self.append_chat_history({
                                "role": "user",
                                "content": reflection_prompt
                            })

                            logger.warning(f"代码执行错误，第 {retry_count} 次重试")
                            continue
                        else:
                            # 执行成功，继续流程
                            await self.append_chat_history({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": "execute_code",
                                "content": result
                            })
                            continue

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise CoderAgentError(f"CoderAgent 执行失败，已重试 {max_retries} 次: {str(e)}")

                logger.error(f"CoderAgent 执行异常，第 {retry_count} 次重试: {str(e)}")
                await asyncio.sleep(2 ** retry_count)  # 指数退避
                continue

    def _generate_reflection_prompt(self, error_message: str, failed_code: str, retry_count: int) -> str:
        """生成智能反思提示"""
        return f"""
代码执行出现错误，请分析错误原因并修正代码。

错误信息: {error_message}

失败的代码:
```python
{failed_code}
```

这是第 {retry_count} 次重试，请：
1. 仔细分析错误原因
2. 修正代码中的问题
3. 确保代码能够正确执行

请提供修正后的代码。
"""
```

### 5.2 分层重试策略

```python
# 工具级别重试
class BaseCodeInterpreter:
    async def execute_code_with_retry(self, code: str, max_retries: int = 3) -> tuple[str, bool, str]:
        """工具级别的重试机制"""
        for attempt in range(max_retries):
            try:
                result, error_occurred, error_message = await self.execute_code(code)

                if not error_occurred:
                    return result, error_occurred, error_message

                # 简单错误的自动修复
                if self._is_simple_error(error_message):
                    fixed_code = self._auto_fix_code(code, error_message)
                    if fixed_code != code:
                        code = fixed_code
                        continue

                # 无法自动修复，返回错误
                return result, error_occurred, error_message

            except Exception as e:
                if attempt == max_retries - 1:
                    return "", True, str(e)
                await asyncio.sleep(1)

        return "", True, "重试次数已达上限"

    def _is_simple_error(self, error_message: str) -> bool:
        """判断是否为简单错误"""
        simple_errors = [
            "ModuleNotFoundError",
            "ImportError",
            "NameError",
            "IndentationError"
        ]
        return any(error in error_message for error in simple_errors)

    def _auto_fix_code(self, code: str, error_message: str) -> str:
        """自动修复简单错误"""
        if "ModuleNotFoundError" in error_message:
            # 自动添加缺失的 import
            missing_module = self._extract_missing_module(error_message)
            return f"import {missing_module}\n{code}"

        if "IndentationError" in error_message:
            # 自动修复缩进
            return self._fix_indentation(code)

        return code
```

**多层次错误处理：**
- 🔧 **自动修复**: 简单错误的自动修复
- 🧠 **智能反思**: 基于错误信息的 LLM 分析
- 🔄 **分层重试**: 工具级 + Agent 级重试
- 📊 **错误分析**: 错误模式的统计分析

---

## 六、框架工具系统对比分析

### 6.1 架构模式对比

| 框架 | 工具定义方式 | 执行模式 | 错误处理 | HITL 支持 |
|------|-------------|----------|----------|-----------|
| **MathModelAgent** | JSON Schema | 自主实现 | 智能反思重试 | ❌ 无内置支持 |
| **OpenAI Agents SDK** | 装饰器 | 标准化执行器 | 护栏机制 | ✅ 部分支持 |
| **ADK** | Function Declaration | 内置执行器 | 自动重试 | ✅ 完整支持 |
| **Vercel AI SDK** | TypeScript 对象 | 回调函数 | 异常捕获 | ✅ 手动实现 |

### 6.2 代码执行器对比

| 执行器类型 | 实现框架 | 隔离性 | 性能 | 成本 | 适用场景 |
|-----------|----------|--------|------|------|----------|
| **本地 Jupyter** | MathModelAgent | 中等 | 极高 | 免费 | 开发测试、个人使用 |
| **E2B 沙盒** | MathModelAgent | 极高 | 高 | 按使用付费 | 生产环境 |
| **OpenAI Code Interpreter** | OpenAI Agents SDK | 极高 | 中等 | 按调用付费 | 集成简单场景 |
| **Vertex AI Code Executor** | ADK | 极高 | 高 | Google 计费 | Google 云环境 |

### 6.3 Function Calling 实现深度对比

```python
# MathModelAgent: 完全自主实现
if tool_call.function.name == "execute_code":
    code = json.loads(tool_call.function.arguments)["code"]
    result = await self.code_interpreter.execute_code(code)
    # 完全控制执行逻辑

# OpenAI Agents SDK: 装饰器抽象
@function_tool
async def execute_code(code: str) -> str:
    return await interpreter.execute(code)
# 高度抽象，易于使用

# ADK: 声明式配置
tools=[
    ToolConfig(
        function_declarations=[
            FunctionDeclaration(
                name="execute_code",
                description="Execute Python code",
                parameters=code_schema
            )
        ]
    )
]
# 配置驱动，生产级特性

# Vercel AI SDK: TypeScript 原生
tools: {
  executeCode: {
    description: 'Execute Python code',
    parameters: z.object({
      code: z.string(),
    }),
    execute: async ({ code }) => {
      return await executeCode(code);
    },
  },
}
// 类型安全，开发体验佳
```

---

## 七、最佳实践与设计原则

### 7.1 工具设计原则

1. **单一职责**: 每个工具专注一个特定功能
2. **幂等性**: 相同输入产生相同输出
3. **错误透明**: 清晰的错误信息和恢复机制
4. **性能可观测**: 执行时间、成功率等指标监控

### 7.2 安全执行原则

```python
# 安全执行检查清单
class ToolSafetyChecker:
    def __init__(self):
        self.dangerous_patterns = [
            r'rm\s+-rf\s+/',      # 删除系统文件
            r'sudo\s+',           # 权限提升
            r'__import__\s*\(',   # 动态导入
            r'eval\s*\(',         # 代码执行
            r'exec\s*\(',         # 代码执行
        ]

    def is_safe_code(self, code: str) -> tuple[bool, str]:
        """检查代码是否安全"""
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"检测到危险操作: {pattern}"
        return True, ""

    def sanitize_output(self, output: str) -> str:
        """清理输出中的敏感信息"""
        # 移除可能的密钥、密码等敏感信息
        output = re.sub(r'api[_-]?key[_-]?=?["\']?[\w-]+', '[API_KEY_REDACTED]', output, flags=re.IGNORECASE)
        output = re.sub(r'password[_-]?=?["\']?[\w-]+', '[PASSWORD_REDACTED]', output, flags=re.IGNORECASE)
        return output
```

### 7.3 性能优化策略

1. **连接池**: 复用代码执行器连接
2. **预热机制**: 预先初始化常用环境
3. **结果缓存**: 缓存相同代码的执行结果
4. **并发控制**: 限制同时执行的工具数量

---

## 总结

工具系统与执行层是 AI Agent 从"能说"到"能做"的关键跨越。通过对比分析四个主流框架的实现，我们总结出以下关键洞察：

### 🎯 **核心理念**
- **职责分离**: LLM 负责决策，系统负责执行
- **标准协议**: 基于 OpenAI Function Calling 确保兼容性
- **安全第一**: 沙盒隔离 + HITL 确认 + 安全检查

### 🛠️ **技术选型指南**
- **MathModelAgent**: 需要完全控制和深度定制
- **OpenAI Agents SDK**: 企业级应用，标准化需求
- **ADK**: Google 云环境，生产级 HITL
- **Vercel AI SDK**: 全栈应用，开发体验优先

### 🚀 **进化方向**
- **多模态工具**: 图像、音频、视频处理能力
- **自适应执行**: 根据上下文智能选择执行策略
- **联邦工具**: 跨组织的工具共享和协作
- **实时协作**: 人机协作的实时交互模式

这一层的设计质量直接决定了 AI Agent 的实际能力边界。只有构建了安全、可靠、高效的工具执行层，Agent 才能从聊天机器人真正进化为生产力工具。
