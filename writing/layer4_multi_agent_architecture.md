# 第四层：多智能体架构与协作层

## 概述

多智能体架构与协作层是 AI Agent 系统的核心组织结构，定义了多个智能体之间的分工、通信和协作机制。该层通过合理的架构设计，将复杂的任务分解为多个专业化智能体，实现"分而治之"的策略，从而提高系统的整体性能和可维护性。

## 核心职责

- **专业化分工**：将复杂任务按领域或功能分解为专门的智能体
- **协作机制**：定义智能体间的通信协议和数据传递方式
- **流程编排**：控制智能体的执行顺序和条件切换
- **状态管理**：维护跨智能体的上下文和状态一致性
- **容错处理**：处理智能体执行失败和异常情况

## 主流架构模式

### 1. 层次化架构（Hierarchical Architecture）

层次化架构采用父子关系组织智能体，通过上下级委托实现任务分配和执行。

#### ADK 的层次化实现

ADK（Agent Development Kit）采用配置驱动的层次化设计，支持多种智能体类型：

```python
# ADK 层次化智能体配置
from google.adk.agents import LlmAgent

# 子智能体定义
booking_agent = LlmAgent(
    name="Booker",
    description="Handles flight and hotel bookings."
)

info_agent = LlmAgent(
    name="Info",
    description="Provides general information and answers questions."
)

# 父智能体（协调者）
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction="You are an assistant. Delegate booking tasks to Booker and info requests to Info.",
    description="Main coordinator.",
    sub_agents=[booking_agent, info_agent]  # 层次化关系
)
```

**核心特性：**
- **上下文继承**：子智能体自动继承父智能体的对话历史和上下文
- **智能转移**：基于 LLM 理解的动态任务分配
- **权限控制**：不同级别智能体的权限和功能限制

#### 智能体转移机制

ADK 提供了灵活的智能体转移机制：

```python
# ADK 智能体转移工具
def transfer_to_agent(agent_name: str, tool_context: ToolContext) -> None:
    """将问题转移给另一个智能体

    Args:
        agent_name: 要转移到的智能体名称
        tool_context: 工具上下文
    """
    tool_context.actions.transfer_to_agent = agent_name
```

**转移决策逻辑：**
```python
class AutoFlow(BaseLlmFlow):
    """自动流程控制"""

    async def _handle_agent_transfer(self, ctx: InvocationContext) -> bool:
        """处理智能体间的转移"""

        # 1. 分析当前上下文
        transfer_decision = await self._analyze_transfer_need(ctx)

        # 2. 选择目标智能体
        if transfer_decision.should_transfer:
            target_agent = self._select_target_agent(
                ctx.agent,
                transfer_decision.reason
            )

            # 3. 执行转移
            if target_agent:
                ctx.transfer_to_agent(target_agent)
                return True

        return False
```

### 2. 线性流水线架构（Linear Pipeline Architecture）

线性流水线架构按照预定义的顺序执行智能体，每个智能体的输出作为下一个智能体的输入。

#### MathModelAgent 的流水线实现

MathModelAgent 采用经典的线性流水线模式，专门针对数学建模场景优化：

```python
# MathModelAgent 流水线架构
class MathModelWorkFlow:
    async def execute(self, problem: Problem):
        # 1. 协调员：问题理解和格式化
        coordinator_response = await coordinator_agent.run(problem.ques_all)

        # 2. 建模手：数学建模设计
        modeler_response = await modeler_agent.run(coordinator_response)

        # 3. 代码手：代码实现执行（循环处理子问题）
        for subtask in solution_flows:
            coder_response = await coder_agent.run(subtask)
            writer_response = await writer_agent.run(coder_response)

        # 4. 写作手：论文撰写
        for key, value in write_flows.items():
            writer_response = await writer_agent.run(value)
```

**专业化智能体定义：**

```python
# 基础 Agent 类
class Agent:
    def __init__(self, task_id: str, model: LLM, max_chat_turns: int = 30):
        self.task_id = task_id
        self.model = model
        self.chat_history: list[dict] = []
        self.max_chat_turns = max_chat_turns

# 专业化智能体
class CoordinatorAgent(Agent):
    """协调员：问题理解和格式化"""
    async def run(self, ques_all: str) -> CoordinatorToModeler:
        # 将用户问题转换为结构化 JSON
        pass

class ModelerAgent(Agent):
    """建模手：数学建模设计"""
    async def run(self, coordinator_to_modeler: CoordinatorToModeler) -> ModelerToCoder:
        # 设计数学模型和求解方案
        pass

class CoderAgent(Agent):
    """代码手：代码实现执行"""
    async def run(self, modeler_to_coder: ModelerToCoder) -> CoderToWriter:
        # 编写和执行代码，处理错误重试
        pass

class WriterAgent(Agent):
    """写作手：学术论文撰写"""
    async def run(self, coder_to_writer: CoderToWriter) -> WriterResponse:
        # 撰写学术论文章节
        pass
```

**数据传递协议：**
```python
# 标准化的数据结构
class CoordinatorToModeler(BaseModel):
    questions: dict
    ques_count: int

class ModelerToCoder(BaseModel):
    questions_solution: dict[str, str]

class CoderToWriter(BaseModel):
    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None
```

**设计优势：**
- **确定性执行**：固定的执行顺序，结果可预测
- **专业化优化**：每个智能体专注特定领域
- **错误隔离**：单个智能体失败不影响整体流程

### 3. 图网络架构（Graph Network Architecture）

图网络架构将智能体视为图中的节点，通过边和路由函数实现灵活的执行路径。

#### LangGraph 的图形化实现

LangGraph 提供了强大的图形化多智能体编排能力：

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI

# 定义路由函数
def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    """监督者：决定下一个执行的智能体"""
    response = model.invoke([
        SystemMessage(content="You are a supervisor. Decide which agent should handle the request."),
        *state["messages"]
    ])

    # 根据监督者决策路由到相应智能体
    return Command(goto=response["next_agent"])

def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
    """智能体1：处理特定任务"""
    response = model.invoke(state["messages"])
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
    """智能体2：处理其他任务"""
    response = model.invoke(state["messages"])
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

# 构建多智能体图
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)

builder.add_edge(START, "supervisor")
builder.add_edge("agent_1", "supervisor")
builder.add_edge("agent_2", "supervisor")

graph = builder.compile()
```

#### Handoff 机制

LangGraph 通过 Handoff 机制实现智能体间的控制权转移：

```python
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """创建智能体转移工具"""
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  # 转移到目标智能体
            update={"messages": [tool_message]},  # 更新状态
            graph=Command.PARENT,  # 在父图中导航
        )

    return handoff_tool

# 为智能体添加转移工具
flight_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_flight, create_handoff_tool(agent_name="hotel_assistant")],
    prompt="You are a flight booking assistant",
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_hotel, create_handoff_tool(agent_name="flight_assistant")],
    prompt="You are a hotel booking assistant",
    name="hotel_assistant"
)
```

### 4. Swarm 架构（群体协作架构）

Swarm 架构允许智能体基于专业化动态转移控制权，系统记住最后活跃的智能体。

```python
from langgraph_swarm import createSwarm, createHandoffTool

# 创建转移工具
const transferToHotelAssistant = createHandoffTool({
  agentName: "hotel_assistant",
  description: "Transfer user to the hotel-booking assistant.",
});

const transferToFlightAssistant = createHandoffTool({
  agentName: "flight_assistant",
  description: "Transfer user to the flight-booking assistant.",
});

// 定义智能体
const flightAssistant = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  tools: [bookFlight, transferToHotelAssistant],
  stateModifier: "You are a flight booking assistant",
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  tools: [bookHotel, transferToFlightAssistant],
  stateModifier: "You are a hotel booking assistant",
  name: "hotel_assistant",
});

// 创建 Swarm 系统
const swarm = createSwarm({
  agents: [flightAssistant, hotelAssistant],
  defaultActiveAgent: "flight_assistant",
});
```

## OpenAI Agents SDK 的 Handoff 机制

OpenAI Agents SDK 提供了简洁的 Handoff API，支持智能体间的无缝转移：

```python
from agents import Agent, handoff

# 创建智能体
billing_agent = Agent(name="Billing agent")
refund_agent = Agent(name="Refund agent")

# 使用 handoff 创建转移关系
triage_agent = Agent(
    name="Triage agent",
    handoffs=[billing_agent, handoff(refund_agent)]
)

# 自定义 handoff
def on_handoff(ctx: RunContextWrapper[None]):
    print("Handoff called")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    tool_name_override="custom_handoff_tool",
    tool_description_override="Custom description",
)

# 带输入数据的 handoff
class EscalationData(BaseModel):
    reason: str

async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent called with reason: {input_data.reason}")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    input_type=EscalationData,
)
```

## 协作模式对比

| 架构模式 | 优势 | 劣势 | 适用场景 |
|---------|------|------|----------|
| **层次化** | 权责清晰、易于管理、上下文继承 | 灵活性较低、单点故障风险 | 企业级应用、标准化流程 |
| **线性流水线** | 确定性高、错误隔离、专业化强 | 顺序依赖、并行度低 | 专业领域任务、固定流程 |
| **图网络** | 灵活性强、动态路由、支持并行 | 复杂度高、调试困难 | 复杂推理、动态决策 |
| **Swarm** | 自组织、动态协作、状态记忆 | 控制难度大、一致性挑战 | 开放式对话、个性化服务 |

## 状态管理与通信

### 1. 状态传递机制

```python
# ADK 的状态继承
class InvocationContext:
    """调用上下文，支持状态在智能体间传递"""

    def __init__(self):
        self.state = {}
        self.agent_stack = []  # 智能体调用栈

    def transfer_to_agent(self, target_agent: BaseAgent):
        """转移控制权并传递状态"""
        self.agent_stack.append(self.current_agent)
        self.current_agent = target_agent
        # 状态自动继承
```

### 2. 消息传递协议

```python
# LangGraph 的 Command 机制
class Command:
    """命令对象，结合控制流和状态更新"""

    def __init__(self, goto: str = None, update: dict = None, graph: str = None):
        self.goto = goto          # 下一个智能体
        self.update = update      # 状态更新
        self.graph = graph        # 图导航指令

# 使用示例
return Command(
    goto="agent_2",              # 转移到 agent_2
    update={"messages": [...]}    # 更新消息状态
)
```

### 3. 上下文过滤

```python
# OpenAI Agents SDK 的输入过滤
from agents.extensions import handoff_filters

handoff_obj = handoff(
    agent=agent,
    input_filter=handoff_filters.remove_all_tools,  # 移除所有工具调用
)

# 自定义过滤器
def custom_filter(input_data: HandoffInputData) -> HandoffInputData:
    """自定义上下文过滤逻辑"""
    filtered_messages = [
        msg for msg in input_data.messages
        if msg.role != "system"  # 移除系统消息
    ]
    return HandoffInputData(
        messages=filtered_messages,
        input_data=input_data.input_data
    )
```

## 容错与监控

### 1. 智能体级容错

```python
# MathModelAgent 的重试机制
class CoderAgent(Agent):
    async def run(self, prompt: str) -> CoderToWriter:
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = await self.model.chat(tools=coder_tools)
                if has_tool_calls:
                    result = await self.code_interpreter.execute_code(code)
                    if error_occurred:
                        # 智能反思错误
                        error_analysis = await self.analyze_error(error)
                        retry_count += 1
                        continue
                    else:
                        return result
                else:
                    return result
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"执行失败: {e}")
```

### 2. 系统级监控

```python
# ADK 的事件驱动监控
class AgentEvent:
    """智能体事件"""

    def __init__(self, agent_name: str, event_type: str, data: dict):
        self.agent_name = agent_name
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()

class EventMonitor:
    """事件监控器"""

    async def on_agent_start(self, event: AgentEvent):
        """智能体开始执行"""
        logger.info(f"Agent {event.agent_name} started")

    async def on_agent_complete(self, event: AgentEvent):
        """智能体执行完成"""
        logger.info(f"Agent {event.agent_name} completed")

    async def on_agent_error(self, event: AgentEvent):
        """智能体执行错误"""
        logger.error(f"Agent {event.agent_name} failed: {event.data}")
```

## 最佳实践

### 1. 专业化设计原则

- **单一职责**：每个智能体专注特定领域
- **边界清晰**：明确智能体的职责边界
- **接口标准**：定义统一的数据交换格式

### 2. 协作模式选择

- **简单任务**：使用线性流水线
- **复杂决策**：使用图网络架构
- **企业应用**：使用层次化架构
- **开放对话**：使用 Swarm 架构

### 3. 状态管理策略

- **最小状态**：只传递必要的状态信息
- **不可变性**：状态更新采用不可变模式
- **版本控制**：为状态添加版本信息

### 4. 错误处理原则

- **快速失败**：尽早发现和报告错误
- **优雅降级**：提供备选执行路径
- **完整日志**：记录详细的执行轨迹

## 总结

多智能体架构与协作层是构建复杂 AI 系统的关键。通过选择合适的架构模式、设计有效的协作机制、实现可靠的状态管理，我们可以构建出既灵活又可靠的多智能体系统。

**核心价值：**
- **模块化**：将复杂系统分解为可管理的模块
- **专业化**：每个智能体专注特定领域，提高效率
- **可扩展**：易于添加新的智能体和功能
- **容错性**：单个智能体失败不影响整体系统

**技术趋势：**
- **动态编排**：从固定流程向动态决策发展
- **自适应协作**：智能体根据上下文自动调整协作策略
- **异构集成**：支持不同类型智能体的无缝集成
- **智能监控**：基于 AI 的系统监控和优化

通过深入理解和应用这些架构模式，我们可以构建出更加强大和智能的 AI Agent 系统。
