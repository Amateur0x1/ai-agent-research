# 第六层：实时通信与人机协作层（HITL）- 技术实现详解

## 概述

实时通信与人机协作（Human-in-the-Loop，HITL）是生产级 Agent 的关键能力：一方面需要将系统内部状态以事件方式实时呈现给用户界面；另一方面要在关键节点安全地引入人工审核、确认或补充信息，并能可靠地暂停与恢复执行。本层聚焦四类落地能力：

- 实时事件流（SSE/WebSocket）与协议约定
- 人在回路的中断/恢复机制（基于检查点）
- 高风险工具调用的人工确认流程
- UI 状态与多端一致性的可靠传输

本文基于项目中 LangGraph、MathModelAgent、ADK（工具确认）、AG-UI 事件协议的真实实现进行拆解与对比，给出工程化落地方案。

---

## 一、实时通信：事件流与消息通道

### 1.1 MathModelAgent：WebSocket + Redis 发布订阅

在 MathModelAgent 中，后端通过 Redis 作为消息总线，前端通过 WebSocket 订阅任务级事件，形成「后端推送/前端实时渲染」的链路：

```python
# backend/app/routers/ws_router.py
@router.websocket("/task/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    # 1) 校验 task_id 是否存在于 Redis
    # 2) 建立 WebSocket 连接并加入连接管理器
    # 3) 订阅 Redis 频道 task:{task_id}:messages
    # 4) 循环读取 Redis 消息，转发给 WebSocket 客户端
```

任务入口在 `modeling_router` 中触发后台执行，并在关键阶段通过 Redis 广播状态：

```python
# backend/app/routers/modeling_router.py
await redis_manager.publish_message(task_id, SystemMessage(content="任务开始处理"))
task = asyncio.create_task(MathModelWorkFlow().execute(problem))
...
await redis_manager.publish_message(task_id, SystemMessage(content="任务处理完成", type="success"))
```

前端使用 `TaskWebSocket` 建立连接并维护消息列表：

```ts
// frontend/src/stores/task.ts
function connectWebSocket(taskId: string) {
  const wsUrl = `${baseUrl}/task/${taskId}`
  ws = new TaskWebSocket(wsUrl, (data) => {
    messages.value.push(data)
  })
  ws.connect()
}
```

要点：
- **解耦**：Redis 作为后端内部总线，WebSocket 仅负责对外分发
- **任务隔离**：频道按 `task:{task_id}:messages` 划分
- **结构化消息**：统一 `SystemMessage`/`AgentMessage` 等 Schema，便于前端渲染

### 1.2 AG-UI：标准化事件协议（SSE/二进制）

AG-UI 定义了通用的事件协议与类型体系，支持 HTTP SSE 与高性能二进制传输：

```typescript
// docs/concepts/architecture.mdx（摘要）
// 支持的传输：
// - SSE（文本）
// - HTTP Binary（高性能自定义传输）
// 标准事件类别：
// - Lifecycle: RUN_STARTED/RUN_FINISHED/STEP_STARTED/...
// - Text: TEXT_MESSAGE_* 逐字流
// - Tool: TOOL_CALL_* 工具调用拆分
// - State: STATE_SNAPSHOT/STATE_DELTA
```

协议层关键价值：
- **类型完备**：事件强类型定义（如 `EventType`、`TextMessageContentEvent`）
- **增量状态**：`STATE_DELTA` 使用 JSON Patch（RFC 6902）降低带宽
- **跨后端复用**：统一事件语义可适配不同 Agent 引擎

---

## 二、人在回路：中断与恢复（LangGraph）

LangGraph 原生支持通过 `interrupt` 实现动态中断，依托于检查点机制将状态持久化，从而做到「可无限期等待人工输入」并在之后恢复：

```python
# docs/concepts/human_in_the_loop.md & how-tos/add-human-in-the-loop.md（摘要）
from langgraph.types import interrupt, Command

def node(state: State):
    # 在节点内触发动态中断，将问题或上下文回传给客户端
    answer = interrupt("what is your age?")
    # 恢复时携带 Command 注入的人类答案继续执行
    return {"human_value": answer}

# 要求：
# 1) 编译图时启用 checkpointer（每个超级步骤自动检查点）
# 2) 运行时指定 thread_id（用于恢复定位）
```

关键能力：
- **持久化暂停**：依赖检查点/线程状态，可长时间等待
- **可恢复执行**：恢复后从节点起点重新执行，保证一致性
- **灵活断点**：动态（interrupt）与静态（interrupt_before/after）均可用

与实时通信结合建议：
- 中断触发时向前端发送 `RUN_PAUSED`/自定义事件，附带 `interrupt` 的上下文提示
- 前端收集人类输入后，通过后端 API 调用向图提交 `Command` 恢复执行

---

## 三、工具调用的人工确认（ADK）

ADK 在工具层面内置「确认门」：当工具被标记为需要确认（或满足阈值函数）时，自动暂停并等待用户确认结果，再继续执行。

### 3.1 工具级确认阈值

```python
# src/google/adk/tools/function_tool.py（摘要）
if require_confirmation:
  if not tool_context.tool_confirmation:
    tool_context.request_confirmation(
      hint="Please approve or reject ...",
    )
    return {"error": "This tool call requires confirmation ..."}
  elif not tool_context.tool_confirmation.confirmed:
    return {"error": "This tool call is rejected."}

# -> 未确认时返回提示；确认结果写入 tool_context 后再继续真实执行
```

### 3.2 业务样例：请假审批（HITL）

```python
# contributing/samples/human_tool_confirmation/agent.py（摘要）
def request_time_off(days: int, tool_context: ToolContext):
  if days > 2:
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(hint="请审批该工具调用...")
      return {"status": "Manager approval is required."}
    # 已有确认结果 -> 按确认值继续
```

### 3.3 事件链路（前后端交互）

```python
# flows/llm_flows/request_confirmation.py（摘要）
# 解析会话事件中用户提供的 FunctionResponse -> ToolConfirmation
# 将之回填到调用链，解除等待并继续工具执行
```

要点：
- **确认粒度在工具层**：比工作流级暂停更细，安全边界清晰
- **提示/负载可定制**：`hint` 与默认 `payload` 帮助 UI 生成确认表单
- **与 UI 协议相容**：可映射为 AG-UI 的 `TOOL_CALL_*` 与自定义确认事件

---

## 四、UI 协议与状态一致性（AG-UI）

AG-UI 的事件协议可作为「后端-前端」的通用契约，将 LangGraph/ADK/自研的内部状态流标准化输出：

- **生命周期**：`RUN_STARTED/STEP_STARTED/.../RUN_FINISHED`
- **逐字流**：`TEXT_MESSAGE_START/TEXT_MESSAGE_CONTENT/TEXT_MESSAGE_END`
- **工具调用**：`TOOL_CALL_START/ARGS/END`（可在 ARGS 中逐步填充参数）
- **状态同步**：`STATE_SNAPSHOT/STATE_DELTA`（前端以补丁方式高效更新）

映射建议：
- MathModelAgent 的 Redis 消息 -> 转换为 AG-UI 事件枚举
- ADK 工具确认 -> 映射为 `TOOL_CALL_*` + 自定义 `CONFIRMATION_REQUIRED` 事件
- LangGraph `interrupt` -> 自定义 `HUMAN_INPUT_REQUIRED` 事件，恢复时带 `Command`

---

## 五、工程化实践与模式对比

### 5.1 能力对比

- **暂停/恢复**：LangGraph（基于检查点，适合任意节点） > ADK（工具级确认）
- **协议完备**：AG-UI 事件模型最完善，便于前端通用渲染
- **实现复杂度**：MathModelAgent（WebSocket+Redis）最易落地

### 5.2 组合式落地建议

1) 简单项目/单体后端：MathModelAgent 模式
- WebSocket + Redis 推送实时状态
- 自定义消息结构体即可

2) 标准化前后端对接：AG-UI 协议
- 统一事件类型，前端组件通用复用
- 支持 SSE 或二进制，兼顾兼容性与性能

3) 严格审批/可回退需求：LangGraph + ADK 组合
- 关键节点用 `interrupt` 做运行级中断
- 高风险工具用 ADK 确认门控，分层安全

---

## 六、最佳实践清单

- **分层暂停**：工具级确认（ADK）与工作流级中断（LangGraph）配合使用
- **状态可追溯**：开启检查点，所有暂停点可恢复/回放
- **协议统一**：对外统一为 AG-UI 事件，前端零胶水
- **幂等恢复**：中断恢复后节点从起点重跑，确保一致性
- **风险白名单**：仅对高风险工具开启确认，降低交互频率
- **渐进增强**：基础版先上 WebSocket + Redis，逐步引入 AG-UI/interrupt

---

## 七、总结

本层从「实时」与「可控」两条主线构建生产级体验：
- 以 Redis/WebSocket 或 AG-UI 协议实现可靠的实时可视化
- 以 LangGraph `interrupt` 与 ADK 工具确认打造可暂停、可审计、可恢复的人在回路

二者相辅相成：前者解决「看得见」，后者解决「可介入」。在工程落地中，建议采用「协议统一 + 分层暂停 + 检查点恢复」的组合方案，既保证用户体验，又确保执行的安全与可追溯性。


