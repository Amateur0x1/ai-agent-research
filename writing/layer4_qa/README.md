# 第四层：多智能体架构与协作层 - Q&A

## Q1: 什么是多智能体系统？为什么需要多智能体架构？

**A:** 多智能体系统是由多个自主智能体组成的系统，这些智能体通过协作和通信来完成复杂的任务。

**需要多智能体架构的原因：**

1. **专业化分工**：单个智能体难以在多个领域都达到专家水平
2. **可扩展性**：通过增加智能体来扩展系统能力
3. **容错性**：单个智能体失败不影响整体系统
4. **并行处理**：多个智能体可以并行工作，提高效率
5. **模块化设计**：便于开发、测试和维护

**示例：**
```python
# 单智能体 vs 多智能体
# 单智能体：需要处理所有任务
single_agent = Agent(
    name="general_agent",
    tools=[web_search, code_execution, writing, analysis, ...]
)

# 多智能体：专业化分工
search_agent = Agent(name="searcher", tools=[web_search])
code_agent = Agent(name="coder", tools=[code_execution])
write_agent = Agent(name="writer", tools=[writing])
analysis_agent = Agent(name="analyst", tools=[analysis])
```

## Q2: 层次化架构和线性流水线架构有什么区别？如何选择？

**A:** 两种架构的主要区别在于智能体间的关系和执行模式。

### 层次化架构
```python
# ADK 层次化示例
coordinator = LlmAgent(
    name="Coordinator",
    sub_agents=[agent_a, agent_b, agent_c]  # 父子关系
)

# 特点：
# - 树状结构
# - 上下文继承
# - 智能路由
# - 权限控制
```

### 线性流水线架构
```python
# MathModelAgent 流水线示例
coordinator_response = await coordinator.run(input)
modeler_response = await modeler.run(coordinator_response)
coder_response = await coder.run(modeler_response)
writer_response = await writer.run(coder_response)

# 特点：
# - 顺序执行
# - 数据传递
# - 确定性高
# - 错误隔离
```

### 选择指南

| 场景 | 推荐架构 | 原因 |
|------|----------|------|
| 企业级应用 | 层次化 | 权责清晰、易于管理 |
| 标准化流程 | 线性流水线 | 确定性高、结果可预测 |
| 复杂决策 | 图网络 | 灵活性强、动态路由 |
| 开放对话 | Swarm | 自组织、状态记忆 |

## Q3: Handoff 机制是如何工作的？不同框架的实现有什么差异？

**A:** Handoff 是智能体间控制权转移的核心机制，不同框架有不同的实现方式。

### LangGraph 的 Handoff
```python
def create_handoff_tool(agent_name: str):
    @tool(f"transfer_to_{agent_name}")
    def handoff_tool(state, tool_call_id):
        return Command(
            goto=agent_name,           # 转移目标
            update={"messages": [...]}, # 状态更新
            graph=Command.PARENT       # 父图导航
        )
    return handoff_tool

# 特点：
# - 基于工具调用
# - Command 对象控制
# - 支持状态更新
# - 图级别导航
```

### OpenAI Agents SDK 的 Handoff
```python
from agents import Agent, handoff

# 简单 handoff
triage_agent = Agent(
    name="Triage agent",
    handoffs=[billing_agent, handoff(refund_agent)]
)

# 自定义 handoff
handoff_obj = handoff(
    agent=agent,
    on_handoff=callback_func,      # 回调函数
    input_filter=custom_filter,    # 输入过滤
    tool_name_override="custom_name" # 工具名覆盖
)

# 特点：
# - 声明式配置
# - 回调机制
# - 输入过滤
# - 易于使用
```

### ADK 的智能体转移
```python
# LLM 驱动的转移
def transfer_to_agent(agent_name: str, tool_context: ToolContext):
    tool_context.actions.transfer_to_agent = agent_name

# 特点：
# - LLM 决策驱动
# - 上下文自动继承
# - 层次化支持
# - 配置简单
```

## Q4: 如何设计智能体间的数据传递协议？

**A:** 好的数据传递协议应该确保类型安全、版本兼容和易于扩展。

### 1. 使用强类型数据结构
```python
# Pydantic 模型定义
class CoordinatorToModeler(BaseModel):
    questions: dict
    ques_count: int
    timestamp: datetime

class ModelerToCoder(BaseModel):
    questions_solution: dict[str, str]
    model_type: str
    complexity_level: int

class CoderToWriter(BaseModel):
    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None
    execution_time: float
```

### 2. 版本化协议
```python
class MessageEnvelope(BaseModel):
    version: str = "1.0"
    sender: str
    receiver: str
    timestamp: datetime
    payload: dict

    def is_compatible(self, required_version: str) -> bool:
        return self.version.startswith(required_version.split('.')[0])
```

### 3. 错误处理协议
```python
class AgentError(BaseModel):
    error_code: str
    error_message: str
    error_details: dict
    retry_after: int | None = None

class AgentResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: AgentError | None = None
```

## Q5: 如何处理多智能体系统中的错误和异常？

**A:** 多智能体系统需要分层级的错误处理策略。

### 1. 智能体级错误处理
```python
class ResilientAgent(Agent):
    async def run_with_retry(self, input_data, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await self.run(input_data)
            except Exception as e:
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，记录并上报
                    await self.report_failure(e, input_data)
                    raise AgentExecutionError(
                        agent_name=self.name,
                        error=str(e),
                        attempts=max_retries
                    )
                else:
                    # 智能重试
                    await self.smart_retry(e, attempt)
```

### 2. 系统级错误恢复
```python
class ErrorRecoveryManager:
    async def handle_agent_failure(self, failed_agent: str, error: Exception):
        """处理智能体失败"""

        # 1. 记录失败信息
        await self.log_failure(failed_agent, error)

        # 2. 尝试恢复策略
        recovery_strategy = await self.select_recovery_strategy(failed_agent, error)

        if recovery_strategy == "retry":
            await self.retry_agent(failed_agent)
        elif recovery_strategy == "fallback":
            await self.use_fallback_agent(failed_agent)
        elif recovery_strategy == "skip":
            await self.skip_agent(failed_agent)
        else:
            await self.abort_workflow()
```

### 3. 断路器模式
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e
```

## Q6: 如何监控和调试多智能体系统？

**A:** 多智能体系统需要全面的监控和调试工具。

### 1. 执行追踪
```python
class ExecutionTracer:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.events = []

    async def trace_agent_execution(self, agent_name: str, input_data, output_data):
        event = {
            "trace_id": self.trace_id,
            "agent": agent_name,
            "timestamp": datetime.now(),
            "input_hash": hash(str(input_data)),
            "output_hash": hash(str(output_data)),
            "execution_time": None
        }

        start_time = time.time()
        try:
            # 执行智能体
            result = await self.execute_agent(agent_name, input_data)
            event["status"] = "success"
            return result
        except Exception as e:
            event["status"] = "error"
            event["error"] = str(e)
            raise
        finally:
            event["execution_time"] = time.time() - start_time
            self.events.append(event)
            await self.save_trace(event)
```

### 2. 可视化调试
```python
class WorkflowVisualizer:
    def generate_execution_graph(self, trace_data):
        """生成执行流程图"""

        graph = {
            "nodes": [],
            "edges": []
        }

        for event in trace_data:
            # 添加节点
            node = {
                "id": event["agent"],
                "label": event["agent"],
                "status": event["status"],
                "duration": event["execution_time"]
            }
            graph["nodes"].append(node)

            # 添加边
            if event.get("next_agent"):
                edge = {
                    "from": event["agent"],
                    "to": event["next_agent"],
                    "data_flow": event.get("data_summary", "")
                }
                graph["edges"].append(edge)

        return graph
```

### 3. 性能监控
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record_agent_metrics(self, agent_name: str, metrics: dict):
        """记录智能体性能指标"""
        self.metrics[agent_name].append({
            "timestamp": datetime.now(),
            "execution_time": metrics["execution_time"],
            "token_usage": metrics["token_usage"],
            "success_rate": metrics["success_rate"],
            "error_count": metrics["error_count"]
        })

    def get_performance_summary(self, agent_name: str):
        """获取性能摘要"""
        agent_metrics = self.metrics[agent_name]
        if not agent_metrics:
            return None

        return {
            "avg_execution_time": np.mean([m["execution_time"] for m in agent_metrics]),
            "avg_token_usage": np.mean([m["token_usage"] for m in agent_metrics]),
            "success_rate": np.mean([m["success_rate"] for m in agent_metrics]),
            "total_errors": sum([m["error_count"] for m in agent_metrics])
        }
```

## Q7: 如何测试多智能体系统？

**A:** 多智能体系统需要多层次的测试策略。

### 1. 单元测试
```python
class TestAgentUnit:
    async def test_coordinator_agent(self):
        """测试协调员智能体"""
        agent = CoordinatorAgent("test_task", mock_llm)

        # 测试正常输入
        result = await agent.run("数学建模问题：优化生产计划")
        assert result.ques_count > 0
        assert "ques1" in result.questions

        # 测试异常输入
        with pytest.raises(ValueError):
            await agent.run("")  # 空输入
```

### 2. 集成测试
```python
class TestAgentIntegration:
    async def test_pipeline_execution(self):
        """测试完整流水线执行"""
        workflow = MathModelWorkFlow()

        # 模拟输入
        problem = Problem(
            task_id="test_001",
            ques_all="请优化工厂生产计划"
        )

        # 执行完整流程
        result = await workflow.execute(problem)

        # 验证输出
        assert result.status == "completed"
        assert len(result.sections) > 0
        assert result.paper_content is not None
```

### 3. 性能测试
```python
class TestPerformance:
    async def test_concurrent_execution(self):
        """测试并发执行性能"""
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                self.execute_workflow(f"task_{i}")
            )
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # 验证性能指标
        assert execution_time < 60  # 10个任务在60秒内完成
        assert all(r.status == "completed" for r in results)
```

## Q8: 如何优化多智能体系统的性能？

**A:** 性能优化需要从多个维度考虑。

### 1. 智能体并行化
```python
class ParallelWorkflow:
    async def execute_parallel_tasks(self, tasks):
        """并行执行独立任务"""

        # 识别可并行执行的任务
        parallel_groups = self.identify_parallel_tasks(tasks)

        results = {}
        for group in parallel_groups:
            # 并行执行同一组内的任务
            group_tasks = [
                asyncio.create_task(agent.run(task))
                for agent, task in group
            ]
            group_results = await asyncio.gather(*group_tasks)

            # 收集结果
            for (agent, task), result in zip(group, group_results):
                results[f"{agent.name}_{task.id}"] = result

        return results
```

### 2. 缓存策略
```python
class AgentCache:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}

    async def get_cached_result(self, agent_name: str, input_hash: str):
        """获取缓存结果"""
        cache_key = f"{agent_name}:{input_hash}"

        if cache_key in self.cache:
            if time.time() < self.cache_ttl[cache_key]:
                return self.cache[cache_key]
            else:
                del self.cache[cache_key]
                del self.cache_ttl[cache_key]

        return None

    async def cache_result(self, agent_name: str, input_hash: str, result, ttl=3600):
        """缓存结果"""
        cache_key = f"{agent_name}:{input_hash}"
        self.cache[cache_key] = result
        self.cache_ttl[cache_key] = time.time() + ttl
```

### 3. 资源池化
```python
class AgentPool:
    def __init__(self, agent_class, pool_size=5):
        self.agent_class = agent_class
        self.pool_size = pool_size
        self.available_agents = asyncio.Queue(maxsize=pool_size)
        self.all_agents = []

        # 预创建智能体池
        for _ in range(pool_size):
            agent = agent_class()
            self.all_agents.append(agent)
            self.available_agents.put_nowait(agent)

    async def get_agent(self):
        """获取可用智能体"""
        return await self.available_agents.get()

    async def return_agent(self, agent):
        """归还智能体到池中"""
        await self.available_agents.put(agent)

    async def execute_with_agent(self, task):
        """使用池中智能体执行任务"""
        agent = await self.get_agent()
        try:
            return await agent.run(task)
        finally:
            await self.return_agent(agent)
```

通过这些Q&A，我们可以更好地理解和应用多智能体架构与协作层的各种技术和最佳实践。
