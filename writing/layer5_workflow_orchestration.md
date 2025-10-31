# 第五层:工作流编排与控制层 - 技术实现详解

## 概述

工作流编排与控制层是 AI Agent 系统的"指挥中枢",负责定义任务的执行路径、控制节点间的流转逻辑,并确保整个系统按照预期顺序或条件执行。该层的核心理念是:**在保证灵活性的同时,提供确定性的执行控制**。本文将通过 **LangGraph**、**MathModelAgent**、**ADK** 和 **Vercel AI SDK** 的真实代码实现,深入分析不同场景下的工作流编排模式。

## 核心职责

1. **执行路径定义** - 明确任务的执行顺序和条件分支
2. **状态驱动路由** - 基于运行时状态动态决策下一步执行
3. **并行与串行控制** - 支持节点的顺序执行和并行执行
4. **确定性保证** - 在特定场景下提供可预测的执行路径
5. **检查点管理** - 在关键节点保存状态,支持中断和恢复

---

## 一、LangGraph:图形化工作流编排

LangGraph 采用 **图形计算模型** 来表达 Agent 工作流,通过节点 (Nodes) 和边 (Edges) 的组合实现灵活的流程控制。

### 1.1 核心概念:超级步骤 (Super-Step)

LangGraph 的执行基于离散的"超级步骤"概念,这是受 Google Pregel 系统启发的设计:

```python
# LangGraph 的超级步骤概念
# 一个超级步骤 = 一次图节点的迭代

# 并行执行的节点属于同一个超级步骤
# 串行执行的节点属于不同的超级步骤

# 执行流程:
# 1. 所有节点初始状态为 inactive
# 2. 节点接收到消息后变为 active 并执行
# 3. 执行完成后返回更新,标记为 inactive
# 4. 当所有节点都 inactive 且无消息传递时,图执行终止
```

**设计亮点:**
- **清晰的执行边界**: 每个超级步骤都是一个完整的执行单元
- **自动检查点**: 每个超级步骤结束后自动保存状态
- **并行支持**: 同一超级步骤内的节点可并行执行
- **时间旅行**: 可回退到任意超级步骤的检查点

### 1.2 图形定义:StateGraph 核心类

```python
# langgraph/libs/langgraph/langgraph/graph/state.py
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: list
    documents: list
    question: str
    generation: str

# 创建状态图
workflow = StateGraph(State)

# 添加节点 - 节点是执行工作的函数
workflow.add_node("retrieve", retrieve_documents)      # 文档检索
workflow.add_node("grade_docs", grade_documents)       # 文档评分
workflow.add_node("generate", generate_answer)         # 生成答案
workflow.add_node("web_search", web_search)            # 网络搜索

# 添加普通边 - 确定性路径
workflow.add_edge(START, "retrieve")                   # 从 START 开始
workflow.add_edge("web_search", "generate")            # web_search -> generate

# 添加条件边 - 动态路由
workflow.add_conditional_edges(
    "grade_docs",                                      # 源节点
    decide_to_generate,                                # 路由函数
    {
        "web_search": "web_search",                    # 路由映射
        "generate": "generate"
    }
)

workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "not supported": "generate",                   # 幻觉:重新生成
        "not useful": "web_search",                    # 无效:搜索网络
        "useful": END                                  # 有效:结束
    }
)

# 编译图 - 必须编译后才能使用
graph = workflow.compile(checkpointer=checkpointer)
```

**设计亮点:**
- **声明式定义**: 先定义图结构,再编译执行
- **类型安全**: 基于 TypedDict 的状态定义
- **灵活路由**: 支持确定性边和条件边的混合使用

### 1.3 节点实现:状态转换函数

```python
# langgraph/examples/rag/langgraph_adaptive_rag_cohere.ipynb
def retrieve(state):
    """文档检索节点"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """文档相关性评分节点"""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({
            "question": question, 
            "document": d.page_content
        })
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    return {"documents": filtered_docs, "question": question}

def generate(state):
    """答案生成节点"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({
        "documents": documents, 
        "question": question
    })
    return {"generation": generation}
```

**关键特性:**
- **纯函数设计**: 接收 state,返回 state 更新
- **独立可测**: 每个节点可单独测试
- **自动合并**: 返回的状态会自动合并到全局状态

### 1.4 条件路由:状态驱动决策

```python
# 路由函数示例
def route_question(state) -> str:
    """根据问题类型路由到不同数据源"""
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    
    # 基于 LLM 决策选择数据源
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    
    if datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state) -> str:
    """决定是否生成答案或重新搜索"""
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        print("---DECISION: NO RELEVANT DOCUMENTS, WEB SEARCH---")
        return "web_search"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation(state) -> str:
    """评估生成质量,决定下一步"""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke({
        "documents": documents, 
        "generation": generation
    })
    
    if score.binary_score == "yes":
        print("---GENERATION IS GROUNDED---")
        answer_score = answer_grader.invoke({
            "question": question, 
            "generation": generation
        })
        if answer_score.binary_score == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        print("---GENERATION NOT GROUNDED, RE-TRY---")
        return "not supported"
```

**设计优势:**
- **运行时决策**: 基于完整状态进行智能路由
- **多路径支持**: 一个节点可路由到多个目标
- **循环支持**: 可以回到之前的节点重新处理

### 1.5 检查点与持久化

```python
# langgraph/docs/docs/concepts/persistence.md
from langgraph.checkpoint.memory import InMemorySaver

# 配置检查点保存器
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 执行时指定 thread_id
config = {"configurable": {"thread_id": "task_123"}}
result = graph.invoke({"question": "..."}, config)

# 检查点特性:
# 1. 每个超级步骤自动保存
# 2. 包含完整的状态快照
# 3. 支持从任意检查点恢复
# 4. 实现 HITL (人在回路) 和时间旅行
```

**检查点优势:**
- **状态持久化**: 系统崩溃后可恢复
- **中断与恢复**: 支持人工审核后继续执行
- **历史回溯**: 可查看任意时刻的状态
- **分布式支持**: 可使用 Redis/Postgres 作为后端

### 1.6 完整工作流示例

```python
# 自适应 RAG 工作流
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(GraphState)

# 定义节点
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("llm_fallback", llm_fallback)

# 入口路由 - 决定使用哪个数据源
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback"
    }
)

# 文档评分后的路由
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

# 生成质量评估
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "not supported": "generate",      # 循环:重新生成
        "not useful": "web_search",       # 回退:网络搜索
        "useful": END                     # 成功:结束
    }
)

workflow.add_edge("llm_fallback", END)

# 编译并执行
graph = workflow.compile()
result = graph.invoke({"question": "What is LangGraph?"})
```

**工作流特性:**
- **自适应路由**: 根据内容质量动态选择路径
- **质量保证**: 内置生成质量检查和重试
- **多数据源**: 支持向量数据库、网络搜索、直接 LLM
- **循环优化**: 自动重试直到生成满意结果

---

## 二、MathModelAgent:固定流程编排

MathModelAgent 采用 **固定流程模式**,将数学建模专家的经验编码到系统中,确保每次执行都包含所有必要的步骤。

### 2.1 工作流基类设计

```python
# MathModelAgent/backend/app/core/workflow.py
class WorkFlow:
    """工作流基类"""
    def __init__(self):
        pass
    
    def execute(self) -> str:
        """执行工作流"""
        pass

class MathModelWorkFlow(WorkFlow):
    """数学建模工作流"""
    task_id: str
    work_dir: str
    ques_count: int = 0
    questions: dict[str, str | int] = {}
    
    async def execute(self, problem: Problem):
        """执行完整的数学建模流程"""
        # 流程设计理念:
        # 1. 固定顺序执行 - 符合数学建模竞赛规范
        # 2. 专业化分工 - 每个 Agent 负责特定环节
        # 3. 状态传递 - 前序结果作为后续输入
        # 4. 实时反馈 - 通过 Redis 推送进度
```

### 2.2 固定流程实现:顺序执行

```python
# MathModelAgent/backend/app/core/workflow.py
async def execute(self, problem: Problem):
    """数学建模完整流程"""
    self.task_id = problem.task_id
    self.work_dir = create_work_dir(self.task_id)
    
    # === 第一阶段:问题分析 ===
    llm_factory = LLMFactory(self.task_id)
    coordinator_llm, modeler_llm, coder_llm, writer_llm = llm_factory.get_all_llms()
    
    coordinator_agent = CoordinatorAgent(self.task_id, coordinator_llm)
    
    await redis_manager.publish_message(
        self.task_id,
        SystemMessage(content="识别用户意图和拆解问题ing...")
    )
    
    # Step 1: 问题理解与拆解
    coordinator_response = await coordinator_agent.run(problem.ques_all)
    self.questions = coordinator_response.questions
    self.ques_count = coordinator_response.ques_count
    
    await redis_manager.publish_message(
        self.task_id,
        SystemMessage(content="识别完成,任务转交给建模手")
    )
    
    # === 第二阶段:建模设计 ===
    await redis_manager.publish_message(
        self.task_id,
        SystemMessage(content="建模手开始建模ing...")
    )
    
    modeler_agent = ModelerAgent(self.task_id, modeler_llm)
    
    # Step 2: 建模方案设计
    modeler_response = await modeler_agent.run(coordinator_response)
    
    # === 第三阶段:代码执行环境准备 ===
    user_output = UserOutput(work_dir=self.work_dir, ques_count=self.ques_count)
    
    await redis_manager.publish_message(
        self.task_id,
        SystemMessage(content="正在创建代码沙盒环境")
    )
    
    notebook_serializer = NotebookSerializer(work_dir=self.work_dir)
    code_interpreter = await create_interpreter(
        kind="local",
        task_id=self.task_id,
        work_dir=self.work_dir,
        notebook_serializer=notebook_serializer,
        timeout=3000
    )
    
    scholar = OpenAlexScholar(task_id=self.task_id, email=settings.OPENALEX_EMAIL)
    
    # === 第四阶段:初始化执行 Agent ===
    coder_agent = CoderAgent(
        task_id=problem.task_id,
        model=coder_llm,
        work_dir=self.work_dir,
        max_chat_turns=settings.MAX_CHAT_TURNS,
        max_retries=settings.MAX_RETRIES,
        code_interpreter=code_interpreter
    )
    
    writer_agent = WriterAgent(
        task_id=problem.task_id,
        model=writer_llm,
        comp_template=problem.comp_template,
        format_output=problem.format_output,
        scholar=scholar
    )
    
    # === 第五阶段:求解流程 ===
    flows = Flows(self.questions)
    solution_flows = flows.get_solution_flows(self.questions, modeler_response)
    config_template = get_config_template(problem.comp_template)
    
    # 循环执行每个子问题
    for key, value in solution_flows.items():
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=f"代码手开始求解{key}")
        )
        
        # Step 3: 代码执行与求解
        coder_response = await coder_agent.run(
            prompt=value["coder_prompt"],
            subtask_title=key
        )
        
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=f"代码手求解成功{key}", type="success")
        )
        
        writer_prompt = flows.get_writer_prompt(
            key, coder_response.code_response, code_interpreter, config_template
        )
        
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=f"论文手开始写{key}部分")
        )
        
        # Step 4: 论文撰写
        writer_response = await writer_agent.run(
            writer_prompt,
            available_images=coder_response.created_images,
            sub_title=key
        )
        
        user_output.set_res(key, writer_response)
    
    await code_interpreter.cleanup()
    
    # === 第六阶段:论文完整性补充 ===
    write_flows = flows.get_write_flows(user_output, config_template, problem.ques_all)
    
    for key, value in write_flows.items():
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=f"论文手开始写{key}部分")
        )
        
        writer_response = await writer_agent.run(prompt=value, sub_title=key)
        user_output.set_res(key, writer_response)
    
    # 保存最终结果
    user_output.save_result()
```

**固定流程优势:**
- **确定性执行**: 每次都按相同顺序执行
- **专家经验编码**: 流程设计符合数学建模规范
- **易于调试**: 出错时容易定位问题节点
- **状态可追溯**: 每个阶段都有明确的输出

### 2.3 流程配置:Flows 管理

```python
# MathModelAgent/backend/app/core/flows.py
class Flows:
    """流程配置管理"""
    def __init__(self, questions: dict[str, str | int]):
        self.flows: dict[str, dict] = {}
        self.questions: dict[str, str | int] = questions
    
    def get_solution_flows(
        self, questions: dict[str, str | int], modeler_response: ModelerToCoder
    ):
        """获取求解流程配置"""
        # 动态生成子问题流程
        questions_quesx = {
            key: value
            for key, value in questions.items()
            if key.startswith("ques") and key != "ques_count"
        }
        
        ques_flow = {
            key: {
                "coder_prompt": f"""
                    参考建模手给出的解决方案{modeler_response.questions_solution[key]}
                    完成如下问题{value}
                """
            }
            for key, value in questions_quesx.items()
        }
        
        # 固定流程顺序
        flows = {
            "eda": {
                "coder_prompt": f"""
                    参考建模手给出的解决方案{modeler_response.questions_solution["eda"]}
                    对当前目录下数据进行EDA分析(数据清洗,可视化)
                """
            },
            **ques_flow,
            "sensitivity_analysis": {
                "coder_prompt": f"""
                    参考建模手给出的解决方案{modeler_response.questions_solution["sensitivity_analysis"]}
                    完成敏感性分析
                """
            }
        }
        return flows
    
    def get_write_flows(
        self, user_output: UserOutput, config_template: dict, bg_ques_all: str
    ):
        """获取论文撰写流程配置"""
        model_build_solve = user_output.get_model_build_solve()
        
        # 论文各部分按固定顺序生成
        flows = {
            "firstPage": f"""...""",
            "RepeatQues": f"""...""",
            "analysisQues": f"""...""",
            "modelAssumption": f"""...""",
            "symbol": f"""...""",
            "judge": f"""..."""
        }
        return flows
```

**流程管理特性:**
- **配置驱动**: 流程通过配置文件控制
- **动态扩展**: 根据问题数量动态生成流程
- **模板化**: 每个步骤都有固定的提示模板

### 2.4 输出管理:UserOutput 序列化

```python
# MathModelAgent/backend/app/models/user_output.py
class UserOutput:
    """用户输出管理"""
    def __init__(self, work_dir: str, ques_count: int):
        self.work_dir = work_dir
        self.res: dict[str, dict] = {}
        self.ques_count: int = ques_count
        self._init_seq()
    
    def _init_seq(self):
        """初始化输出顺序"""
        ques_str = [f"ques{i}" for i in range(1, self.ques_count + 1)]
        
        # 固定的论文结构顺序
        self.seq = [
            "firstPage",          # 标题、摘要、关键词
            "RepeatQues",         # 一、问题重述
            "analysisQues",       # 二、问题分析
            "modelAssumption",    # 三、模型假设
            "symbol",             # 四、符号说明
            "eda",                # 五、数据预处理
            *ques_str,            # 六、模型建立与求解
            "sensitivity_analysis", # 七、敏感性分析
            "judge"               # 八、模型评价
        ]
    
    def set_res(self, key: str, writer_response: WriterResponse):
        """按顺序保存结果"""
        self.res[key] = {
            "response_content": writer_response.response_content,
            "footnotes": writer_response.footnotes
        }
    
    def save_result(self):
        """按固定顺序保存最终论文"""
        # 按 self.seq 顺序拼接所有部分
        final_paper = ""
        for key in self.seq:
            if key in self.res:
                final_paper += self.res[key]["response_content"]
        
        # 保存到文件
        output_path = os.path.join(self.work_dir, "final_paper.md")
        with open(output_path, "w") as f:
            f.write(final_paper)
```

**设计权衡:**
- **专业化胜过通用化**: 固定流程确保符合竞赛规范
- **可靠性优先**: 确定性流程比灵活路由更可靠
- **领域优化**: 针对数学建模场景深度优化

---

## 三、ADK:多模式工作流支持

ADK 提供了 **SingleFlow**、**SequentialAgent** 和 **ParallelAgent** 三种工作流模式。

### 3.1 SingleFlow:单一智能体工作流

```python
# adk-python/src/google/adk/flows/llm_flows/single_flow.py
class SingleFlow(BaseLlmFlow):
    """单一智能体流程
    
    特点:
    - 只考虑智能体本身和工具
    - 不允许子智能体
    - 适用于简单的工具调用场景
    """
    
    def __init__(self):
        super().__init__()
        # 请求处理器链
        self.request_processors += [
            basic.request_processor,
            auth_preprocessor.request_processor,
            request_confirmation.request_processor,
            instructions.request_processor,
            identity.request_processor,
            contents.request_processor,
            context_cache_processor.request_processor,
            _nl_planning.request_processor,
            _code_execution.request_processor,
            _output_schema_processor.request_processor
        ]
        
        # 响应处理器链
        self.response_processors += [
            _nl_planning.response_processor,
            _code_execution.response_processor
        ]
```

### 3.2 SequentialAgent:顺序执行

```python
# adk-python/src/google/adk/agents/sequential_agent.py
class SequentialAgent(BaseAgent):
    """顺序执行智能体"""
    
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if not self.sub_agents:
            return
        
        # 从状态中恢复执行进度
        agent_state = self._load_agent_state(ctx, SequentialAgentState)
        start_index = self._get_start_index(agent_state)
        
        pause_invocation = False
        resuming_sub_agent = agent_state is not None
        
        # 按顺序执行子智能体
        for i in range(start_index, len(self.sub_agents)):
            sub_agent = self.sub_agents[i]
            
            if not resuming_sub_agent:
                # 保存当前状态
                if ctx.is_resumable:
                    agent_state = SequentialAgentState(
                        current_sub_agent=sub_agent.name
                    )
                    yield self._create_agent_state_event(ctx, agent_state=agent_state)
                
                # 重置子智能体状态
                ctx.reset_agent_state(sub_agent.name)
            
            # 执行子智能体
            async with Aclosing(sub_agent.run_async(ctx)) as agen:
                async for event in agen:
                    yield event
                    if ctx.should_pause_invocation(event):
                        pause_invocation = True
            
            # 如果需要暂停,跳过剩余智能体
            if pause_invocation:
                return
            
            resuming_sub_agent = False
        
        # 标记执行完成
        if ctx.is_resumable:
            yield self._create_agent_state_event(ctx, end_of_agent=True)
```

**关键特性:**
- **状态恢复**: 支持从中断点继续执行
- **顺序保证**: 严格按定义顺序执行
- **暂停支持**: 可在任意节点暂停等待审批

### 3.3 ParallelAgent:并行执行

```python
# adk-python/src/google/adk/agents/parallel_agent.py
async def _merge_agent_run(
    agent_runs: list[AsyncGenerator[Event, None]]
) -> AsyncGenerator[Event, None]:
    """合并多个智能体的事件流
    
    保证:
    - 智能体并行执行
    - 事件按顺序处理
    - 等待事件被消费后再生成新事件
    """
    sentinel = object()
    queue = asyncio.Queue()
    
    async def process_an_agent(events_for_one_agent):
        """处理单个智能体的事件"""
        try:
            async for event in events_for_one_agent:
                resume_signal = asyncio.Event()
                # 将事件放入队列
                await queue.put((event, resume_signal))
                # 等待事件被消费
                await resume_signal.wait()
        finally:
            # 标记智能体完成
            await queue.put((sentinel, None))
    
    async with asyncio.TaskGroup() as tg:
        # 创建并行任务
        for events_for_one_agent in agent_runs:
            tg.create_task(process_an_agent(events_for_one_agent))
        
        sentinel_count = 0
        # 处理所有事件直到所有智能体完成
        while sentinel_count < len(agent_runs):
            event, resume_signal = await queue.get()
            
            if event is sentinel:
                sentinel_count += 1
            else:
                yield event
                # 通知可以继续生成事件
                resume_signal.set()
```

### 3.4 工作流协调示例

```python
# adk-python/contributing/samples/workflow_triage/README.md
# 工作流分流示例

# 执行管理器 Agent
execution_manager = LlmAgent(
    name="execution_manager_agent",
    model="gemini-2.5-flash",
    description="分析用户请求并决定调用哪些执行 Agent",
    tools=[update_execution_plan]
)

# 并行工作 Agent
code_agent = LlmAgent(
    name="code_agent",
    description="处理代码生成任务"
)

math_agent = LlmAgent(
    name="math_agent",
    description="执行数学计算"
)

# 并行执行器
worker_parallel_agent = ParallelAgent(
    name="worker_parallel_agent",
    sub_agents=[code_agent, math_agent]
)

# 总结 Agent
summary_agent = LlmAgent(
    name="execution_summary_agent",
    description="总结所有激活 Agent 的输出"
)

# 顺序编排
plan_execution_agent = SequentialAgent(
    name="plan_execution_agent",
    sub_agents=[
        worker_parallel_agent,   # 并行执行
        summary_agent            # 串行总结
    ]
)

# 执行流程:
# 1. execution_manager 分析请求
# 2. 更新执行计划 (哪些 Agent 需要执行)
# 3. worker_parallel_agent 并行执行相关 Agent
# 4. summary_agent 汇总结果
```

**架构优势:**
- **动态选择**: 根据任务动态激活相关 Agent
- **并行优化**: 无依赖的 Agent 可并行执行
- **层次清晰**: SequentialAgent 包含 ParallelAgent

---

## 四、Vercel AI SDK:循环控制与步骤管理

Vercel AI SDK 提供了灵活的 **循环控制机制** 和 **步骤级定制能力**。

### 4.1 内置循环控制

```typescript
// ai/packages/ai/src/generate-text/generate-text.ts
import { generateText, stepCountIs } from 'ai';

const result = await generateText({
  model: 'openai/gpt-4o',
  prompt: '分析最新销售数据并创建总结报告',
  tools: {
    fetchSalesData: tool({...}),
    analyzeTrends: tool({...}),
    createReport: tool({...})
  },
  // 停止条件 - 最多执行 10 步
  stopWhen: stepCountIs(10),
  
  // 步骤准备 - 每步执行前调用
  prepareStep: ({ stepNumber, steps, messages }) => {
    console.log(`准备执行第 ${stepNumber} 步`);
    
    // 可以动态修改参数
    if (stepNumber > 5) {
      return {
        model: 'openai/gpt-4o-mini',  // 切换到更快的模型
        temperature: 0.3              // 降低随机性
      };
    }
  },
  
  // 步骤完成回调
  onStepFinish: (step) => {
    console.log(`第 ${step.stepNumber} 步完成`);
    console.log(`工具调用:`, step.toolCalls);
    console.log(`工具结果:`, step.toolResults);
    
    // 可以记录到数据库、监控系统等
    saveToDatabase(step);
  }
});
```

**设计亮点:**
- **声明式停止**: 通过 `stopWhen` 定义停止条件
- **动态配置**: `prepareStep` 允许每步使用不同参数
- **可观测性**: `onStepFinish` 提供完整的步骤信息

### 4.2 自定义停止条件

```typescript
// ai/content/docs/03-agents/04-loop-control.mdx
import { StopCondition } from 'ai';

// 自定义停止条件
const stopOnSuccess: StopCondition<Tools> = ({ toolResults }) => {
  // 当任何工具返回 success 时停止
  return toolResults.some(result => result.result.status === 'success');
};

const stopOnMaxCost: StopCondition<Tools> = ({ steps }) => {
  // 计算总成本
  const totalCost = steps.reduce((sum, step) => sum + step.usage.totalTokens, 0);
  return totalCost > 100000;  // 超过 10 万 token 停止
};

// 组合多个条件
const result = await generateText({
  model: 'openai/gpt-4o',
  prompt: '...',
  tools: {...},
  stopWhen: [
    stepCountIs(10),      // 最多 10 步
    stopOnSuccess,        // 或成功完成
    stopOnMaxCost         // 或成本超限
  ]
});
```

### 4.3 手动循环控制

```typescript
// ai/content/docs/03-agents/01-overview.mdx
import { generateText, ModelMessage } from 'ai';

const messages: ModelMessage[] = [
  { role: 'user', content: '帮我分析这个问题并提出解决方案' }
];

let step = 0;
const maxSteps = 10;
const results = [];

while (step < maxSteps) {
  const result = await generateText({
    model: 'openai/gpt-4o',
    messages,
    tools: {
      analyze: tool({...}),
      research: tool({...}),
      propose: tool({...})
    }
  });
  
  // 保存结果
  results.push(result);
  
  // 添加响应到历史
  messages.push(...result.response.messages);
  
  // 自定义停止逻辑
  if (result.text) {
    console.log('模型生成了文本响应,任务完成');
    break;
  }
  
  // 检查是否有特定工具被调用
  const proposeTool = result.toolCalls.find(tc => tc.toolName === 'propose');
  if (proposeTool) {
    console.log('已提出解决方案,任务完成');
    break;
  }
  
  step++;
}

console.log(`共执行 ${step} 步`);
```

**手动控制优势:**
- **完全控制**: 可实现任意复杂的停止逻辑
- **中间干预**: 可在循环中检查和修改状态
- **灵活路由**: 可根据结果动态调整后续步骤

### 4.4 工作流模式:顺序处理

```typescript
// ai/content/docs/03-agents/02-workflows.mdx
// 顺序处理模式 - 固定步骤的流水线

async function contentGenerationPipeline(topic: string) {
  // Step 1: 研究
  const research = await generateText({
    model: 'openai/gpt-4o',
    prompt: `研究主题: ${topic}`,
    tools: { webSearch: searchTool },
    stopWhen: stepCountIs(3)
  });
  
  // Step 2: 大纲
  const outline = await generateText({
    model: 'openai/gpt-4o',
    prompt: `基于研究内容创建大纲:\n${research.text}`,
    stopWhen: stepCountIs(1)
  });
  
  // Step 3: 撰写
  const content = await generateText({
    model: 'openai/gpt-4o',
    prompt: `按照大纲撰写文章:\n${outline.text}`,
    stopWhen: stepCountIs(1)
  });
  
  // Step 4: 审核
  const reviewed = await generateText({
    model: 'openai/gpt-4o',
    prompt: `审核并改进文章:\n${content.text}`,
    stopWhen: stepCountIs(2)
  });
  
  return reviewed.text;
}
```

**顺序模式特点:**
- **明确步骤**: 每个阶段职责清晰
- **数据流**: 前一步的输出是下一步的输入
- **可预测**: 执行路径完全确定

---

## 五、工作流编排模式对比

### 5.1 框架对比表

| 特性 | LangGraph | MathModelAgent | ADK | Vercel AI SDK |
|------|-----------|----------------|-----|---------------|
| **核心理念** | 图形化编排 | 固定流程 | 多模式支持 | 循环控制 |
| **路由方式** | 条件边 + 状态驱动 | 固定顺序 | 配置驱动 | 停止条件 |
| **并行支持** | ✅ 同一超级步骤 | ❌ 纯串行 | ✅ ParallelAgent | ❌ 纯串行 |
| **检查点** | ✅ 自动保存 | ⚠️ Redis 消息 | ✅ 状态恢复 | ❌ 需手动实现 |
| **循环支持** | ✅ 自动循环 | ❌ 固定迭代 | ⚠️ 通过 SubAgent | ✅ 内置循环 |
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **确定性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **学习曲线** | 陡峭 | 平缓 | 中等 | 平缓 |
| **适用场景** | 复杂自适应流程 | 固定领域流程 | 企业级应用 | 通用 Agent |

### 5.2 选择建议

**选择 LangGraph 当你需要:**
- 复杂的条件分支和循环
- 运行时动态路由决策
- 自适应的工作流(如 RAG 质量检查)
- 完整的状态管理和时间旅行

**选择固定流程 (MathModelAgent) 当你需要:**
- 确定性的执行顺序
- 领域专家经验的编码
- 高可靠性和可预测性
- 简单的调试和维护

**选择 ADK 当你需要:**
- 企业级的可靠性
- 多种工作流模式的组合
- Gemini 生态的深度集成
- 配置驱动的流程控制

**选择 Vercel AI SDK 当你需要:**
- 轻量级的循环控制
- 灵活的步骤级定制
- 简单的停止条件管理
- 快速原型开发

---

## 六、工作流编排最佳实践

### 6.1 设计原则

**1. 明确确定性需求**
```python
# 高确定性场景 - 固定流程
if domain_specific and requires_reliability:
    use_fixed_workflow()  # MathModelAgent 模式
else:
    use_dynamic_workflow()  # LangGraph 模式
```

**2. 合理使用并行**
```python
# 并行执行无依赖的任务
parallel_tasks = [
    agent_1.execute(),  # 数据处理
    agent_2.execute(),  # 文献检索
    agent_3.execute()   # 代码生成
]
results = await asyncio.gather(*parallel_tasks)
```

**3. 设置合理的停止条件**
```typescript
// 多重停止条件保护
stopWhen: [
  stepCountIs(10),           // 防止无限循环
  stopOnSuccess,             // 任务完成
  stopOnTimeout(300000),     // 超时保护
  stopOnCost(100000)         // 成本控制
]
```

**4. 实现完善的错误处理**
```python
async def execute_with_retry(node_func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await node_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"节点执行失败,重试 {attempt + 1}/{max_retries}")
            await asyncio.sleep(2 ** attempt)  # 指数退避
```

### 6.2 性能优化

**1. 检查点策略**
```python
# 只在关键节点保存检查点
@checkpoint(when="critical_nodes_only")
def expensive_operation(state):
    # 避免频繁保存降低性能
    pass
```

**2. 状态压缩**
```python
# 定期压缩历史状态
if len(state["messages"]) > 50:
    state["messages"] = compress_messages(state["messages"])
```

**3. 懒加载与缓存**
```python
@lru_cache(maxsize=128)
def load_model(model_name: str):
    # 缓存模型加载
    return load_expensive_model(model_name)
```

### 6.3 可观测性

**1. 结构化日志**
```python
logger.info(
    "工作流执行",
    extra={
        "task_id": task_id,
        "step": step_name,
        "duration_ms": duration,
        "tokens_used": tokens
    }
)
```

**2. 实时进度反馈**
```python
# Redis 发布进度消息
await redis_manager.publish_message(
    task_id,
    SystemMessage(
        content=f"正在执行第 {step}/{total_steps} 步",
        progress=step / total_steps
    )
)
```

**3. 链路追踪**
```python
# OpenTelemetry 追踪
with tracer.start_as_current_span("workflow_execution") as span:
    span.set_attribute("workflow.id", workflow_id)
    span.set_attribute("workflow.type", "math_modeling")
    result = await execute_workflow()
```

---

## 七、总结

工作流编排与控制层是 AI Agent 系统中**确保任务有序执行**的关键层次。不同的编排模式各有优劣:

- **LangGraph** 提供了最强大的图形化编排能力,适合需要复杂条件分支和自适应路由的场景
- **固定流程模式** (如 MathModelAgent) 在特定领域提供了最高的可靠性和确定性
- **ADK** 通过多种 Agent 类型支持灵活的工作流组合
- **Vercel AI SDK** 提供了轻量级的循环控制,适合快速开发

选择合适的工作流模式取决于:
1. **灵活性需求** - 是否需要运行时动态决策
2. **确定性要求** - 是否需要完全可预测的执行路径
3. **复杂度** - 任务的分支和循环复杂度
4. **性能要求** - 是否需要并行执行优化

在实际应用中,可以**混合使用多种模式**:在整体上使用图形化编排提供灵活性,在关键子流程中使用固定流程保证可靠性。这种混合架构能够在灵活性和确定性之间取得最佳平衡。

