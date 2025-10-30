# 多智能体系统实现案例对比

## 概述

本文档通过实现同一个"智能内容创作平台"来展示不同框架的多智能体架构实现差异。该平台包含研究、写作、编辑、发布四个核心功能模块。

## 需求定义

### 系统功能
1. **研究智能体**：收集资料、搜索信息
2. **写作智能体**：生成初稿内容
3. **编辑智能体**：内容审核和修改
4. **发布智能体**：格式化和发布

### 业务流程
```
用户请求 → 研究 → 写作 → 编辑 → 发布 → 最终内容
    ↓        ↓      ↓      ↓      ↓
  需求分析  资料收集  初稿生成  内容优化  格式发布
```

## ADK 实现

### 1. 层次化架构设计

```python
# content_platform_adk.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from typing import List, Dict

# 工具定义
@FunctionTool
def web_search(query: str) -> str:
    """网络搜索工具"""
    # 实现搜索逻辑
    return f"搜索结果：{query}"

@FunctionTool
def generate_content(topic: str, research_data: str) -> str:
    """内容生成工具"""
    # 实现内容生成逻辑
    return f"关于{topic}的内容：{research_data}"

@FunctionTool
def edit_content(content: str) -> str:
    """内容编辑工具"""
    # 实现编辑逻辑
    return f"编辑后的内容：{content}"

@FunctionTool
def publish_content(content: str, platform: str) -> str:
    """内容发布工具"""
    # 实现发布逻辑
    return f"在{platform}发布：{content}"

# 子智能体定义
class ResearchAgent(LlmAgent):
    """研究智能体"""
    name: str = "researcher"
    description: str = "负责收集资料和研究信息"
    instruction: str = """
    你是研究专家，负责：
    1. 理解用户需求
    2. 搜索相关资料
    3. 整理研究数据
    4. 为写作提供素材
    """
    tools: List = [web_search]
    disallow_transfer_to_parent: bool = False

class WritingAgent(LlmAgent):
    """写作智能体"""
    name: str = "writer"
    description: str = "负责生成初稿内容"
    instruction: str = """
    你是写作专家，负责：
    1. 基于研究资料生成初稿
    2. 确保内容结构清晰
    3. 保持语言流畅自然
    4. 符合内容要求
    """
    tools: List = [generate_content]

class EditingAgent(LlmAgent):
    """编辑智能体"""
    name: str = "editor"
    description: str = "负责内容审核和修改"
    instruction: str = """
    你是编辑专家，负责：
    1. 审核内容质量
    2. 修改语法和表达
    3. 优化内容结构
    4. 确保发布标准
    """
    tools: List = [edit_content]

class PublishingAgent(LlmAgent):
    """发布智能体"""
    name: str = "publisher"
    description: str = "负责格式化和发布内容"
    instruction: str = """
    你是发布专家，负责：
    1. 格式化最终内容
    2. 选择合适的发布平台
    3. 执行发布操作
    4. 记录发布结果
    """
    tools: List = [publish_content]
    disallow_transfer_to_parent: bool = True  # 终端节点

# 根协调者
class ContentPlatformCoordinator(LlmAgent):
    """内容平台协调者"""
    name: str = "content_coordinator"
    description: str = "智能内容创作平台协调者"
    instruction: str = """
    你是内容创作平台的协调者，负责：
    1. 分析用户的内容创作需求
    2. 协调各个专业智能体
    3. 监控创作流程进度
    4. 确保内容质量标准

    工作流程：
    - 研究需求 → researcher智能体
    - 内容创作 → writer智能体
    - 内容编辑 → editor智能体
    - 内容发布 → publisher智能体
    """
    sub_agents: List = []  # 将在初始化时设置

def create_content_platform():
    """创建内容平台系统"""

    # 创建子智能体
    researcher = ResearchAgent(model="gemini-2.0-flash")
    writer = WritingAgent(model="gemini-2.0-flash")
    editor = EditingAgent(model="gemini-2.0-flash")
    publisher = PublishingAgent(model="gemini-2.0-flash")

    # 创建协调者
    coordinator = ContentPlatformCoordinator(
        model="gemini-2.0-flash",
        sub_agents=[researcher, writer, editor, publisher]
    )

    return coordinator

# 使用示例
async def create_content_with_adk(user_request: str):
    """使用ADK创建内容"""
    platform = create_content_platform()

    # 执行内容创作流程
    result = await platform.run(user_request)

    return result
```

### 2. 配置文件方式

```yaml
# content_platform_config.yaml
agent_class: LlmAgent
name: content_coordinator
model: gemini-2.0-flash
description: 智能内容创作平台协调者
instruction: |
  你是内容创作平台的协调者，负责分析用户需求并协调各个专业智能体。

sub_agents:
  - config_path: researcher_agent.yaml
  - config_path: writer_agent.yaml
  - config_path: editor_agent.yaml
  - config_path: publisher_agent.yaml

---
# researcher_agent.yaml
agent_class: LlmAgent
name: researcher
model: gemini-2.0-flash
description: 负责收集资料和研究信息
instruction: |
  你是研究专家，负责收集资料和研究信息。
tools:
  - web_search

---
# writer_agent.yaml
agent_class: LlmAgent
name: writer
model: gemini-2.0-flash
description: 负责生成初稿内容
instruction: |
  你是写作专家，负责基于研究资料生成初稿。
tools:
  - generate_content
```

## LangGraph 实现

### 1. 图形化架构设计

```python
# content_platform_langgraph.py
from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.tools import tool

# 状态定义
class ContentState(TypedDict):
    user_request: str
    research_data: str
    draft_content: str
    edited_content: str
    final_content: str
    platform: str
    messages: List[dict]
    current_step: str

# 工具定义
@tool
def web_search(query: str) -> str:
    """网络搜索工具"""
    return f"搜索结果：{query}"

@tool
def generate_content(topic: str, research_data: str) -> str:
    """内容生成工具"""
    return f"基于研究数据生成的内容：{research_data}"

@tool
def edit_content(content: str) -> str:
    """内容编辑工具"""
    return f"编辑后的内容：{content}"

@tool
def publish_content(content: str, platform: str) -> str:
    """内容发布工具"""
    return f"在{platform}发布：{content}"

# 智能体节点定义
def researcher_node(state: ContentState) -> Command[Literal["writer"]]:
    """研究智能体节点"""

    model = ChatOpenAI(model="gpt-4")
    model_with_tools = model.bind_tools([web_search])

    # 执行研究
    response = model_with_tools.invoke([
        {"role": "system", "content": "你是研究专家，负责收集资料。"},
        {"role": "user", "content": f"研究主题：{state['user_request']}"}
    ])

    # 处理工具调用
    if response.tool_calls:
        research_data = web_search.invoke(response.tool_calls[0]['args']['query'])
    else:
        research_data = response.content

    return Command(
        goto="writer",
        update={
            "research_data": research_data,
            "messages": state["messages"] + [{"role": "assistant", "content": f"研究完成：{research_data}"}]
        }
    )

def writer_node(state: ContentState) -> Command[Literal["editor"]]:
    """写作智能体节点"""

    model = ChatOpenAI(model="gpt-4")
    model_with_tools = model.bind_tools([generate_content])

    # 执行写作
    response = model_with_tools.invoke([
        {"role": "system", "content": "你是写作专家，负责生成初稿。"},
        {"role": "user", "content": f"基于研究数据写作：{state['research_data']}"}
    ])

    # 处理工具调用
    if response.tool_calls:
        draft_content = generate_content.invoke(
            topic=state['user_request'],
            research_data=state['research_data']
        )
    else:
        draft_content = response.content

    return Command(
        goto="editor",
        update={
            "draft_content": draft_content,
            "messages": state["messages"] + [{"role": "assistant", "content": f"初稿完成：{draft_content}"}]
        }
    )

def editor_node(state: ContentState) -> Command[Literal["publisher"]]:
    """编辑智能体节点"""

    model = ChatOpenAI(model="gpt-4")
    model_with_tools = model.bind_tools([edit_content])

    # 执行编辑
    response = model_with_tools.invoke([
        {"role": "system", "content": "你是编辑专家，负责内容审核和修改。"},
        {"role": "user", "content": f"编辑内容：{state['draft_content']}"}
    ])

    # 处理工具调用
    if response.tool_calls:
        edited_content = edit_content.invoke(response.tool_calls[0]['args']['content'])
    else:
        edited_content = response.content

    return Command(
        goto="publisher",
        update={
            "edited_content": edited_content,
            "messages": state["messages"] + [{"role": "assistant", "content": f"编辑完成：{edited_content}"}]
        }
    )

def publisher_node(state: ContentState) -> Command[Literal[END]]:
    """发布智能体节点"""

    model = ChatOpenAI(model="gpt-4")
    model_with_tools = model.bind_tools([publish_content])

    # 执行发布
    response = model_with_tools.invoke([
        {"role": "system", "content": "你是发布专家，负责格式化和发布。"},
        {"role": "user", "content": f"发布内容：{state['edited_content']}"}
    ])

    # 处理工具调用
    if response.tool_calls:
        final_content = publish_content.invoke(
            content=state['edited_content'],
            platform="blog"
        )
    else:
        final_content = response.content

    return Command(
        goto=END,
        update={
            "final_content": final_content,
            "messages": state["messages"] + [{"role": "assistant", "content": f"发布完成：{final_content}"}]
        }
    )

# 构建图
def create_content_graph():
    """创建内容创作图"""

    builder = StateGraph(ContentState)

    # 添加节点
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("editor", editor_node)
    builder.add_node("publisher", publisher_node)

    # 添加边
    builder.add_edge(START, "researcher")

    # 编译图
    return builder.compile()

# 使用示例
async def create_content_with_langgraph(user_request: str):
    """使用LangGraph创建内容"""

    graph = create_content_graph()

    # 初始状态
    initial_state = {
        "user_request": user_request,
        "research_data": "",
        "draft_content": "",
        "edited_content": "",
        "final_content": "",
        "platform": "blog",
        "messages": [{"role": "user", "content": user_request}],
        "current_step": "research"
    }

    # 执行图
    result = await graph.ainvoke(initial_state)

    return result["final_content"]
```

### 2. 动态路由版本

```python
# content_platform_dynamic.py
def supervisor_router(state: ContentState) -> Literal["researcher", "writer", "editor", "publisher", END]:
    """动态路由决策"""

    current_step = state.get("current_step", "research")

    if current_step == "research":
        return "researcher"
    elif current_step == "write":
        return "writer"
    elif current_step == "edit":
        return "editor"
    elif current_step == "publish":
        return "publisher"
    else:
        return END

def dynamic_researcher(state: ContentState) -> Command[Literal["supervisor"]]:
    """动态研究节点"""

    # 执行研究逻辑
    research_data = web_search(state["user_request"])

    return Command(
        goto="supervisor",
        update={
            "research_data": research_data,
            "current_step": "write"
        }
    )

def create_dynamic_content_graph():
    """创建动态内容创作图"""

    builder = StateGraph(ContentState)

    # 添加节点
    builder.add_node("supervisor", supervisor_router)
    builder.add_node("researcher", dynamic_researcher)
    builder.add_node("writer", dynamic_writer)
    builder.add_node("editor", dynamic_editor)
    builder.add_node("publisher", dynamic_publisher)

    # 添加边
    builder.add_edge(START, "supervisor")
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("editor", "supervisor")
    builder.add_edge("publisher", "supervisor")

    return builder.compile()
```

## OpenAI Agents SDK 实现

### 1. 简洁Handoff架构

```python
# content_platform_openai.py
from agents import Agent, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# 工具定义
async def web_search(query: str) -> str:
    """网络搜索工具"""
    return f"搜索结果：{query}"

async def generate_content(topic: str, research_data: str) -> str:
    """内容生成工具"""
    return f"基于研究数据生成的内容：{research_data}"

async def edit_content(content: str) -> str:
    """内容编辑工具"""
    return f"编辑后的内容：{content}"

async def publish_content(content: str, platform: str) -> str:
    """内容发布工具"""
    return f"在{platform}发布：{content}"

# 智能体定义
researcher = Agent(
    name="Researcher",
    instructions=f"""
    {RECOMMENDED_PROMPT_PREFIX}
    你是研究专家，负责：
    1. 理解用户的内容需求
    2. 搜索相关资料和信息
    3. 整理研究数据
    4. 为写作提供充分的素材

    完成研究后，将结果传递给Writer进行内容创作。
    """,
    tools=[web_search],
    handoffs=[]  # 将在triage中设置
)

writer = Agent(
    name="Writer",
    instructions=f"""
    {RECOMMENDED_PROMPT_PREFIX}
    你是写作专家，负责：
    1. 基于研究资料生成高质量初稿
    2. 确保内容结构清晰、逻辑连贯
    3. 保持语言流畅自然
    4. 符合用户的内容要求

    完成写作后，将初稿传递给Editor进行审核编辑。
    """,
    tools=[generate_content],
    handoffs=[]
)

editor = Agent(
    name="Editor",
    instructions=f"""
    {RECOMMENDED_PROMPT_PREFIX}
    你是编辑专家，负责：
    1. 审核内容质量和准确性
    2. 修改语法错误和表达不当
    3. 优化内容结构和逻辑
    4. 确保内容达到发布标准

    完成编辑后，将最终内容传递给Publisher进行发布。
    """,
    tools=[edit_content],
    handoffs=[]
)

publisher = Agent(
    name="Publisher",
    instructions=f"""
    {RECOMMENDED_PROMPT_PREFIX}
    你是发布专家，负责：
    1. 格式化最终内容
    2. 选择合适的发布平台
    3. 执行发布操作
    4. 记录发布结果和反馈

    这是流程的最后一步，完成后直接向用户返回最终结果。
    """,
    tools=[publish_content],
    handoffs=[]  # 终端节点
)

# 设置Handoff关系
researcher.handoffs = [handoff(writer)]
writer.handoffs = [handoff(editor)]
editor.handoffs = [handoff(publisher)]

# 分诊智能体
triage_agent = Agent(
    name="Content Triage",
    instructions=f"""
    {RECOMMENDED_PROMPT_PREFIX}
    你是内容创作平台的分诊专家，负责：
    1. 分析用户的内容创作需求
    2. 确定内容类型和复杂度
    3. 将任务分配给合适的专业智能体

    分配策略：
    - 所有内容创作任务都从Researcher开始
    - Researcher负责研究，Writer负责写作，Editor负责编辑，Publisher负责发布
    """,
    handoffs=[researcher]
)

# 使用示例
async def create_content_with_openai(user_request: str):
    """使用OpenAI Agents SDK创建内容"""

    result = await triage_agent.run(user_request)

    return result
```

### 2. 自定义Handoff版本

```python
# content_platform_custom_handoff.py
from agents import Agent, handoff, RunContextWrapper
from pydantic import BaseModel

class HandoffData(BaseModel):
    """Handoff数据结构"""
    step: str
    content: str
    metadata: dict = {}

def create_handoff_with_callback(target_agent: Agent, step_name: str):
    """创建带回调的Handoff"""

    async def on_handoff(ctx: RunContextWrapper, input_data: HandoffData):
        """Handoff回调函数"""
        print(f"🔄 转移到 {target_agent.name} - 步骤: {step_name}")
        print(f"📝 内容预览: {input_data.content[:100]}...")

        # 可以在这里添加额外的处理逻辑
        # 如：记录日志、发送通知等

    return handoff(
        agent=target_agent,
        on_handoff=on_handoff,
        input_type=HandoffData,
        tool_name_override=f"transfer_to_{step_name}",
        tool_description_override=f"转移到{target_agent.name}进行{step_name}"
    )

# 重新定义智能体
researcher = Agent(
    name="Researcher",
    instructions="你是研究专家，负责收集资料。",
    tools=[web_search],
    handoffs=[create_handoff_with_callback(writer, "content_writing")]
)

writer = Agent(
    name="Writer",
    instructions="你是写作专家，负责生成初稿。",
    tools=[generate_content],
    handoffs=[create_handoff_with_callback(editor, "content_editing")]
)

editor = Agent(
    name="Editor",
    instructions="你是编辑专家，负责内容审核。",
    tools=[edit_content],
    handoffs=[create_handoff_with_callback(publisher, "content_publishing")]
)

publisher = Agent(
    name="Publisher",
    instructions="你是发布专家，负责格式化和发布。",
    tools=[publish_content],
    handoffs=[]
)
```

## MathModelAgent 风格实现

### 1. 线性流水线架构

```python
# content_platform_mathmodel.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import asyncio

# 基础智能体类
class BaseAgent(ABC):
    """基础智能体类"""

    def __init__(self, name: str, model_config: Dict[str, Any]):
        self.name = name
        self.model_config = model_config
        self.chat_history = []

    @abstractmethod
    async def run(self, input_data: Any) -> Any:
        """执行智能体任务"""
        pass

    async def append_chat_history(self, message: Dict[str, str]):
        """添加聊天历史"""
        self.chat_history.append(message)

        # 内存管理
        if len(self.chat_history) > 50:
            # 简单截断，实际可以使用智能压缩
            self.chat_history = self.chat_history[-30:]

# 数据传递模型
from pydantic import BaseModel

class ResearchToWriter(BaseModel):
    """研究到写作的数据传递"""
    research_data: str
    sources: list[str]
    confidence_score: float

class WriterToEditor(BaseModel):
    """写作到编辑的数据传递"""
    draft_content: str
    word_count: int
    content_type: str

class EditorToPublisher(BaseModel):
    """编辑到发布的数据传递"""
    edited_content: str
    quality_score: float
    edit_summary: str

class PublisherResult(BaseModel):
    """发布结果"""
    final_content: str
    platform: str
    publish_url: str
    timestamp: str

# 专业化智能体实现
class ResearcherAgent(BaseAgent):
    """研究智能体"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__("Researcher", model_config)

    async def run(self, user_request: str) -> ResearchToWriter:
        """执行研究任务"""

        await self.append_chat_history({
            "role": "user",
            "content": f"研究主题：{user_request}"
        })

        # 模拟研究过程
        research_data = await self._perform_research(user_request)
        sources = await self._find_sources(user_request)
        confidence_score = await self._assess_confidence(research_data)

        await self.append_chat_history({
            "role": "assistant",
            "content": f"研究完成，找到{len(sources)}个资料源"
        })

        return ResearchToWriter(
            research_data=research_data,
            sources=sources,
            confidence_score=confidence_score
        )

    async def _perform_research(self, topic: str) -> str:
        """执行研究"""
        # 模拟网络搜索
        await asyncio.sleep(1)  # 模拟网络延迟
        return f"关于{topic}的详细研究数据..."

    async def _find_sources(self, topic: str) -> list[str]:
        """查找资料源"""
        return [f"资料源{i}" for i in range(1, 6)]

    async def _assess_confidence(self, data: str) -> float:
        """评估置信度"""
        return 0.85

class WriterAgent(BaseAgent):
    """写作智能体"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__("Writer", model_config)

    async def run(self, research_data: ResearchToWriter) -> WriterToEditor:
        """执行写作任务"""

        await self.append_chat_history({
            "role": "user",
            "content": f"基于研究数据写作：{research_data.research_data[:200]}..."
        })

        # 模拟写作过程
        draft_content = await self._generate_draft(research_data)
        word_count = len(draft_content.split())
        content_type = await self._determine_content_type(draft_content)

        await self.append_chat_history({
            "role": "assistant",
            "content": f"初稿完成，字数：{word_count}"
        })

        return WriterToEditor(
            draft_content=draft_content,
            word_count=word_count,
            content_type=content_type
        )

    async def _generate_draft(self, research_data: ResearchToWriter) -> str:
        """生成初稿"""
        await asyncio.sleep(2)  # 模拟写作时间
        return f"基于研究数据生成的完整文章内容：{research_data.research_data}"

    async def _determine_content_type(self, content: str) -> str:
        """确定内容类型"""
        if "技术" in content:
            return "技术文章"
        elif "新闻" in content:
            return "新闻报道"
        else:
            return "通用文章"

class EditorAgent(BaseAgent):
    """编辑智能体"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__("Editor", model_config)

    async def run(self, writer_data: WriterToEditor) -> EditorToPublisher:
        """执行编辑任务"""

        await self.append_chat_history({
            "role": "user",
            "content": f"编辑内容：{writer_data.draft_content[:200]}..."
        })

        # 模拟编辑过程
        edited_content = await self._edit_content(writer_data.draft_content)
        quality_score = await self._assess_quality(edited_content)
        edit_summary = await self._generate_edit_summary(writer_data.draft_content, edited_content)

        await self.append_chat_history({
            "role": "assistant",
            "content": f"编辑完成，质量评分：{quality_score}"
        })

        return EditorToPublisher(
            edited_content=edited_content,
            quality_score=quality_score,
            edit_summary=edit_summary
        )

    async def _edit_content(self, content: str) -> str:
        """编辑内容"""
        await asyncio.sleep(1.5)  # 模拟编辑时间
        return f"编辑优化后的内容：{content}"

    async def _assess_quality(self, content: str) -> float:
        """评估内容质量"""
        return 0.92

    async def _generate_edit_summary(self, original: str, edited: str) -> str:
        """生成编辑摘要"""
        return "修正了语法错误，优化了段落结构，提升了可读性"

class PublisherAgent(BaseAgent):
    """发布智能体"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__("Publisher", model_config)

    async def run(self, editor_data: EditorToPublisher) -> PublisherResult:
        """执行发布任务"""

        await self.append_chat_history({
            "role": "user",
            "content": f"发布内容：{editor_data.edited_content[:200]}..."
        })

        # 模拟发布过程
        final_content = await self._format_content(editor_data.edited_content)
        platform = await self._select_platform(editor_data.edited_content)
        publish_url = await self._publish_content(final_content, platform)
        timestamp = await self._get_timestamp()

        await self.append_chat_history({
            "role": "assistant",
            "content": f"发布完成，平台：{platform}"
        })

        return PublisherResult(
            final_content=final_content,
            platform=platform,
            publish_url=publish_url,
            timestamp=timestamp
        )

    async def _format_content(self, content: str) -> str:
        """格式化内容"""
        await asyncio.sleep(0.5)  # 模拟格式化时间
        return f"格式化后的内容：\n\n{content}"

    async def _select_platform(self, content: str) -> str:
        """选择发布平台"""
        if "技术" in content:
            return "GitHub"
        elif "新闻" in content:
            return "Twitter"
        else:
            return "Blog"

    async def _publish_content(self, content: str, platform: str) -> str:
        """发布内容"""
        await asyncio.sleep(1)  # 模拟发布时间
        return f"https://{platform}.example.com/content/{hash(content)}"

    async def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

# 工作流编排
class ContentPlatformWorkflow:
    """内容平台工作流"""

    def __init__(self, model_configs: Dict[str, Dict[str, Any]]):
        self.model_configs = model_configs
        self.agents = self._create_agents()

    def _create_agents(self) -> Dict[str, BaseAgent]:
        """创建智能体"""
        return {
            "researcher": ResearcherAgent(self.model_configs["researcher"]),
            "writer": WriterAgent(self.model_configs["writer"]),
            "editor": EditorAgent(self.model_configs["editor"]),
            "publisher": PublisherAgent(self.model_configs["publisher"])
        }

    async def execute(self, user_request: str) -> PublisherResult:
        """执行完整工作流"""

        print(f"🚀 开始内容创作流程：{user_request}")

        try:
            # 1. 研究阶段
            print("📚 执行研究...")
            research_result = await self.agents["researcher"].run(user_request)

            # 2. 写作阶段
            print("✍️ 执行写作...")
            writer_result = await self.agents["writer"].run(research_result)

            # 3. 编辑阶段
            print("📝 执行编辑...")
            editor_result = await self.agents["editor"].run(writer_result)

            # 4. 发布阶段
            print("🌐 执行发布...")
            publisher_result = await self.agents["publisher"].run(editor_result)

            print("✅ 内容创作完成！")
            return publisher_result

        except Exception as e:
            print(f"❌ 工作流执行失败：{e}")
            raise

# 使用示例
async def create_content_with_mathmodel(user_request: str):
    """使用MathModelAgent风格创建内容"""

    # 模型配置
    model_configs = {
        "researcher": {"model": "gpt-4o-mini", "temperature": 0.1},
        "writer": {"model": "gpt-4", "temperature": 0.7},
        "editor": {"model": "gpt-4", "temperature": 0.3},
        "publisher": {"model": "gpt-4o-mini", "temperature": 0.1}
    }

    # 创建工作流
    workflow = ContentPlatformWorkflow(model_configs)

    # 执行工作流
    result = await workflow.execute(user_request)

    return result

# 测试代码
async def test_all_implementations():
    """测试所有实现"""

    user_request = "写一篇关于人工智能发展趋势的文章"

    print("=" * 60)
    print("测试不同框架的多智能体实现")
    print("=" * 60)

    # 测试ADK实现
    print("\n🔹 ADK实现：")
    try:
        adk_result = await create_content_with_adk(user_request)
        print(f"ADK结果：{adk_result}")
    except Exception as e:
        print(f"ADK错误：{e}")

    # 测试LangGraph实现
    print("\n🔹 LangGraph实现：")
    try:
        langgraph_result = await create_content_with_langgraph(user_request)
        print(f"LangGraph结果：{langgraph_result}")
    except Exception as e:
        print(f"LangGraph错误：{e}")

    # 测试OpenAI Agents SDK实现
    print("\n🔹 OpenAI Agents SDK实现：")
    try:
        openai_result = await create_content_with_openai(user_request)
        print(f"OpenAI结果：{openai_result}")
    except Exception as e:
        print(f"OpenAI错误：{e}")

    # 测试MathModelAgent风格实现
    print("\n🔹 MathModelAgent风格实现：")
    try:
        mathmodel_result = await create_content_with_mathmodel(user_request)
        print(f"MathModel结果：{mathmodel_result}")
    except Exception as e:
        print(f"MathModel错误：{e}")

if __name__ == "__main__":
    asyncio.run(test_all_implementations())
```

## 实现对比总结

### 代码复杂度对比

| 框架 | 代码行数 | 配置复杂度 | 学习曲线 | 调试难度 |
|------|----------|------------|----------|----------|
| **ADK** | ~200行 | 中等 | 中等 | 中等 |
| **LangGraph** | ~250行 | 高 | 陡峭 | 高 |
| **OpenAI SDK** | ~150行 | 低 | 平缓 | 低 |
| **MathModelAgent** | ~300行 | 低 | 中等 | 低 |

### 功能特性对比

| 特性 | ADK | LangGraph | OpenAI SDK | MathModelAgent |
|------|-----|-----------|-------------|----------------|
| **动态路由** | ✅ LLM驱动 | ✅ 图路由 | ❌ 固定Handoff | ❌ 固定顺序 |
| **状态管理** | ✅ 继承式 | ✅ 检查点 | ✅ 会话管理 | ✅ 传递式 |
| **并行执行** | ❌ 顺序 | ✅ 图并行 | ❌ 顺序 | ❌ 顺序 |
| **错误处理** | ✅ 分层 | ✅ 图级 | ✅ 简单 | ✅ 重试 |
| **监控追踪** | ✅ 企业级 | ✅ 图可视化 | ❌ 基础 | ✅ 自定义 |
| **配置方式** | ✅ YAML+代码 | ✅ 纯代码 | ✅ 纯代码 | ✅ 纯代码 |

### 适用场景推荐

1. **企业级应用** → ADK
   - 需要权限控制和审计
   - 符合企业组织结构
   - 长期维护需求

2. **复杂决策系统** → LangGraph
   - 需要动态路由
   - 复杂的业务逻辑
   - 可视化流程需求

3. **快速原型开发** → OpenAI Agents SDK
   - 快速验证概念
   - 团队技能有限
   - 简单协作场景

4. **专业领域应用** → MathModelAgent风格
   - 固定流程优化
   - 专业化要求高
   - 结果一致性重要

通过这个详细的实现对比，开发者可以根据具体需求选择最适合的框架和实现方式。
