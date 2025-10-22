# LangGraph 多智能体协作 Demo

## 演示场景：智能内容创作平台

这个demo展示如何使用LangGraph构建一个多智能体协作的内容创作系统，包含研究、写作、审核、发布等环节。

## 完整代码实现

### 1. 环境设置和依赖

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Literal, Annotated
from typing_extensions import TypedDict
import sqlite3
from datetime import datetime

# 初始化模型和检查点
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
checkpointer = SqliteSaver("content_creation.db")
```

### 2. 状态定义

```python
class ContentCreationState(TypedDict):
    """内容创作工作流状态"""
    # 消息历史
    messages: Annotated[list, "消息列表"]

    # 内容创作流程状态
    topic: str                    # 创作主题
    research_notes: str           # 研究笔记
    outline: str                  # 内容大纲
    draft_content: str            # 初稿内容
    reviewed_content: str         # 审核后内容
    final_content: str            # 最终内容

    # 流程控制
    current_stage: str            # 当前阶段
    quality_score: float          # 质量评分
    needs_human_review: bool      # 是否需要人工审核
    approval_status: str          # 审批状态

    # 协作信息
    assigned_agent: str           # 当前负责的智能体
    collaboration_notes: list[str] # 协作备注
```

### 3. 工具定义

```python
from langchain_core.tools import tool

@tool
def research_web(query: str) -> str:
    """搜索网络获取研究资料"""
    # 模拟网络搜索
    return f"研究结果：关于'{query}'的最新信息和数据..."

@tool
def check_plagiarism(content: str) -> dict:
    """检查内容原创性"""
    # 模拟查重检查
    similarity_score = 0.15  # 15%相似度
    return {
        "similarity_score": similarity_score,
        "is_original": similarity_score < 0.3,
        "sources": ["source1.com", "source2.com"]
    }

@tool
def save_draft(content: str, version: str) -> str:
    """保存草稿版本"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"draft_{version}_{timestamp}.md"
    # 模拟保存文件
    return f"草稿已保存: {filename}"

@tool
def notify_human_reviewer(content: str, stage: str) -> str:
    """通知人工审核员"""
    return f"已通知人工审核员进行{stage}阶段审核"
```

### 4. 智能体节点定义

```python
def researcher_agent(state: ContentCreationState) -> ContentCreationState:
    """研究员智能体 - 负责收集资料"""

    messages = state["messages"]
    topic = state["topic"]

    # 执行研究任务
    research_prompt = f"""
    你是一个专业的研究员。请针对主题 "{topic}" 进行深入研究。

    任务要求:
    1. 收集相关的背景信息和数据
    2. 分析当前趋势和观点
    3. 找出关键要点和争议点
    4. 提供研究摘要和建议

    请提供详细的研究报告。
    """

    # 调用LLM进行研究
    response = llm.invoke([HumanMessage(content=research_prompt)])
    research_notes = response.content

    # 使用工具获取额外信息
    web_research = research_web(topic)

    return {
        **state,
        "research_notes": f"{research_notes}\n\n补充研究:\n{web_research}",
        "current_stage": "research_completed",
        "assigned_agent": "writer",
        "messages": messages + [
            HumanMessage(content=research_prompt),
            AIMessage(content=f"研究完成: {research_notes[:200]}...")
        ]
    }

def writer_agent(state: ContentCreationState) -> ContentCreationState:
    """写作智能体 - 负责内容创作"""

    messages = state["messages"]
    topic = state["topic"]
    research_notes = state["research_notes"]

    # 创建大纲
    outline_prompt = f"""
    基于以下研究资料，为主题 "{topic}" 创建详细的内容大纲:

    研究资料:
    {research_notes}

    请创建一个逻辑清晰、结构完整的大纲。
    """

    outline_response = llm.invoke([HumanMessage(content=outline_prompt)])
    outline = outline_response.content

    # 撰写初稿
    draft_prompt = f"""
    基于以下大纲撰写完整的文章:

    主题: {topic}
    大纲: {outline}
    研究资料: {research_notes}

    要求:
    1. 内容丰富、逻辑清晰
    2. 语言流畅、表达准确
    3. 包含具体案例和数据
    4. 字数控制在1000-1500字
    """

    draft_response = llm.invoke([HumanMessage(content=draft_prompt)])
    draft_content = draft_response.content

    # 保存草稿
    save_result = save_draft(draft_content, "v1")

    return {
        **state,
        "outline": outline,
        "draft_content": draft_content,
        "current_stage": "draft_completed",
        "assigned_agent": "reviewer",
        "messages": messages + [
            AIMessage(content=f"写作完成，已生成大纲和初稿。{save_result}")
        ]
    }

def reviewer_agent(state: ContentCreationState) -> ContentCreationState:
    """审核智能体 - 负责质量把控"""

    messages = state["messages"]
    draft_content = state["draft_content"]
    topic = state["topic"]

    # 质量评估
    review_prompt = f"""
    请对以下文章进行全面审核和评估:

    主题: {topic}
    文章内容: {draft_content}

    评估维度:
    1. 内容准确性和完整性 (1-10分)
    2. 逻辑结构和条理性 (1-10分)
    3. 语言表达和流畅度 (1-10分)
    4. 创新性和吸引力 (1-10分)

    请提供:
    - 总体评分 (1-10分)
    - 具体修改建议
    - 是否需要人工审核 (评分<7分需要)
    """

    review_response = llm.invoke([HumanMessage(content=review_prompt)])
    review_result = review_response.content

    # 检查原创性
    plagiarism_check = check_plagiarism(draft_content)

    # 解析评分 (简化处理)
    quality_score = 7.5  # 模拟评分
    needs_review = quality_score < 7.0 or not plagiarism_check["is_original"]

    if needs_review:
        notify_result = notify_human_reviewer(draft_content, "quality_review")
        approval_status = "pending_human_review"
    else:
        approval_status = "auto_approved"

    return {
        **state,
        "reviewed_content": draft_content if not needs_review else "",
        "quality_score": quality_score,
        "needs_human_review": needs_review,
        "approval_status": approval_status,
        "current_stage": "review_completed",
        "assigned_agent": "publisher" if not needs_review else "human_reviewer",
        "collaboration_notes": state["collaboration_notes"] + [
            f"审核完成，评分: {quality_score}, 原创性: {plagiarism_check['is_original']}"
        ],
        "messages": messages + [
            AIMessage(content=f"审核完成: {review_result[:200]}...")
        ]
    }

def human_review_node(state: ContentCreationState) -> ContentCreationState:
    """人工审核节点 - 处理需要人工干预的情况"""

    print("🔔 需要人工审核!")
    print(f"📄 内容主题: {state['topic']}")
    print(f"📊 质量评分: {state['quality_score']}")
    print(f"📝 当前状态: {state['approval_status']}")
    print("\n请选择操作:")
    print("1. 批准发布")
    print("2. 要求修改")
    print("3. 拒绝发布")

    # 在实际应用中，这里会是Web界面或API调用
    # 这里模拟用户选择
    human_decision = "1"  # 模拟批准

    if human_decision == "1":
        return {
            **state,
            "approval_status": "human_approved",
            "final_content": state["draft_content"],
            "assigned_agent": "publisher",
            "collaboration_notes": state["collaboration_notes"] + [
                "人工审核通过，批准发布"
            ]
        }
    elif human_decision == "2":
        return {
            **state,
            "approval_status": "revision_required",
            "assigned_agent": "writer",
            "collaboration_notes": state["collaboration_notes"] + [
                "人工审核要求修改"
            ]
        }
    else:
        return {
            **state,
            "approval_status": "rejected",
            "assigned_agent": "end",
            "collaboration_notes": state["collaboration_notes"] + [
                "人工审核拒绝发布"
            ]
        }

def publisher_agent(state: ContentCreationState) -> ContentCreationState:
    """发布智能体 - 负责最终发布"""

    final_content = state.get("final_content") or state["reviewed_content"]

    # 模拟发布流程
    publish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        **state,
        "final_content": final_content,
        "current_stage": "published",
        "collaboration_notes": state["collaboration_notes"] + [
            f"内容已于 {publish_time} 成功发布"
        ],
        "messages": state["messages"] + [
            AIMessage(content=f"✅ 内容发布成功! 发布时间: {publish_time}")
        ]
    }
```

### 5. 路由函数

```python
def route_next_agent(state: ContentCreationState) -> Literal["researcher", "writer", "reviewer", "human_review", "publisher", "end"]:
    """智能路由决策"""

    current_stage = state.get("current_stage", "start")
    approval_status = state.get("approval_status", "")
    assigned_agent = state.get("assigned_agent", "researcher")

    # 路由逻辑
    if current_stage == "start":
        return "researcher"
    elif current_stage == "research_completed":
        return "writer"
    elif current_stage == "draft_completed":
        return "reviewer"
    elif current_stage == "review_completed":
        if state.get("needs_human_review", False):
            return "human_review"
        else:
            return "publisher"
    elif approval_status == "human_approved":
        return "publisher"
    elif approval_status == "revision_required":
        return "writer"
    elif approval_status in ["rejected", "published"]:
        return "end"
    else:
        return "end"

def should_continue(state: ContentCreationState) -> Literal["continue", "end"]:
    """判断是否继续工作流"""

    current_stage = state.get("current_stage", "")
    approval_status = state.get("approval_status", "")

    if current_stage == "published" or approval_status == "rejected":
        return "end"

    return "continue"
```

### 6. 图构建和编译

```python
def create_content_creation_workflow():
    """创建内容创作工作流"""

    # 创建状态图
    workflow = StateGraph(ContentCreationState)

    # 添加节点
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("reviewer", reviewer_agent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("publisher", publisher_agent)

    # 设置入口点
    workflow.set_entry_point("researcher")

    # 添加条件边
    workflow.add_conditional_edges(
        "researcher",
        route_next_agent,
        {
            "writer": "writer",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "writer",
        route_next_agent,
        {
            "reviewer": "reviewer",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "reviewer",
        route_next_agent,
        {
            "human_review": "human_review",
            "publisher": "publisher",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "human_review",
        route_next_agent,
        {
            "writer": "writer",
            "publisher": "publisher",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "publisher",
        route_next_agent,
        {
            "end": END
        }
    )

    # 编译工作流
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]  # 在人工审核前中断
    )

    return app
```

### 7. 使用示例

```python
async def main():
    """主函数演示"""

    # 创建工作流
    app = create_content_creation_workflow()

    # 初始输入
    initial_state = {
        "topic": "人工智能在教育领域的应用前景",
        "messages": [],
        "current_stage": "start",
        "collaboration_notes": [],
        "quality_score": 0.0,
        "needs_human_review": False,
        "approval_status": "",
        "assigned_agent": "researcher"
    }

    # 配置线程ID用于检查点
    config = {"configurable": {"thread_id": "content_creation_001"}}

    try:
        print("🚀 开始内容创作工作流...")

        # 执行工作流
        async for event in app.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                print(f"\n📍 节点: {node_name}")
                print(f"🎯 当前阶段: {node_output.get('current_stage', 'N/A')}")
                print(f"👤 负责智能体: {node_output.get('assigned_agent', 'N/A')}")

                if node_output.get("collaboration_notes"):
                    print(f"📝 协作备注: {node_output['collaboration_notes'][-1]}")

        print("\n✅ 工作流执行完成!")

        # 获取最终状态
        final_state = app.get_state(config)
        print(f"\n📊 最终状态:")
        print(f"   - 阶段: {final_state.values.get('current_stage')}")
        print(f"   - 质量评分: {final_state.values.get('quality_score')}")
        print(f"   - 审批状态: {final_state.values.get('approval_status')}")

    except Exception as e:
        print(f"❌ 执行出错: {e}")

# 演示回退功能
async def demo_rollback():
    """演示回退功能"""

    app = create_content_creation_workflow()
    config = {"configurable": {"thread_id": "rollback_demo"}}

    # 获取状态历史
    state_history = app.get_state_history(config)

    print("📚 状态历史:")
    for i, state in enumerate(state_history):
        print(f"   {i+1}. 步骤{state.step}: {state.values.get('current_stage', 'N/A')}")

    # 回退到特定步骤
    if len(state_history) > 2:
        target_state = state_history[1]  # 回退到第2个状态
        print(f"\n🔄 回退到步骤 {target_state.step}")

        # 从该状态继续执行
        new_input = {"topic": "修改后的主题"}
        result = await app.ainvoke(new_input, config=config, start_from=target_state)
        print("✅ 回退并继续执行完成")

# 运行演示
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## 关键特性演示

### 1. 状态管理和持久化
- ✅ 使用SQLite检查点保存状态
- ✅ 支持工作流中断和恢复
- ✅ 状态在智能体间传递和更新

### 2. 多智能体协作
- ✅ 研究员 → 写作者 → 审核员 → 发布者的协作链
- ✅ 智能路由决策根据状态自动选择下一个智能体
- ✅ 并行处理和条件分支

### 3. 人机交互 (HITL)
- ✅ 在关键节点设置中断点
- ✅ 等待人工审核和决策
- ✅ 根据人工输入调整工作流路径

### 4. 错误处理和回退
- ✅ 支持工作流回退到任意检查点
- ✅ 状态历史查看和分析
- ✅ 从中断点恢复执行

### 5. 工具集成
- ✅ 网络搜索、文件保存、查重检查等工具
- ✅ 工具调用结果影响状态和流程

这个demo展示了LangGraph在构建复杂多智能体协作系统方面的强大能力，特别是其状态管理、检查点机制和人机协作功能。
