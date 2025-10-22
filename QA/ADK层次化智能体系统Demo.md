# ADK 层次化智能体系统 Demo

## 演示场景：智能客服支持系统

这个demo展示如何使用ADK构建一个层次化的智能客服系统，包含一线客服、技术专家、主管审批等多级智能体协作。

## 完整代码实现

### 1. 环境设置和依赖

```python
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# ADK 核心导入
from agent_dev_kit import LlmAgent, BaseAgent, BaseModel, FunctionTool
from agent_dev_kit import InvocationContext, Event, EventActions
from agent_dev_kit import BaseTool, ToolContext, BaseToolset
from agent_dev_kit import BaseAgentState, experimental
from agent_dev_kit.core import types
from agent_dev_kit.flows import AutoFlow, SingleFlow

# 模型配置
MODEL_CONFIG = {
    "model": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 1000
}
```

### 2. 状态和数据结构定义

```python
# 客服工单状态
class TicketStatus(Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

# 客服工单数据结构
@dataclass
class SupportTicket:
    ticket_id: str
    customer_id: str
    title: str
    description: str
    category: str
    priority: TicketPriority
    status: TicketStatus
    created_at: datetime
    assigned_agent: Optional[str] = None
    resolution: Optional[str] = None
    escalation_reason: Optional[str] = None
    interaction_history: List[Dict] = None

    def __post_init__(self):
        if self.interaction_history is None:
            self.interaction_history = []

# 客户信息
@dataclass
class CustomerInfo:
    customer_id: str
    name: str
    email: str
    phone: str
    tier: str  # "basic", "premium", "enterprise"
    account_status: str
    previous_tickets: List[str]

# 智能体状态管理
@experimental
class CustomerServiceState(BaseAgentState):
    """客服智能体状态"""
    current_ticket: Optional[SupportTicket] = None
    customer_info: Optional[CustomerInfo] = None
    escalation_count: int = 0
    resolution_attempts: int = 0
    requires_approval: bool = False
    pending_actions: List[Dict] = []
    knowledge_base_queries: List[str] = []
```

### 3. 工具系统定义

```python
# 客户信息查询工具
class CustomerLookupTool(BaseTool):
    name: str = "customer_lookup"
    description: str = "查询客户详细信息"

    async def run(self, context: ToolContext, customer_id: str) -> CustomerInfo:
        """查询客户信息"""
        # 模拟数据库查询
        customers_db = {
            "CUST001": CustomerInfo(
                customer_id="CUST001",
                name="张三",
                email="zhangsan@example.com",
                phone="13812345678",
                tier="premium",
                account_status="active",
                previous_tickets=["TK001", "TK005"]
            ),
            "CUST002": CustomerInfo(
                customer_id="CUST002",
                name="李四",
                email="lisi@example.com",
                phone="13987654321",
                tier="basic",
                account_status="active",
                previous_tickets=["TK003"]
            )
        }

        customer = customers_db.get(customer_id)
        if not customer:
            raise ValueError(f"客户 {customer_id} 不存在")

        return customer

# 知识库搜索工具
class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_search"
    description: str = "搜索知识库获取解决方案"

    async def run(self, context: ToolContext, query: str, category: str = "") -> Dict:
        """搜索知识库"""
        # 模拟知识库搜索
        knowledge_base = {
            "登录问题": {
                "solutions": [
                    "清除浏览器缓存和Cookie",
                    "检查密码是否正确",
                    "尝试重置密码"
                ],
                "success_rate": 0.85
            },
            "支付问题": {
                "solutions": [
                    "检查银行卡余额",
                    "确认支付信息正确",
                    "联系银行确认交易状态"
                ],
                "success_rate": 0.90
            },
            "账户问题": {
                "solutions": [
                    "验证身份信息",
                    "检查账户状态",
                    "联系客服处理"
                ],
                "success_rate": 0.75
            }
        }

        # 简单关键词匹配
        for kb_category, info in knowledge_base.items():
            if kb_category in query or category in kb_category:
                return {
                    "category": kb_category,
                    "solutions": info["solutions"],
                    "success_rate": info["success_rate"],
                    "query": query
                }

        return {
            "category": "通用",
            "solutions": ["联系技术支持", "提交工单"],
            "success_rate": 0.50,
            "query": query
        }

# 工单操作工具集
class TicketOperationToolset(BaseToolset):
    prefix: str = "ticket"

    async def get_tools_with_prefix(self, ctx) -> List[BaseTool]:
        return [
            TicketUpdateTool(prefix=self.prefix),
            TicketEscalateTool(prefix=self.prefix),
            TicketCloseTool(prefix=self.prefix)
        ]

class TicketUpdateTool(BaseTool):
    name: str = "update_ticket"
    description: str = "更新工单状态和信息"

    async def run(self, context: ToolContext, ticket_id: str, **updates) -> bool:
        """更新工单"""
        print(f"📝 更新工单 {ticket_id}: {updates}")
        return True

class TicketEscalateTool(BaseTool):
    name: str = "escalate_ticket"
    description: str = "将工单升级到上级处理"
    requires_confirmation: bool = True  # 需要确认的工具

    async def run(self, context: ToolContext, ticket_id: str, reason: str, target_level: str) -> bool:
        """升级工单"""
        print(f"⬆️ 工单 {ticket_id} 升级到 {target_level}, 原因: {reason}")
        return True

    async def get_confirmation_prompt(self, ticket_id: str, reason: str, target_level: str) -> str:
        return f"""
        即将升级工单处理:
        - 工单ID: {ticket_id}
        - 升级原因: {reason}
        - 目标级别: {target_level}

        请确认是否升级?
        """

class TicketCloseTool(BaseTool):
    name: str = "close_ticket"
    description: str = "关闭工单"
    requires_confirmation: bool = True

    async def run(self, context: ToolContext, ticket_id: str, resolution: str) -> bool:
        """关闭工单"""
        print(f"✅ 关闭工单 {ticket_id}, 解决方案: {resolution}")
        return True
```

### 4. 层次化智能体定义

```python
# 一线客服智能体
class FrontlineAgent(LlmAgent):
    """一线客服智能体 - 处理基础问题"""

    name: str = "frontline_support"
    description: str = "处理常见客服问题，提供基础支持"
    instruction: str = """
    你是一线客服代表，负责处理客户的基础问题。

    工作职责:
    1. 热情接待客户，了解问题详情
    2. 查询客户信息和历史记录
    3. 搜索知识库寻找解决方案
    4. 尝试解决常见问题
    5. 无法解决时升级给技术专家

    注意事项:
    - 保持专业和友善的态度
    - 准确记录问题和处理过程
    - 及时识别需要升级的复杂问题
    """

    tools: List = [CustomerLookupTool(), KnowledgeBaseTool()]
    disallow_transfer_to_peers: bool = True  # 不允许平级转移

    async def _evaluate_escalation_need(self, context: InvocationContext) -> bool:
        """评估是否需要升级"""
        state = self._load_agent_state(context, CustomerServiceState)
        if not state or not state.current_ticket:
            return False

        # 升级条件
        escalation_criteria = [
            state.resolution_attempts >= 3,  # 尝试次数过多
            state.current_ticket.priority in [TicketPriority.HIGH, TicketPriority.URGENT],
            state.customer_info and state.customer_info.tier == "enterprise",
            "技术" in state.current_ticket.category,
            "账户冻结" in state.current_ticket.description
        ]

        return any(escalation_criteria)

# 技术专家智能体
class TechnicalExpertAgent(LlmAgent):
    """技术专家智能体 - 处理复杂技术问题"""

    name: str = "technical_expert"
    description: str = "处理复杂技术问题和深度故障排查"
    instruction: str = """
    你是技术专家，负责处理复杂的技术问题。

    工作职责:
    1. 接收一线客服升级的技术问题
    2. 进行深度技术分析和故障排查
    3. 提供专业的技术解决方案
    4. 指导客户进行高级操作
    5. 必要时申请系统级操作权限

    专业能力:
    - 系统架构和网络问题诊断
    - 数据库和API接口故障排查
    - 账户权限和安全问题处理
    - 高级配置和定制化需求
    """

    tools: List = [CustomerLookupTool(), KnowledgeBaseTool(), TicketOperationToolset()]

    async def _check_approval_needed(self, context: InvocationContext, action: str) -> bool:
        """检查是否需要主管审批"""
        high_impact_actions = [
            "数据恢复", "账户解冻", "退款处理",
            "系统配置修改", "权限提升"
        ]
        return any(action_type in action for action_type in high_impact_actions)

# 客服主管智能体
class SupervisorAgent(LlmAgent):
    """客服主管智能体 - 处理审批和协调"""

    name: str = "supervisor"
    description: str = "负责审批高级操作和协调团队工作"
    instruction: str = """
    你是客服主管，负责团队协调和重要决策审批。

    工作职责:
    1. 审批高风险操作和特殊请求
    2. 协调各级客服资源分配
    3. 处理客户投诉和升级问题
    4. 制定解决方案和工作指导
    5. 监督服务质量和客户满意度

    决策原则:
    - 优先考虑客户体验和满意度
    - 平衡公司政策和客户需求
    - 确保操作合规和风险可控
    - 及时决策避免客户等待
    """

    tools: List = [CustomerLookupTool(), TicketOperationToolset()]
    disallow_transfer_to_parent: bool = True  # 主管级别，无上级转移

# 协调者智能体 (根智能体)
class CustomerServiceCoordinator(LlmAgent):
    """客服协调者 - 根智能体，负责整体流程控制"""

    name: str = "service_coordinator"
    description: str = "智能客服系统协调者，负责问题分析和智能体分配"
    instruction: str = """
    你是智能客服系统的协调者，负责分析客户问题并分配给合适的智能体处理。

    工作流程:
    1. 接收客户问题，进行初步分析
    2. 根据问题类型和复杂度选择合适的处理智能体
    3. 监控处理进度，必要时进行调度
    4. 确保问题得到及时有效的解决

    分配策略:
    - 常见问题 → 一线客服
    - 技术问题 → 技术专家
    - 投诉/特殊请求 → 客服主管
    - 紧急问题 → 直接升级
    """

    sub_agents: List[BaseAgent] = []  # 将在初始化时设置
    tools: List = [CustomerLookupTool()]

def create_customer_service_system():
    """创建层次化客服系统"""

    # 创建子智能体
    frontline = FrontlineAgent(**MODEL_CONFIG)
    technical_expert = TechnicalExpertAgent(**MODEL_CONFIG)
    supervisor = SupervisorAgent(**MODEL_CONFIG)

    # 创建根协调者智能体
    coordinator = CustomerServiceCoordinator(
        **MODEL_CONFIG,
        sub_agents=[frontline, technical_expert, supervisor]
    )

    return coordinator
```

### 5. 智能体转移和协作逻辑

```python
# 智能体转移决策
class ServiceTransferLogic:
    """客服智能体转移逻辑"""

    @staticmethod
    async def analyze_transfer_need(context: InvocationContext) -> Dict[str, Any]:
        """分析转移需求"""
        current_agent = context.agent.name
        state = context.get_state_value("service_state")

        transfer_decision = {
            "should_transfer": False,
            "target_agent": None,
            "reason": "",
            "priority": "normal"
        }

        if current_agent == "frontline_support":
            # 一线客服的转移逻辑
            if state and hasattr(state, 'current_ticket'):
                ticket = state.current_ticket

                # 技术问题转技术专家
                if "技术" in ticket.category or "故障" in ticket.description:
                    transfer_decision.update({
                        "should_transfer": True,
                        "target_agent": "technical_expert",
                        "reason": "技术问题需要专家处理"
                    })

                # 高优先级直接升级
                elif ticket.priority in [TicketPriority.HIGH, TicketPriority.URGENT]:
                    transfer_decision.update({
                        "should_transfer": True,
                        "target_agent": "supervisor",
                        "reason": "高优先级问题",
                        "priority": "high"
                    })

        elif current_agent == "technical_expert":
            # 技术专家的转移逻辑
            if state and state.requires_approval:
                transfer_decision.update({
                    "should_transfer": True,
                    "target_agent": "supervisor",
                    "reason": "需要主管审批"
                })

        return transfer_decision

# 自定义流程控制
class CustomerServiceFlow(AutoFlow):
    """客服专用流程控制"""

    async def _handle_agent_transfer(self, ctx: InvocationContext) -> bool:
        """处理智能体转移"""
        transfer_analysis = await ServiceTransferLogic.analyze_transfer_need(ctx)

        if transfer_analysis["should_transfer"]:
            target_agent_name = transfer_analysis["target_agent"]
            reason = transfer_analysis["reason"]

            # 查找目标智能体
            target_agent = None
            for sub_agent in ctx.agent.sub_agents:
                if sub_agent.name == target_agent_name:
                    target_agent = sub_agent
                    break

            if target_agent:
                # 记录转移信息
                ctx.set_state_value("transfer_history", {
                    "from": ctx.agent.name,
                    "to": target_agent_name,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })

                # 执行转移
                print(f"🔄 智能体转移: {ctx.agent.name} → {target_agent_name}")
                print(f"📝 转移原因: {reason}")

                # 这里会调用目标智能体
                return True

        return False
```

### 6. 回调和监控系统

```python
# 智能体执行前回调
async def before_agent_callback(callback_context) -> Optional[types.Content]:
    """智能体执行前的回调处理"""
    ctx = callback_context.ctx
    agent_name = ctx.agent.name

    print(f"🚀 启动智能体: {agent_name}")
    print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")

    # 记录智能体启动
    ctx.set_state_value("agent_start_time", datetime.now())

    # 检查智能体状态
    if agent_name != "service_coordinator":
        ticket_info = ctx.get_state_value("current_ticket")
        if ticket_info:
            print(f"🎫 处理工单: {ticket_info.get('ticket_id', 'N/A')}")

    return None

# 智能体执行后回调
async def after_agent_callback(callback_context) -> Optional[types.Content]:
    """智能体执行后的回调处理"""
    ctx = callback_context.ctx
    agent_name = ctx.agent.name

    start_time = ctx.get_state_value("agent_start_time")
    if start_time:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"⏱️ {agent_name} 执行耗时: {duration:.2f}秒")

    # 更新服务统计
    stats = ctx.get_state_value("service_stats", {})
    stats[agent_name] = stats.get(agent_name, 0) + 1
    ctx.set_state_value("service_stats", stats)

    print(f"✅ 智能体 {agent_name} 执行完成")
    return None

# 工具确认回调
async def tool_confirmation_callback(tool: BaseTool, args: Dict, context: ToolContext) -> Optional[Dict]:
    """工具确认回调"""
    if hasattr(tool, 'requires_confirmation') and tool.requires_confirmation:
        print(f"\n🔔 工具确认请求:")
        print(f"🛠️ 工具: {tool.name}")
        print(f"📝 参数: {args}")

        if hasattr(tool, 'get_confirmation_prompt'):
            prompt = await tool.get_confirmation_prompt(**args)
            print(f"❓ 确认信息:\n{prompt}")

        # 模拟用户确认 (实际应用中会是Web界面)
        confirmation = input("请确认是否执行 (y/n): ").lower().strip()

        if confirmation != 'y':
            raise Exception(f"用户取消了工具 {tool.name} 的执行")

    return None
```

### 7. 使用示例和演示

```python
async def demo_customer_service_system():
    """演示客服系统"""

    # 创建客服系统
    service_system = create_customer_service_system()

    # 配置回调
    service_system.before_agent_callback = before_agent_callback
    service_system.after_agent_callback = after_agent_callback

    # 模拟客户问题
    test_scenarios = [
        {
            "ticket_id": "TK2024001",
            "customer_id": "CUST001",
            "title": "登录问题",
            "description": "无法登录系统，提示密码错误",
            "category": "账户问题",
            "priority": TicketPriority.MEDIUM
        },
        {
            "ticket_id": "TK2024002",
            "customer_id": "CUST002",
            "title": "支付失败",
            "description": "支付时系统报错，订单未生成",
            "category": "技术问题",
            "priority": TicketPriority.HIGH
        },
        {
            "ticket_id": "TK2024003",
            "customer_id": "CUST001",
            "title": "账户被冻结",
            "description": "企业账户突然被冻结，影响业务",
            "category": "账户问题",
            "priority": TicketPriority.URGENT
        }
    ]

    print("🎯 开始客服系统演示\n")

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{'='*50}")
        print(f"📋 场景 {i}: {scenario['title']}")
        print(f"🆔 工单ID: {scenario['ticket_id']}")
        print(f"👤 客户ID: {scenario['customer_id']}")
        print(f"📊 优先级: {scenario['priority'].value}")
        print(f"📝 描述: {scenario['description']}")
        print(f"{'='*50}\n")

        # 创建工单
        ticket = SupportTicket(
            ticket_id=scenario["ticket_id"],
            customer_id=scenario["customer_id"],
            title=scenario["title"],
            description=scenario["description"],
            category=scenario["category"],
            priority=scenario["priority"],
            status=TicketStatus.CREATED,
            created_at=datetime.now()
        )

        # 创建调用上下文
        context = InvocationContext(
            invocation_id=f"inv_{scenario['ticket_id']}",
            session_id=f"session_{scenario['customer_id']}",
            agent=service_system
        )

        # 设置初始状态
        context.set_state_value("current_ticket", ticket)
        context.set_state_value("service_stats", {})

        try:
            # 运行客服系统
            print("🚀 开始处理客户问题...\n")

            async for event in service_system.run_async(context):
                if event.content:
                    print(f"💬 {event.author}: {event.content}")

                if event.actions.end_of_agent:
                    print(f"✅ 智能体 {event.author} 处理完成")

                # 状态更新
                if event.actions.state_delta:
                    for key, value in event.actions.state_delta.items():
                        context.set_state_value(key, value)

            # 显示处理结果
            print(f"\n📊 处理统计:")
            stats = context.get_state_value("service_stats", {})
            for agent_name, count in stats.items():
                print(f"   - {agent_name}: {count}次调用")

            print(f"\n✅ 工单 {ticket.ticket_id} 处理完成\n")

        except Exception as e:
            print(f"❌ 处理出错: {e}")

        # 等待用户确认继续
        if i < len(test_scenarios):
            input("按回车键继续下一个场景...")
            print()

# 状态持久化演示
async def demo_state_persistence():
    """演示状态持久化"""

    print("💾 状态持久化演示")

    # 创建持久化的智能体状态
    persistent_state = CustomerServiceState(
        escalation_count=2,
        resolution_attempts=3,
        requires_approval=True,
        pending_actions=[
            {"action": "账户解冻", "timestamp": datetime.now().isoformat()}
        ],
        knowledge_base_queries=["登录问题", "账户冻结"]
    )

    print(f"📊 智能体状态:")
    print(f"   - 升级次数: {persistent_state.escalation_count}")
    print(f"   - 解决尝试: {persistent_state.resolution_attempts}")
    print(f"   - 需要审批: {persistent_state.requires_approval}")
    print(f"   - 待处理操作: {len(persistent_state.pending_actions)}")
    print(f"   - 知识库查询历史: {persistent_state.knowledge_base_queries}")

    # 状态序列化和恢复
    state_json = persistent_state.model_dump(mode='json')
    print(f"\n💽 状态序列化: {state_json}")

    # 从JSON恢复状态
    restored_state = CustomerServiceState.model_validate(state_json)
    print(f"🔄 状态恢复成功: {restored_state.escalation_count}次升级")

# 工具确认流程演示
async def demo_tool_confirmation():
    """演示工具确认流程"""

    print("🛠️ 工具确认流程演示")

    # 创建需要确认的工具
    escalate_tool = TicketEscalateTool()
    close_tool = TicketCloseTool()

    # 模拟工具调用
    mock_context = ToolContext(session_id="demo_session")

    try:
        # 尝试升级工单 (需要确认)
        print("\n1. 尝试升级工单...")
        await tool_confirmation_callback(
            escalate_tool,
            {
                "ticket_id": "TK2024001",
                "reason": "技术问题复杂，需要专家处理",
                "target_level": "技术专家"
            },
            mock_context
        )
        result = await escalate_tool.run(
            mock_context,
            ticket_id="TK2024001",
            reason="技术问题复杂，需要专家处理",
            target_level="技术专家"
        )
        print(f"✅ 升级成功: {result}")

    except Exception as e:
        print(f"❌ 升级被取消: {e}")

# 主演示函数
async def main():
    """主演示函数"""

    print("🎪 ADK 层次化智能体系统演示")
    print("=" * 60)

    demos = [
        ("客服系统协作演示", demo_customer_service_system),
        ("状态持久化演示", demo_state_persistence),
        ("工具确认流程演示", demo_tool_confirmation)
    ]

    for name, demo_func in demos:
        print(f"\n🚀 开始 {name}")
        print("-" * 40)
        try:
            await demo_func()
        except Exception as e:
            print(f"❌ 演示出错: {e}")

        print(f"✅ {name} 完成")
        input("按回车键继续...")

if __name__ == "__main__":
    asyncio.run(main())
```

## 关键特性演示

### 1. 层次化智能体架构 ✅
- **根协调者**: 分析问题并分配给合适的子智能体
- **一线客服**: 处理常见问题，必要时升级
- **技术专家**: 处理复杂技术问题
- **客服主管**: 审批高风险操作和特殊请求

### 2. 智能体转移机制 ✅
- **自动转移**: 基于问题类型和复杂度自动选择智能体
- **条件转移**: 根据业务规则进行智能路由
- **转移历史**: 记录完整的转移链路和原因

### 3. 工具确认流程 (HITL) ✅
- **风险识别**: 自动识别高风险操作
- **人工确认**: 重要操作需要人工审批
- **确认流程**: 提供详细的确认信息和选项

### 4. 状态管理和持久化 ✅
- **智能体状态**: 每个智能体独立的状态管理
- **状态继承**: 子智能体继承父智能体上下文
- **状态序列化**: 支持状态的保存和恢复

### 5. 事件驱动架构 ✅
- **事件流**: 完整的事件驱动处理流程
- **回调系统**: 智能体执行前后的监控和处理
- **状态变更**: 通过事件进行状态同步

### 6. 企业级特性 ✅
- **权限控制**: 不同级别智能体的权限限制
- **审批流程**: 多层级审批和决策机制
- **监控统计**: 完整的执行统计和性能监控

这个demo展示了ADK在构建企业级多智能体系统方面的强大能力，特别是其层次化架构、智能转移机制和工具确认流程等核心特性。
