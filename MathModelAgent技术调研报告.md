# MathModelAgent 技术调研报告

## 项目概述

**MathModelAgent** 是一个专为数学建模设计的多智能体自动化系统，能够自动完成从问题分析到代码实现再到论文撰写的完整数学建模流程。系统设计目标是将原本需要 3 天的数学建模比赛时间压缩到 1 小时，自动生成可获奖级别的建模论文。

### 核心特性
- 🤖 **多智能体协作**: 协调员、建模手、代码手、写作手四个专业化 Agent
- 💻 **代码解释器**: 支持本地 Jupyter 和云端 E2B 代码执行环境
- 📝 **自动论文生成**: 按照学术标准生成完整的数学建模论文
- 🔄 **多模型支持**: 每个 Agent 可配置不同的 LLM 模型
- 🧩 **自定义模板**: 支持 Prompt 注入和模板自定义

## 技术架构

### 整体架构图

```
用户输入 → CoordinatorAgent → ModelerAgent → CoderAgent → WriterAgent → 论文输出
    ↓              ↓              ↓            ↓            ↓
问题识别      问题分析        建模方案     代码实现     论文撰写
    ↓              ↓              ↓            ↓            ↓
JSON格式化    数学模型        Jupyter      学术论文     最终成果
```

### 核心组件

#### 1. Agent 架构设计

所有 Agent 都继承自基础 `Agent` 类，实现了统一的对话管理和内存清理机制：

```python
class Agent:
    def __init__(self, task_id: str, model: LLM, max_chat_turns: int = 30, max_memory: int = 12):
        self.task_id = task_id
        self.model = model
        self.chat_history: list[dict] = []  # 对话历史
        self.max_chat_turns = max_chat_turns  # 最大对话轮次
        self.max_memory = max_memory  # 最大记忆轮次
```

**关键特性**：
- **内存管理**: 当对话历史超过限制时，自动使用 LLM 总结压缩历史
- **工具调用安全**: 确保不会在工具调用序列中间切断对话
- **异常处理**: 完善的错误处理和重试机制

#### 2. 专业化 Agent 实现

##### CoordinatorAgent (协调员)
- **职责**: 识别用户意图，判断是否为数学建模问题，格式化问题结构
- **输入**: 用户原始问题描述
- **输出**: 结构化的 JSON 格式问题
- **特点**: 包含重试机制，确保 JSON 解析成功

```python
# 输出格式示例
{
  "title": "问题标题",
  "background": "问题背景",
  "ques_count": 3,
  "ques1": "问题1描述",
  "ques2": "问题2描述", 
  "ques3": "问题3描述"
}
```

##### ModelerAgent (建模手)
- **职责**: 针对每个问题设计数学模型和求解方案
- **输入**: CoordinatorAgent 格式化后的问题
- **输出**: 每个问题的建模思路和方案
- **特点**: 专注于数学模型设计，不涉及具体代码实现

```python
# 输出格式示例
{
  "eda": "数据分析EDA方案，可视化方案",
  "ques1": "问题1的建模思路和模型方案",
  "ques2": "问题2的建模思路和模型方案",
  "sensitivity_analysis": "敏感性分析方案"
}
```

##### CoderAgent (代码手)
- **职责**: 根据建模方案编写和执行 Python 代码
- **输入**: ModelerAgent 的建模方案
- **输出**: 代码执行结果和生成的图片
- **核心功能**:
  - **工具调用**: 使用 `execute_code` 工具执行代码
  - **错误处理**: 自动检测错误并反思修正
  - **重试机制**: 最大重试次数限制
  - **代码保存**: 自动保存为 Jupyter Notebook

**代码执行流程**:
```python
while retry_count < max_retries:
    response = await self.model.chat(tools=coder_tools)
    if has_tool_calls:
        result = await code_interpreter.execute_code(code)
        if error_occurred:
            # 反思错误并重试
            continue
        else:
            # 继续下一步
            continue
    else:
        # 任务完成
        return result
```

##### WriterAgent (写作手)
- **职责**: 根据代码执行结果撰写学术论文
- **输入**: CoderAgent 的执行结果和生成的图片
- **输出**: 标准格式的学术论文章节
- **特殊功能**:
  - **文献搜索**: 集成 OpenAlex Scholar 进行文献检索
  - **图片引用**: 自动引用代码生成的图表
  - **模板化写作**: 按照比赛模板格式撰写

#### 3. 代码解释器系统

系统支持两种代码执行环境：

##### LocalCodeInterpreter (本地解释器)
- **基于**: Jupyter Kernel
- **优势**: 无需网络，完全本地控制
- **功能**: 代码保存为 .ipynb 文件，便于后续编辑

##### E2BCodeInterpreter (云端解释器)  
- **基于**: E2B 云端沙盒
- **优势**: 隔离环境，无需本地配置
- **适用**: 网络环境良好的场景

**工厂模式创建**:
```python
async def create_interpreter(kind: Literal["remote", "local"] = "local"):
    if not settings.E2B_API_KEY:
        kind = "local"  # 默认本地
    
    if kind == "remote":
        return await E2BCodeInterpreter.create()
    else:
        return LocalCodeInterpreter()
```

#### 4. 工作流编排 (Workflow)

##### 解决方案流程 (Solution Flows)
按照数学建模标准流程执行：
1. **EDA** (探索性数据分析)
2. **Ques1, Ques2, ...** (逐个问题求解)  
3. **Sensitivity Analysis** (敏感性分析)

##### 写作流程 (Write Flows)
按照学术论文结构撰写：
1. **FirstPage** (封面、摘要、关键词)
2. **RepeatQues** (问题重述)
3. **AnalysisQues** (问题分析)
4. **ModelAssumption** (模型假设)
5. **Symbol** (符号说明)
6. **Judge** (模型评价)

#### 5. LLM 集成架构

##### 多模型支持
通过 LLMFactory 为不同 Agent 分配专门的模型：

```python
class LLMFactory:
    def get_all_llms(self) -> tuple[LLM, LLM, LLM, LLM]:
        coordinator_llm = LLM(model=settings.COORDINATOR_MODEL)
        modeler_llm = LLM(model=settings.MODELER_MODEL)  
        coder_llm = LLM(model=settings.CODER_MODEL)
        writer_llm = LLM(model=settings.WRITER_MODEL)
        return coordinator_llm, modeler_llm, coder_llm, writer_llm
```

##### LiteLLM 集成
- 支持 100+ LLM 提供商
- 统一的 API 接口
- 灵活的模型配置

### 实时通信架构

#### WebSocket 实时更新
- **Redis 消息队列**: 用于跨进程消息传递
- **实时状态推送**: 前端实时显示任务进度
- **系统消息**: 错误提示、成功通知等

```python
await redis_manager.publish_message(
    task_id,
    SystemMessage(content="代码手开始求解问题1")
)
```

#### 消息类型
- `SystemMessage`: 系统状态消息
- `InterpreterMessage`: 代码执行消息  
- `WriterMessage`: 写作进度消息

## 关键技术特点

### 1. 无框架设计 (Agentless Workflow)
- **轻量化**: 不依赖重型 Agent 框架
- **成本优化**: 避免框架开销
- **灵活性**: 易于定制和扩展

### 2. 专业化分工
- **CoordinatorAgent**: 专门负责问题理解和格式化
- **ModelerAgent**: 专门负责数学建模设计
- **CoderAgent**: 专门负责代码实现和执行
- **WriterAgent**: 专门负责学术写作

### 3. 容错机制
- **重试策略**: 每个 Agent 都有重试限制
- **错误反思**: CoderAgent 能自动分析错误并修正
- **安全降级**: 内存清理失败时的安全策略

### 4. 模板化和可定制
- **Prompt 模板**: 可通过配置文件自定义 Prompt
- **论文模板**: 支持不同比赛的论文格式
- **流程配置**: 可调整工作流程顺序

## 项目结构分析

### 后端架构 (FastAPI)
```
backend/
├── app/
│   ├── core/                 # 核心业务逻辑
│   │   ├── agents/          # Agent 实现
│   │   ├── llm/             # LLM 集成
│   │   ├── flows.py         # 工作流编排
│   │   ├── workflow.py      # 主工作流
│   │   └── prompts.py       # Prompt 模板
│   ├── tools/               # 工具集成
│   │   ├── base_interpreter.py
│   │   ├── local_interpreter.py
│   │   ├── e2b_interpreter.py
│   │   └── openalex_scholar.py
│   ├── schemas/             # 数据模型
│   ├── routers/             # API 路由
│   ├── services/            # 服务层
│   └── utils/               # 工具函数
```

### 前端架构 (Vue 3 + TypeScript)
```
frontend/
├── src/
│   ├── components/          # 组件库
│   │   ├── AgentEditor/     # Agent 编辑器
│   │   ├── ChatArea.vue     # 聊天界面
│   │   └── NotebookArea.vue # Notebook 显示
│   ├── pages/               # 页面组件
│   ├── stores/              # 状态管理 (Pinia)
│   └── utils/               # 工具函数
```

## 使用示例

### 1. 问题输入
```
用户输入: "某地区人口增长预测模型
背景：该地区历年人口数据如附件所示
问题1：建立人口增长预测模型
问题2：预测未来5年人口变化趋势  
问题3：分析影响人口增长的主要因素"
```

### 2. 系统处理流程
1. **CoordinatorAgent**: 将输入格式化为结构化JSON
2. **ModelerAgent**: 设计逻辑回归、时间序列等模型方案
3. **CoderAgent**: 实现数据分析、模型训练、预测代码
4. **WriterAgent**: 撰写摘要、模型介绍、结果分析等章节

### 3. 最终输出
- **notebook.ipynb**: 完整的代码实现过程
- **res.md**: 标准格式的学术论文
- **图表文件**: 数据可视化结果

## 技术优势

### 1. 专业化和自动化
- **领域专业**: 专门针对数学建模优化
- **端到端**: 从问题到论文的完整自动化
- **标准化**: 符合数学建模比赛标准

### 2. 架构设计优秀
- **模块化**: 各 Agent 职责清晰，易于维护
- **可扩展**: 容易添加新的 Agent 或功能
- **容错性**: 完善的错误处理和重试机制

### 3. 实用性强
- **多环境支持**: 本地和云端代码执行
- **多模型支持**: 可配置不同 LLM
- **模板化**: 支持自定义模板和格式

## 潜在改进方向

### 1. 增强功能
- **人机交互 (HIL)**: 允许用户在流程中干预
- **多语言支持**: R 语言、MATLAB 等
- **更多绘图工具**: PlantUML、Mermaid.js 等
- **知识库集成**: RAG 检索增强

### 2. 性能优化
- **并行处理**: WriterAgent 的并行写作
- **缓存机制**: 减少重复计算
- **流式输出**: 实时显示生成过程

### 3. 质量提升
- **评估反馈**: 结果评估和改进机制
- **基准测试**: 标准化测试集
- **A2A 切换**: Agent 间智能切换

## 技术总结

MathModelAgent 代表了一种创新的多智能体协作范式，在数学建模自动化领域展现了显著的技术优势：

1. **架构设计**: 采用无框架的轻量化设计，各 Agent 专业化分工明确
2. **工程实现**: 完善的错误处理、内存管理和实时通信机制
3. **用户体验**: 提供完整的 Web 界面和实时进度反馈
4. **可扩展性**: 模块化设计便于功能扩展和维护

该项目为 AI Agent 在专业领域的应用提供了优秀的参考案例，展示了多 Agent 协作在复杂任务自动化中的巨大潜力。

---

*本调研报告基于 MathModelAgent 开源项目代码分析整理，项目地址：https://github.com/jihe520/MathModelAgent*