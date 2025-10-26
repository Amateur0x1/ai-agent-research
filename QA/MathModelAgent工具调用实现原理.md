# MathModelAgent 工具调用实现原理

## Q: 如何实现工具调用的？是自己实现的吗？怎么实现的？

### 总体设计思路

**MathModelAgent 是基于 OpenAI Function Calling 标准自己实现的工具调用系统**，没有使用现成的工具框架。采用标准的 Function Calling 协议，但在具体执行层面完全自主实现。

### 工具调用架构图

```
LLM 模型                Agent                    工具执行器
    ↓                    ↓                         ↓
发起工具调用  →  解析工具调用参数  →  执行具体工具逻辑
    ↓                    ↓                         ↓ 
tool_calls        tool_call.function     实际执行结果
    ↓                    ↓                         ↓
返回结果      ←  构造tool响应消息   ←    返回执行结果
```

### 详细实现分析

#### 1. 工具定义 (Function Schema)

**CoderAgent 的工具定义**：
```python
# app/core/functions.py
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
```

**WriterAgent 的工具定义**：
```python
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

#### 2. LLM 调用集成

**通过 LiteLLM 统一接口传递工具**：
```python
# app/core/llm/llm.py
class LLM:
    async def chat(
        self, 
        history: list, 
        tools: list = None,
        tool_choice: str = "auto"
    ):
        response = await litellm.achat(
            model=self.model,
            messages=history,
            tools=tools,          # 传递工具定义
            tool_choice=tool_choice,  # 工具选择策略
            api_key=self.api_key,
            base_url=self.base_url
        )
        return response
```

#### 3. 工具调用检测和解析

**CoderAgent 中的工具调用处理**：
```python
# app/core/agents/coder_agent.py
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

#### 4. 具体工具执行器实现

##### 代码执行工具 (execute_code)

**LocalCodeInterpreter (本地执行)**：
```python
# app/tools/local_interpreter.py
class LocalCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, work_dir: str):
        self.kernel_manager = AsyncKernelManager()
        self.kernel_client = None
        self.work_dir = work_dir
    
    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        """
        执行代码并返回结果
        Returns: (结果文本, 是否出错, 错误信息)
        """
        try:
            # 1. 确保 kernel 运行
            if not self.kernel_client:
                await self._start_kernel()
            
            # 2. 执行代码
            msg_id = self.kernel_client.execute(code, store_history=True)
            
            # 3. 等待执行结果
            reply = await self._wait_for_reply(msg_id)
            
            # 4. 处理执行结果
            if reply['content']['status'] == 'ok':
                # 成功执行
                outputs = await self._collect_outputs(msg_id)
                return self._format_outputs(outputs), False, ""
            else:
                # 执行错误
                error_info = reply['content']
                error_msg = f"{error_info.get('ename', '')}: {error_info.get('evalue', '')}"
                return "", True, error_msg
                
        except Exception as e:
            return "", True, str(e)
    
    async def _collect_outputs(self, msg_id: str) -> list:
        """收集执行输出"""
        outputs = []
        while True:
            try:
                msg = await asyncio.wait_for(
                    self.kernel_client.get_iopub_msg(timeout=1), 
                    timeout=30
                )
                
                if msg['parent_header'].get('msg_id') == msg_id:
                    msg_type = msg['msg_type']
                    content = msg['content']
                    
                    if msg_type == 'stream':
                        outputs.append({
                            'type': 'stream',
                            'name': content['name'],
                            'text': content['text']
                        })
                    elif msg_type == 'display_data':
                        outputs.append({
                            'type': 'display_data',
                            'data': content['data']
                        })
                    elif msg_type == 'execute_result':
                        outputs.append({
                            'type': 'execute_result',
                            'data': content['data']
                        })
                    elif msg_type == 'status' and content['execution_state'] == 'idle':
                        break
                        
            except asyncio.TimeoutError:
                break
                
        return outputs
```

**E2BCodeInterpreter (云端执行)**：
```python
# app/tools/e2b_interpreter.py
class E2BCodeInterpreter(BaseCodeInterpreter):
    def __init__(self, sandbox, task_id: str):
        self.sandbox = sandbox
        self.task_id = task_id
    
    @classmethod
    async def create(cls, task_id: str, work_dir: str):
        # 创建 E2B 沙盒
        sandbox = await Sandbox.create(
            template="base",
            envs={"WORK_DIR": work_dir}
        )
        return cls(sandbox, task_id)
    
    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        """在云端沙盒中执行代码"""
        try:
            # 1. 在沙盒中执行代码
            execution = await self.sandbox.run_code(
                code,
                language="python"
            )
            
            # 2. 处理执行结果
            if execution.error:
                return "", True, execution.error.message
            else:
                # 格式化输出结果
                result_text = ""
                for result in execution.results:
                    if result.text:
                        result_text += result.text
                    if result.png:
                        # 处理图片输出
                        result_text += f"[生成图片: {result.png}]"
                
                return result_text, False, ""
                
        except Exception as e:
            return "", True, str(e)
```

##### 文献搜索工具 (search_papers)

```python
# app/tools/openalex_scholar.py
class OpenAlexScholar:
    def __init__(self, task_id: str, email: str):
        self.task_id = task_id
        self.email = email
        self.base_url = "https://api.openalex.org"
    
    async def search_papers(self, query: str, max_results: int = 5) -> list:
        """搜索学术论文"""
        try:
            # 1. 构造搜索请求
            params = {
                "search": query,
                "per_page": max_results,
                "mailto": self.email
            }
            
            # 2. 调用 OpenAlex API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/works",
                    params=params
                )
                response.raise_for_status()
                
            # 3. 解析结果
            data = response.json()
            papers = []
            
            for work in data.get("results", []):
                paper = {
                    "title": work.get("title", ""),
                    "authors": [author.get("author", {}).get("display_name", "") 
                              for author in work.get("authorships", [])],
                    "year": work.get("publication_year"),
                    "journal": work.get("primary_location", {}).get("source", {}).get("display_name", ""),
                    "doi": work.get("doi", ""),
                    "url": work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else ""
                }
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"文献搜索失败: {str(e)}")
            return []
    
    def papers_to_str(self, papers: list) -> str:
        """将论文列表格式化为字符串"""
        if not papers:
            return "未找到相关论文"
        
        result = "找到以下相关论文：\n"
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper["authors"][:3])  # 最多显示3个作者
            result += f"{i}. {paper['title']}\n"
            result += f"   作者: {authors}\n"
            result += f"   期刊: {paper['journal']} ({paper['year']})\n"
            if paper['doi']:
                result += f"   DOI: {paper['doi']}\n"
            result += "\n"
        
        return result
```

#### 5. WriterAgent 中的工具调用

```python
# app/core/agents/writer_agent.py
class WriterAgent(Agent):
    async def run(self, prompt: str) -> WriterResponse:
        # 调用 LLM
        response = await self.model.chat(
            history=self.chat_history,
            tools=writer_tools,
            tool_choice="auto"
        )
        
        if (hasattr(response.choices[0].message, "tool_calls") 
            and response.choices[0].message.tool_calls):
            
            tool_call = response.choices[0].message.tool_calls[0]
            tool_id = tool_call.id
            
            if tool_call.function.name == "search_papers":
                # 解析搜索查询
                query = json.loads(tool_call.function.arguments)["query"]
                
                # 发布消息到前端
                await redis_manager.publish_message(
                    self.task_id,
                    WriterMessage(input={"query": query})
                )
                
                # 添加助手响应
                await self.append_chat_history(
                    response.choices[0].message.model_dump()
                )
                
                # 执行搜索
                papers = await self.scholar.search_papers(query)
                papers_str = self.scholar.papers_to_str(papers)
                
                # 添加工具响应
                await self.append_chat_history({
                    "role": "tool",
                    "content": papers_str,
                    "tool_call_id": tool_id,
                    "name": "search_papers"
                })
                
                # 继续对话获取最终响应
                next_response = await self.model.chat(
                    history=self.chat_history,
                    tools=writer_tools,
                    tool_choice="auto"
                )
                
                return WriterResponse(
                    response_content=next_response.choices[0].message.content
                )
        else:
            # 直接返回响应
            return WriterResponse(
                response_content=response.choices[0].message.content
            )
```

### 工具调用的消息流程

#### 完整的工具调用对话示例

```python
# 1. 用户请求
{
    "role": "user",
    "content": "请执行代码分析数据"
}

# 2. LLM 决定使用工具
{
    "role": "assistant",
    "content": null,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "execute_code",
                "arguments": '{"code": "import pandas as pd\\ndf = pd.read_csv(\'data.csv\')\\nprint(df.head())"}'
            }
        }
    ]
}

# 3. 工具执行结果
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "name": "execute_code",
    "content": "   name  age  salary\n0  Alice   25   50000\n1    Bob   30   60000\n..."
}

# 4. LLM 基于结果继续对话
{
    "role": "assistant", 
    "content": "数据已成功加载，包含姓名、年龄和薪资信息..."
}
```

### 关键技术特点

#### 1. 标准协议兼容
- 完全遵循 OpenAI Function Calling 协议
- 与主流 LLM 模型兼容 (GPT-4, Claude, 等)
- 通过 LiteLLM 支持多种模型

#### 2. 自主实现执行层
- **不依赖外部工具框架** (如 LangChain Tools)
- **完全自主控制** 工具的执行逻辑和错误处理
- **高度定制化** 的工具行为

#### 3. 异步执行架构
```python
# 所有工具调用都是异步的
async def execute_code(self, code: str) -> tuple[str, bool, str]:
    # 异步执行，支持长时间运行的代码
    pass

async def search_papers(self, query: str) -> list:
    # 异步网络请求
    pass
```

#### 4. 完善的错误处理
```python
# CoderAgent 中的错误处理和重试
if error_occurred:
    retry_count += 1
    reflection_prompt = get_reflection_prompt(error_message, code)
    await self.append_chat_history({
        "role": "user", 
        "content": reflection_prompt
    })
    continue  # 重试
```

#### 5. 实时状态反馈
```python
# 向前端发布工具调用状态
await redis_manager.publish_message(
    self.task_id,
    InterpreterMessage(input={"code": code})
)
```

### 工具扩展机制

#### 添加新工具的步骤

**1. 定义工具 Schema**：
```python
new_tool = {
    "type": "function",
    "function": {
        "name": "new_tool_name",
        "description": "工具描述",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "参数描述"
                }
            },
            "required": ["param1"]
        }
    }
}
```

**2. 实现工具执行器**：
```python
class NewToolExecutor:
    async def execute(self, param1: str) -> str:
        # 实现具体逻辑
        pass
```

**3. 在 Agent 中集成**：
```python
if tool_call.function.name == "new_tool_name":
    param1 = json.loads(tool_call.function.arguments)["param1"]
    result = await new_tool_executor.execute(param1)
    # 处理结果...
```

### 与框架方案对比

| 方案 | 实现方式 | 优势 | 劣势 |
|------|----------|------|------|
| **MathModelAgent** | 自主实现 | ✅ 完全控制<br>✅ 高度定制<br>✅ 无框架依赖 | ❌ 开发量大<br>❌ 需要维护 |
| **LangChain Tools** | 框架提供 | ✅ 开箱即用<br>✅ 生态丰富 | ❌ 框架耦合<br>❌ 定制受限 |
| **AutoGen Tools** | 框架集成 | ✅ 多Agent支持 | ❌ 学习成本<br>❌ 复杂度高 |

### 总结

MathModelAgent 的工具调用实现体现了**自主可控**的设计哲学：

1. **协议标准化**: 遵循 OpenAI Function Calling 标准
2. **执行自主化**: 完全自主实现工具执行逻辑  
3. **错误智能化**: 基于错误信息的智能重试机制
4. **反馈实时化**: 实时向前端反馈执行状态

这种实现方式虽然开发量较大，但获得了**完全的控制权和定制能力**，特别适合像数学建模这样有特定需求的专业领域应用。