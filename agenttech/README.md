# Tech QA Agent with Microsoft Docs MCP

这是一个使用LangGraph和Microsoft Docs MCP的技术问答代理，可以查询Microsoft官方文档和本地技术文档。

## 功能特性

- 🔍 **Microsoft Docs MCP**: 查询Microsoft官方文档
- 📚 **本地文档查询**: 查询前端和后端技术文档
- 🤖 **智能工具选择**: 根据问题自动选择合适的工具
- 🚀 **LangGraph集成**: 可在LangGraph Studio中运行

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

1. 创建 `.env` 文件并添加你的Google API密钥：
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

2. 确保技术文档文件夹 `tech_doc/` 存在，包含：
   - `frontend.md` - 前端开发文档
   - `backend.md` - 后端开发文档

## Microsoft Docs MCP 使用说明

### 1. 配置说明

在 `agent_tech_QA_MCP.py` 中，Microsoft Docs MCP的配置如下：

```python
"microsoft.docs.mcp": {
    "autoApprove": [],
    "type": "streamableHttp",
    "url": "https://learn.microsoft.com/api/mcp"
}
```

### 2. 可用功能

Microsoft Docs MCP提供以下功能：
- 查询Microsoft Learn文档
- 搜索技术文章和教程
- 获取代码示例
- 访问官方API文档

### 3. 使用示例

在LangGraph中，你可以这样使用：

```python
# 用户问题示例
"请告诉我Python的异步编程基础"
"如何配置Azure云服务？"
"ASP.NET Core的最佳实践是什么？"
```

## 运行方式

### 方式1：使用LangGraph Studio

```bash
cd aiworkshop
langgraph dev
```

然后在浏览器中访问：http://localhost:2024

### 方式2：直接运行Python

```python
from agent_tech_QA_MCP import graph
from langchain_core.messages import HumanMessage

# 创建消息
messages = [HumanMessage(content="请告诉我Python的异步编程基础")]

# 运行图
result = graph.invoke({"messages": messages})

# 打印结果
for message in result["messages"]:
    print(f"=== {type(message).__name__} ===")
    print(message.content)
```

### 方式3：测试MCP工具

```bash
python test_microsoft_docs.py
```

## 工具优先级

当用户提问时，系统会按以下优先级选择工具：

1. **Microsoft Docs MCP**: 优先查询Microsoft官方文档
2. **本地文档工具**: 查询前端/后端开发相关内容
3. **组合查询**: 结合多个工具的信息

## 故障排除

### 1. MCP连接失败

如果Microsoft Docs MCP连接失败：
- 检查网络连接
- 确认API端点可访问
- 系统会自动降级使用本地工具

### 2. 工具初始化失败

如果工具初始化失败：
- 检查依赖是否正确安装
- 确认Google API密钥有效
- 查看控制台错误信息

### 3. 文档文件缺失

如果本地文档文件缺失：
- 确保 `tech_doc/` 文件夹存在
- 创建 `frontend.md` 和 `backend.md` 文件
- 或修改代码移除本地工具依赖

## 扩展功能

### 添加更多MCP服务器

你可以在配置中添加更多MCP服务器：

```python
"awslabs.aws-documentation-mcp-server": {
    "command": "uvx",
    "args": ["awslabs.aws-documentation-mcp-server@latest"],
    "type": "stdio"
}
```

### 自定义本地工具

你可以添加更多本地工具：

```python
def custom_tool(param: str) -> str:
    """自定义工具"""
    return f"处理结果: {param}"
```

## 注意事项

1. **网络依赖**: Microsoft Docs MCP需要网络连接
2. **API限制**: 注意Microsoft API的使用限制
3. **响应时间**: 网络查询可能需要较长时间
4. **错误处理**: 系统包含完善的错误处理机制

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！ 