import asyncio
import os
import subprocess
from langchain_core.messages import SystemMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
# 创建MCP客户端
def create_mcp_client():
    """创建MCP客户端"""
    try:
        client = MultiServerMCPClient(
            {
                "read_arch_doc": {
                    "command": "python",
                    # "args": ["mcp_server.py"],
                    "args": ["mcp_from_scratch.py"],
                    "transport": "stdio",
                },
                "microsoft.docs.mcp": {
                    "transport": "streamable_http",
                    "url": "https://learn.microsoft.com/api/mcp",
                },
                "awslabs.aws-documentation-mcp-server": {
                    "command": "uvx",
                    "args": ["awslabs.aws-documentation-mcp-server@latest"],
                    "transport": "streamable_http",
                    "transport": "stdio"
                }
            }
        )
        print("✅ MCP客户端创建成功")
        return client
    except Exception as e:
        print(f"❌ 创建MCP客户端失败: {e}")
        print("💡 这可能是由于事件循环冲突导致的，但不会影响基本功能")
        return None

client = create_mcp_client()


# 全局变量存储工具和LLM
tools = []
llm_with_tools = None

async def initialize_tools():
    """初始化工具和LLM"""
    global tools, llm_with_tools
    
    # 获取MCP工具
    mcp_tools = await client.get_tools()
    
    # 合并工具列表
    # 可以在这里根据需要筛选工具
    # 例如，只选择 'read_document_with_mcp' 和 'microsoft_docs_search'
    # selected_tool_names = ["read_document_with_mcp", "microsoft_docs_search"]
    # tools = [tool for tool in mcp_tools if tool.name in selected_tool_names]
    tools = mcp_tools
    
    # 定义LLM并绑定工具

    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE")
    )
    llm_with_tools = llm.bind_tools(tools)
    
    return tools, llm_with_tools

# 同步初始化函数（用于langgraph dev）
def initialize_tools_sync():
    """同步版本的初始化函数"""
    global tools, llm_with_tools
    if not tools:
        # try:
        tools, llm_with_tools = asyncio.run(initialize_tools())
        # except Exception as e:
        #     print(f"❌ 初始化工具失败: {e}")
            # 使用本地工具作为备用
    return tools, llm_with_tools

# Node
def confirm(state: MessagesState):
    # 确保工具已初始化
    if llm_with_tools is None:
        initialize_tools_sync()
    
    # System message
    sys_msg = SystemMessage(
        content="What ever user asked, you should confirm it by a rhetorical question."
    )

    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Node
def assistant(state: MessagesState):
    # 确保工具已初始化
    if llm_with_tools is None:
        initialize_tools_sync()
    
    sys_msg = SystemMessage(
        content="""
You will act as a senior [Frontend/Backend] Web Programmer. Should answer the user's question based on the tech document provided.
"""
    )
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 初始化工具
initialize_tools_sync()

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

graph = builder.compile()

# 为了在langgraph dev中运行，需要导出graph
__all__ = ["graph"]
