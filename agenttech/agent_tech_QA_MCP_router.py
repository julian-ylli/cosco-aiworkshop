import asyncio
import os
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
import requests
# 创建MCP客户端
class MessagesStateWithVerbosity(MessagesState):
    verbosity: str
def create_mcp_client():
    """创建MCP客户端"""
    try:
        client = MultiServerMCPClient(
            {
                "read_arch_doc": {
                    "command": "python",
                    "args": ["mcp_server.py"],
                    # "args": ["mcp_from_scratch.py"],
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
llm_with_tools_pro = None
async def initialize_tools():
    """初始化工具和LLM"""
    global tools, llm_with_tools
    
    # 获取MCP工具
    mcp_tools = await client.get_tools()
    
    # 合并工具列表
    tools = mcp_tools
    print(f"📦 总工具数量: {len(tools)}")
    
    # 定义LLM并绑定工具
    llm = ChatOpenAI(
        model="gemini-2.0-flash",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE")
    )
    llm_with_tools = llm.bind_tools(tools)
    llm_pro = ChatOpenAI(
        model="gemini-2.5-flash",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE")
    )
    llm_with_tools_pro = llm_pro.bind_tools(tools)
    return tools, llm_with_tools, llm_with_tools_pro

# 同步初始化函数（用于langgraph dev）
def initialize_tools_sync():
    """同步版本的初始化函数"""
    global tools, llm_with_tools, llm_with_tools_pro
    if not tools:
        # try:
        tools, llm_with_tools, llm_with_tools_pro = asyncio.run(initialize_tools())
        # except Exception as e:
        #     print(f"❌ 初始化工具失败: {e}")
            # 使用本地工具作为备用
    return tools, llm_with_tools, llm_with_tools_pro


# Node
def router(state: MessagesStateWithVerbosity):
    # forward the message to classification model deployed on localhost:8080/invocations
    response = requests.post("http://localhost:8080/invocations", json={"inputs": state["messages"][-1].content})
    return {"verbosity": response.json()["predictions"][0]["label"]}

def assistant(state: MessagesStateWithVerbosity):
    # 确保工具已初始化
    if llm_with_tools is None:
        initialize_tools_sync()
    
    sys_msg = SystemMessage(
        content="""
You will act as a [Frontend/Backend] Web Programmer. Should answer the user's question based on the tech document provided.
"""
    )
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def senior_assistant(state: MessagesStateWithVerbosity):
    # 确保工具已初始化
    if llm_with_tools_pro is None:
        initialize_tools_sync()
    
    sys_msg = SystemMessage(
        content="""
You will act as a senior [Frontend/Backend] Web Programmer. Should answer the user's question based on the tech document provided.
You are more experienced and can provide more detailed and comprehensive answers.
"""
    )
    return {"messages": [llm_with_tools_pro.invoke([sys_msg] + state["messages"])]}

def route_to_assistant(state: MessagesStateWithVerbosity):
    """根据verbosity路由到不同的assistant"""
    verbosity = state.get("verbosity", "low")
    if verbosity == "LABEL_1":
        return "senior_assistant"
    elif verbosity == "LABEL_0":
        return "assistant"
    else:
        raise ValueError(f"Unknown verbosity: {verbosity}")

# 初始化工具
initialize_tools_sync()

# Graph

builder = StateGraph(MessagesStateWithVerbosity)

# Define nodes: these do the work
builder.add_node("router", router)
builder.add_node("assistant", assistant)
builder.add_node("senior_assistant", senior_assistant)
builder.add_node("tools", ToolNode(tools))
tools_pro = tools.copy()
builder.add_node("tools_pro", ToolNode(tools_pro))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_to_assistant,
    {
        "assistant": "assistant",
        "senior_assistant": "senior_assistant"
    }
)

# 为assistant添加条件边
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

# 为senior_assistant添加条件边
builder.add_conditional_edges(
    "senior_assistant",
    tools_condition,
    {"tools": "tools_pro", "__end__": END}
)
builder.add_edge("tools", "assistant")
builder.add_edge("tools_pro", "senior_assistant")

graph = builder.compile()

# 为了在langgraph dev中运行，需要导出graph
__all__ = ["graph"]
