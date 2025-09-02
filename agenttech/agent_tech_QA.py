from langchain_core.messages import SystemMessage,BaseMessage,ToolMessage
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
import os
from langchain_core.messages import SystemMessage,BaseMessage,ToolMessage
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import json

from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool

@tool
def find_document(layer: str) -> str:
    """Finds the tech document based on the layer.

    Args:
        layer: the layer of tech document to find only include "frontend", "backend"
    """
    # TODO: 实现工具的逻辑
    # 目标: 根据 `layer` 参数读取并返回 `./tech_doc/frontend.md` 或 `./tech_doc/backend.md` 的内容。
    # 提示:
    # 1. 检查 `layer` 是否为 "frontend" 或 "backend"。
    # 2. 使用 `with open(...) as file:` 安全地打开文件。
    # 3. 读取文件内容并返回。
    # 4. 如果 `layer` 不支持，抛出 `ValueError`。



tools = [find_document]

# Define LLM with bound tools
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",thinking_budget=0)
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE")
)
llm_with_tools = llm.bind_tools(tools)



# Node
def assistant(state: MessagesState):
    sys_msg = SystemMessage(
        content="""
You will act as a senior [Frontend/Backend] Web Programmer. Should answer the user's question based on the tech document provided.
"""
    )
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def should_continue(state: AgentState):
    # TODO: 实现条件判断逻辑
    # 目标: 根据最新消息是否包含工具调用来决定流程是继续 (continue) 还是结束 (end)。
    # 返回结构 - 字符串 "continue" 或 "end"
    # 具体步骤:
    # 1. 获取 `state["messages"]` 中的最后一条消息。
    # 2. 检查 `last_message.tool_calls` 是否存在或为空。
    # 3. 如果 `tool_calls` 为空，表示没有工具需要执行，返回 "end"。
    # 4. 如果 `tool_calls` 不为空，表示有工具需要执行，返回 "continue"。
    return "continue"
# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    should_continue,
    {"continue": "tools", "end": END},
)
builder.add_edge("tools", "assistant")


graph = builder.compile()
