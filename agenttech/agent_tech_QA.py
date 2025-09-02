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
    if layer == "frontend":
        with open("./tech_doc/frontend.md", "r", encoding="utf-8") as file:
            return file.read()  # read file from frontend.md
    elif layer == "backend":
        with open("./tech_doc/backend.md", "r", encoding="utf-8") as file:
            return file.read()  # read file from backend.md
    else:
        raise ValueError("Unsupported layer. Only 'frontend' and 'backend' are allowed.")


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
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
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
