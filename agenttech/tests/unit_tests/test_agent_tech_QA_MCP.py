import sys
import types
import asyncio
import importlib


class _DummyLLMWithTools:
    def __init__(self):
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return {"role": "assistant", "content": "ok"}


class _DummyChatGoogleGenerativeAI:
    def __init__(self, *args, **kwargs):
        pass

    def bind_tools(self, tools):
        return _DummyLLMWithTools()


class _DummyMCPClient:
    def __init__(self, *args, **kwargs):
        self.config = args[0] if args else {}

    async def get_tools(self):
        return ["t1", "t2"]


class _DummySystemMessage:
    def __init__(self, content):
        self.content = content


class _DummyMessagesState(dict):
    pass


class _DummyStateGraph:
    def __init__(self, *_args, **_kwargs):
        self.nodes = {}
        self.edges = []

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, src, dst):
        self.edges.append((src, dst))

    def add_conditional_edges(self, *_args, **_kwargs):
        # record a marker for conditional edges
        self.edges.append(("conditional", "edges"))

    def compile(self):
        return {"nodes": self.nodes, "edges": self.edges}


class _DummyToolNode:
    def __init__(self, tools):
        self.tools = tools


def _dummy_tools_condition(*_args, **_kwargs):
    return "END"


def _install_fakes():
    # langchain_core.messages
    mod_lc_core = types.ModuleType("langchain_core")
    mod_lc_core_messages = types.ModuleType("langchain_core.messages")
    mod_lc_core_messages.SystemMessage = _DummySystemMessage
    sys.modules["langchain_core"] = mod_lc_core
    sys.modules["langchain_core.messages"] = mod_lc_core_messages

    # langchain_google_genai
    mod_lc_gem = types.ModuleType("langchain_google_genai")
    mod_lc_gem.ChatGoogleGenerativeAI = _DummyChatGoogleGenerativeAI
    sys.modules["langchain_google_genai"] = mod_lc_gem

    # langgraph.graph
    mod_langgraph = types.ModuleType("langgraph")
    mod_langgraph_graph = types.ModuleType("langgraph.graph")
    mod_langgraph_graph.START = object()
    mod_langgraph_graph.StateGraph = _DummyStateGraph
    mod_langgraph_graph.MessagesState = _DummyMessagesState
    sys.modules["langgraph"] = mod_langgraph
    sys.modules["langgraph.graph"] = mod_langgraph_graph

    # langgraph.prebuilt
    mod_langgraph_prebuilt = types.ModuleType("langgraph.prebuilt")
    mod_langgraph_prebuilt.tools_condition = _dummy_tools_condition
    mod_langgraph_prebuilt.ToolNode = _DummyToolNode
    sys.modules["langgraph.prebuilt"] = mod_langgraph_prebuilt

    # langchain_mcp_adapters.client
    mod_mcp = types.ModuleType("langchain_mcp_adapters")
    mod_mcp_client = types.ModuleType("langchain_mcp_adapters.client")
    mod_mcp_client.MultiServerMCPClient = _DummyMCPClient
    sys.modules["langchain_mcp_adapters"] = mod_mcp
    sys.modules["langchain_mcp_adapters.client"] = mod_mcp_client


def test_import_and_graph_building(monkeypatch):
    _install_fakes()
    module = importlib.import_module("agent_tech_QA_MCP")

    assert hasattr(module, "graph"), "graph 未定义"
    assert hasattr(module, "initialize_tools_sync")
    assert hasattr(module, "assistant")
    assert hasattr(module, "confirm")


def test_create_mcp_client_returns_instance():
    _install_fakes()
    module = importlib.import_module("agent_tech_QA_MCP")
    client = module.create_mcp_client()
    assert client is not None
    assert isinstance(client, _DummyMCPClient)


def test_initialize_tools_sync_sets_globals():
    _install_fakes()
    # 重新加载模块以获取干净的全局状态
    if "agent_tech_QA_MCP" in sys.modules:
        del sys.modules["agent_tech_QA_MCP"]
    module = importlib.import_module("agent_tech_QA_MCP")

    tools, llm_with_tools = module.initialize_tools_sync()
    assert tools == ["t1", "t2"]
    assert llm_with_tools is not None


def test_assistant_and_confirm_invoke_llm():
    _install_fakes()
    if "agent_tech_QA_MCP" in sys.modules:
        del sys.modules["agent_tech_QA_MCP"]
    module = importlib.import_module("agent_tech_QA_MCP")

    # 确保已初始化
    module.initialize_tools_sync()

    state = {"messages": [{"role": "user", "content": "hi"}]}

    out1 = module.assistant(state)
    assert isinstance(out1, dict) and "messages" in out1
    assert len(out1["messages"]) == 1

    out2 = module.confirm(state)
    assert isinstance(out2, dict) and "messages" in out2
    assert len(out2["messages"]) == 1


