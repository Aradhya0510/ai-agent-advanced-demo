"""
MCP Agent - Tool-calling agent using Model Context Protocol

This module demonstrates real MCP (Model Context Protocol) client-server architecture
for calling Unity Catalog functions via standardized protocol.

Key Differences from Direct UC Pattern (agent.py):
- Uses DatabricksMCPClient instead of UCFunctionToolkit
- Tool discovery via mcp_client.list_tools() (dynamic)
- Tool execution via MCP JSON-RPC protocol over HTTP
- Enables network boundaries and interoperability

Author: Databricks
"""

import asyncio
from typing import Annotated, Any, Generator, List, Sequence, Optional, TypedDict

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_mcp import DatabricksMCPClient
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from pydantic import create_model


#####################
# Configuration
#####################

LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
CATALOG = "main"
SCHEMA = "dbdemos_ai_agent"

system_prompt = """You are an expert telco support assistant with access to internal systems and external APIs via MCP protocol.

TOOL USAGE GUIDELINES:
- Use get_customer_by_email and get_customer_billing_and_subscriptions for customer data
- Use get_weather_by_city when customer reports connectivity issues (weather may be a factor)
- Use calculate_distance to estimate technician arrival times for on-site support
- Use web_search_simulation for latest troubleshooting solutions not in documentation
- Use calculate_math_expression for any calculations

RESPONSE GUIDELINES:
- Be professional, concise, and helpful
- Cite sources when using web search results
- Don't mention internal tool names or reasoning steps to the customer
- Proactively offer relevant information"""


#####################
# MCP Tool Wrapper
#####################

class MCPTool(BaseTool):
    """
    Custom LangChain tool that calls Unity Catalog functions via MCP protocol.
    
    This is the KEY difference from direct UC calling:
    - Direct UC: Python SDK calls UC function directly
    - MCP: Client connects to MCP server endpoint via HTTP, uses JSON-RPC protocol
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type,
        server_url: str,
        ws: WorkspaceClient,
    ):
        super().__init__(name=name, description=description, args_schema=args_schema)
        # Store MCP-specific attributes
        object.__setattr__(self, "server_url", server_url)
        object.__setattr__(self, "workspace_client", ws)
    
    def _run(self, **kwargs) -> str:
        """
        Execute tool via MCP protocol (not direct UC call!).
        
        This creates an MCP client and calls the tool via JSON-RPC protocol.
        """
        mcp_client = DatabricksMCPClient(
            server_url=self.server_url,
            workspace_client=self.workspace_client
        )
        
        # KEY: call_tool uses MCP JSON-RPC protocol over HTTP
        # Not a direct Python SDK call to UC function
        response = mcp_client.call_tool(self.name, kwargs)
        return "".join([c.text for c in response.content])


#####################
# MCP Tool Creation
#####################

def create_langchain_tool_from_mcp(mcp_tool, server_url: str, ws: WorkspaceClient):
    """
    Convert an MCP tool definition into a LangChain-compatible tool.
    
    This dynamically creates a Pydantic schema from the MCP tool's JSON schema.
    """
    schema = mcp_tool.inputSchema.copy()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Map JSON schema types to Python types for input validation
    TYPE_MAPPING = {"integer": int, "number": float, "boolean": bool}
    field_definitions = {}
    
    for field_name, field_info in properties.items():
        field_type_str = field_info.get("type", "string")
        field_type = TYPE_MAPPING.get(field_type_str, str)
        
        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (field_type, None)
    
    # Dynamically create a Pydantic schema for the tool's input arguments
    args_schema = create_model(f"{mcp_tool.name}Args", **field_definitions)
    
    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        args_schema=args_schema,
        server_url=server_url,
        ws=ws,
    )


def create_mcp_tools(ws: WorkspaceClient, mcp_endpoint: str) -> List[MCPTool]:
    """
    Discover and create all MCP tools from the Databricks MCP server endpoint.
    
    KEY DIFFERENCE from Direct UC:
    - Direct UC: Hardcoded list in config: uc_tool_names = ["catalog.schema.function1", ...]
    - MCP: Dynamic discovery via list_tools() - no hardcoding needed!
    """
    # Create MCP client
    mcp_client = DatabricksMCPClient(
        server_url=mcp_endpoint,
        workspace_client=ws
    )
    
    # Tool DISCOVERY via MCP protocol (HTTP call to MCP server)
    available_tools = mcp_client.list_tools()
    
    print(f"🔍 Discovered {len(available_tools)} tools via MCP protocol:")
    for tool in available_tools:
        print(f"   • {tool.name}")
    
    # Convert all discovered MCP tools to LangChain tools
    mcp_tools = []
    for mcp_tool in available_tools:
        langchain_tool = create_langchain_tool_from_mcp(
            mcp_tool,
            mcp_endpoint,
            ws
        )
        mcp_tools.append(langchain_tool)
    
    return mcp_tools


#####################
# Agent State & Logic
#####################

class AgentState(TypedDict):
    """State for the agent workflow, including conversation history"""
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_mcp_agent(llm, mcp_tools, system_prompt):
    """
    Create a LangGraph agent with MCP tools.
    
    This uses the same LangGraph pattern as agent.py, but with MCP tools instead.
    """
    llm = llm.bind_tools(mcp_tools)
    
    def should_continue(state: AgentState):
        """Check if agent should continue calling tools or finish"""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"
    
    # Preprocess: optionally prepend system prompt
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    
    model_runnable = preprocessor | llm
    
    def call_model(state: AgentState, config: RunnableConfig):
        """Invoke the model within the workflow"""
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}
    
    # Create the agent's state machine
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(mcp_tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


#####################
# ResponsesAgent Wrapper
#####################

class LangGraphResponsesAgent(ResponsesAgent):
    """
    Wrapper to make LangGraph agent compatible with Mosaic AI Responses API.
    
    This enables the agent to work with MLflow deployment, evaluation, and monitoring.
    """
    
    def __init__(self, agent):
        self.agent = agent
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Make a prediction (single-step) for the agent"""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done" or event.type == "error"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
    
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream predictions for the agent, yielding output as it's generated"""
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        
        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                # Stream updated messages from the workflow nodes
                for node_data in event[1].values():
                    if len(node_data.get("messages", [])) > 0:
                        yield from output_to_responses_items_stream(node_data["messages"])
            elif event[0] == "messages":
                # Stream generated text message chunks
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except:
                    pass


#####################
# Agent Initialization
#####################

def initialize_mcp_agent():
    """
    Initialize the MCP agent with tools from Databricks MCP server.
    
    This is called when the module is loaded to create the AGENT instance.
    """
    ws = WorkspaceClient()
    
    # Construct MCP endpoint URL
    mcp_endpoint = f"{ws.config.host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"
    
    print(f"🔌 Connecting to MCP Server: {mcp_endpoint}")
    
    # Discover and create MCP tools
    mcp_tools = create_mcp_tools(ws, mcp_endpoint)
    
    print(f"✅ Loaded {len(mcp_tools)} tools via MCP protocol")
    
    # Create agent with MCP tools
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
    agent = create_mcp_agent(llm, mcp_tools, system_prompt)
    
    return LangGraphResponsesAgent(agent)


# Enable MLflow LangChain auto-trace
mlflow.langchain.autolog()

# Initialize agent when module is loaded
AGENT = initialize_mcp_agent()

# Set as MLflow model
mlflow.models.set_model(AGENT)

