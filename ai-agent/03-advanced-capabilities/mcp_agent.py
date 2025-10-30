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
from typing import Annotated, Any, Generator, List, Optional, Sequence, TypedDict, Union

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_mcp import DatabricksMCPClient, DatabricksOAuthClientProvider
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client as connect
from mlflow.models import ModelConfig
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from pydantic import create_model

nest_asyncio.apply()

# Enable MLflow LangChain auto-trace
mlflow.langchain.autolog()


#####################
# MCP Tool Wrapper
#####################

class MCPTool(BaseTool):
    """
    Custom LangChain tool that calls Unity Catalog functions via MCP protocol.
    
    Supports both:
    - Managed MCP servers: Databricks-hosted, uses PAT auth (synchronous)
    - Custom MCP servers: Databricks Apps, uses OAuth M2M auth (asynchronous)
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type,
        server_url: str,
        ws: WorkspaceClient,
        is_custom: bool = False,
    ):
        super().__init__(name=name, description=description, args_schema=args_schema)
        # Store MCP-specific attributes
        object.__setattr__(self, "server_url", server_url)
        object.__setattr__(self, "workspace_client", ws)
        object.__setattr__(self, "is_custom", is_custom)
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool via managed or custom server"""
        if self.is_custom:
            # Use async method for custom MCP servers (OAuth required)
            return asyncio.run(self._run_custom_async(**kwargs))
        else:
            # Use managed MCP server via synchronous call
            mcp_client = DatabricksMCPClient(
                server_url=self.server_url,
                workspace_client=self.workspace_client
            )
            # KEY: call_tool uses MCP JSON-RPC protocol over HTTP
            response = mcp_client.call_tool(self.name, kwargs)
            return "".join([c.text for c in response.content])
    
    async def _run_custom_async(self, **kwargs) -> str:
        """Execute custom MCP tool asynchronously with OAuth"""
        async with connect(
            self.server_url, 
            auth=DatabricksOAuthClientProvider(self.workspace_client)
        ) as (read_stream, write_stream, _):
            # Create an async session with the server and call the tool
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(self.name, kwargs)
                return "".join([c.text for c in response.content])


#####################
# MCP Tool Creation
#####################

# Retrieve tool definitions from a custom MCP server (OAuth required)
async def get_custom_mcp_tools(ws: WorkspaceClient, server_url: str):
    """Get tools from a custom MCP server using OAuth"""
    async with connect(server_url, auth=DatabricksOAuthClientProvider(ws)) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            return tools_response.tools


# Retrieve tool definitions from a managed MCP server
def get_managed_mcp_tools(ws: WorkspaceClient, server_url: str):
    """Get tools from a managed MCP server"""
    mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
    return mcp_client.list_tools()


# Convert an MCP tool definition into a LangChain-compatible tool
def create_langchain_tool_from_mcp(
    mcp_tool, server_url: str, ws: WorkspaceClient, is_custom: bool = False
):
    """Create a LangChain tool from an MCP tool definition"""
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
    
    # Return a configured MCPTool instance
    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        args_schema=args_schema,
        server_url=server_url,
        ws=ws,
        is_custom=is_custom,
    )


# Gather all tools from managed and custom MCP servers into a single list
async def create_mcp_tools(
    ws: WorkspaceClient, 
    managed_server_urls: List[str] = None, 
    custom_server_urls: List[str] = None
) -> List[MCPTool]:
    """
    Create LangChain tools from both managed and custom MCP servers.
    
    Managed servers: Databricks-hosted, use PAT auth (fast, synchronous)
    Custom servers: Databricks Apps, use OAuth M2M auth (async)
    """
    tools = []
    
    if managed_server_urls:
        # Load managed MCP tools (synchronous)
        print(f"🔍 Discovering tools from {len(managed_server_urls)} managed MCP server(s)...")
        for server_url in managed_server_urls:
            try:
                mcp_tools = get_managed_mcp_tools(ws, server_url)
                print(f"   • {server_url}: Found {len(mcp_tools)} tools")
                for mcp_tool in mcp_tools:
                    tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=False)
                    tools.append(tool)
            except Exception as e:
                print(f"   ⚠️  Error loading tools from managed server {server_url}: {e}")
    
    if custom_server_urls:
        # Load custom MCP tools (async)
        print(f"🔍 Discovering tools from {len(custom_server_urls)} custom MCP server(s)...")
        for server_url in custom_server_urls:
            try:
                mcp_tools = await get_custom_mcp_tools(ws, server_url)
                print(f"   • {server_url}: Found {len(mcp_tools)} tools")
                for mcp_tool in mcp_tools:
                    tool = create_langchain_tool_from_mcp(mcp_tool, server_url, ws, is_custom=True)
                    tools.append(tool)
            except Exception as e:
                print(f"   ⚠️  Error loading tools from custom server {server_url}: {e}")
    
    print(f"✅ Total tools loaded: {len(tools)}")
    return tools


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
    
    Supports both managed and custom MCP servers:
    - Managed: Databricks-hosted MCP endpoints (PAT auth)
    - Custom: MCP servers hosted as Databricks Apps (OAuth M2M auth)
    """
    
    def __init__(
        self,
        catalog: str,
        schema: str,
        llm_endpoint_name: str = "databricks-claude-3-7-sonnet",
        system_prompt: Optional[str] = None,
        max_history_messages: Optional[int] = None,
        custom_mcp_server_urls: Optional[List[str]] = None,
    ):
        self.catalog = catalog
        self.schema = schema
        self.llm_endpoint_name = llm_endpoint_name
        self.system_prompt = system_prompt
        self.max_history_messages = max_history_messages or 20  # Default to 20 if None
        
        # Initialize workspace client
        ws = WorkspaceClient()
        
        # Configure managed MCP server URL (catalog.schema endpoint)
        managed_mcp_endpoint = f"{ws.config.host}/api/2.0/mcp/functions/{catalog}/{schema}"
        managed_server_urls = [managed_mcp_endpoint]
        
        print(f"🔌 MCP Agent Configuration:")
        print(f"   Catalog: {catalog}")
        print(f"   Schema: {schema}")
        print(f"   LLM: {llm_endpoint_name}")
        print(f"   Managed MCP servers: {len(managed_server_urls)}")
        print(f"   Custom MCP servers: {len(custom_mcp_server_urls or [])}")
        
        # Discover and create MCP tools (async to support both managed and custom)
        self.mcp_tools = asyncio.run(
            create_mcp_tools(
                ws=ws,
                managed_server_urls=managed_server_urls,
                custom_server_urls=custom_mcp_server_urls,
            )
        )
        
        # Create agent with discovered MCP tools
        llm = ChatDatabricks(endpoint=llm_endpoint_name)
        self.agent = create_mcp_agent(llm, self.mcp_tools, system_prompt)
    
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
    
    def get_resources(self):
        """
        Return all MLflow resources (LLM endpoint + UC functions) used by this agent.
        
        This is required for MLflow model logging to track dependencies.
        """
        from mlflow.models.resources import DatabricksServingEndpoint, DatabricksFunction
        
        resources = [DatabricksServingEndpoint(endpoint_name=self.llm_endpoint_name)]
        
        # Add all UC functions accessed via MCP
        for tool in self.mcp_tools:
            # MCP tool names follow pattern: catalog__schema__function_name
            # We need to extract just the function name and construct: catalog.schema.function_name
            # Example: users__aradhya_chouhan__get_weather_by_city -> users.aradhya_chouhan.get_weather_by_city
            tool_name_parts = tool.name.split('__')
            
            if len(tool_name_parts) >= 3:
                # Extract function name (everything after catalog__schema__)
                function_name = '__'.join(tool_name_parts[2:])
            else:
                # Fallback: use the whole name if pattern doesn't match
                function_name = tool.name
            
            # Construct full qualified name: catalog.schema.function_name
            full_function_name = f"{self.catalog}.{self.schema}.{function_name}"
            resources.append(DatabricksFunction(function_name=full_function_name))
        
        return resources


#####################
# Agent Initialization
#####################

# Only instantiate agent when this file is used as the primary model
# Check if we're being loaded as the main model vs. imported as a library
try:
    _is_primary_model = mlflow.models.model.__mlflow_model__ is None
except:
    _is_primary_model = True

if _is_primary_model:
    try:
        # Load configuration from YAML
        model_config = ModelConfig(development_config="configs/mcp_agent_config.yaml")
        
        # Instantiate MCP agent with support for both managed and custom servers
        AGENT = LangGraphResponsesAgent(
            catalog=model_config.get("catalog"),
            schema=model_config.get("schema"),
            llm_endpoint_name=model_config.get("llm_endpoint_name"),
            system_prompt=model_config.get("system_prompt"),
            max_history_messages=model_config.get("max_history_messages"),
            custom_mcp_server_urls=model_config.get("custom_mcp_server_urls"),
        )
        
        # Register agent with MLflow for inference
        mlflow.models.set_model(AGENT)
    except Exception as e:
        # If instantiation fails (e.g., wrong config path), we're probably being imported
        print(f"⚠️  Could not instantiate MCP agent: {e}")
        pass

