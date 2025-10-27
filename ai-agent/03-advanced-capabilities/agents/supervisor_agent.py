"""
Supervisor Agent - Orchestrates multiple specialized agents
"""
import mlflow
import sys
import os
from typing import Literal, Optional, Any, Generator

# Add path to access base agent and specialized agents
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '02-agent-eval'))

from langchain_core.messages import HumanMessage, AIMessage
from databricks_langchain import ChatDatabricks
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.entities import SpanType

# Import specialized agents
from billing_agent import BillingAgent
from technical_agent import TechnicalAgent
from retention_agent import RetentionAgent


class SupervisorAgent(ResponsesAgent):
    """
    Supervisor agent that routes queries to specialized sub-agents.
    
    Architecture:
        User Query → Supervisor (classify intent) → Specialist Agent → Response
    """
    
    def __init__(self, catalog: str, schema: str, llm_endpoint: str):
        self.catalog = catalog
        self.schema = schema
        self.llm_endpoint = llm_endpoint
        
        # Initialize router LLM
        self.router_llm = ChatDatabricks(endpoint=llm_endpoint)
        
        # Initialize specialized agents
        print("🤖 Initializing specialized agents...")
        self.billing_agent = BillingAgent(catalog, schema)
        self.technical_agent = TechnicalAgent(catalog, schema)
        self.retention_agent = RetentionAgent(catalog, schema)
        
        self.agents = {
            "billing": self.billing_agent,
            "technical": self.technical_agent,
            "retention": self.retention_agent
        }
        
        print(f"✅ Supervisor initialized with {len(self.agents)} specialist agents")
    
    @mlflow.trace(span_type=SpanType.CHAIN, name="route_query")
    def route_query(self, query: str) -> Literal["billing", "technical", "retention"]:
        """
        Classify query intent and route to appropriate specialist agent.
        
        Args:
            query: User's question
            
        Returns:
            Agent name to route to: "billing", "technical", or "retention"
        """
        routing_prompt = f"""You are a query router for a customer support system. Classify this query into ONE category.

Query: {query}

Categories:
1. billing - Payments, invoices, subscriptions, charges, bills, pricing, refunds
2. technical - Device issues, errors, connectivity, troubleshooting, not working, slow internet, router, modem
3. retention - Cancellations, complaints, dissatisfaction, "cancel service", "terrible service", upgrades

Instructions:
- Consider the primary intent of the query
- If multiple intents, choose the most critical one
- For cancellation/complaint queries, ALWAYS choose "retention"
- For error codes or connectivity, choose "technical"
- For payment/subscription queries, choose "billing"

Respond with ONLY the category name: billing, technical, or retention"""
        
        response = self.router_llm.invoke([HumanMessage(content=routing_prompt)])
        intent = response.content.strip().lower()
        
        # Validate and default to billing if uncertain
        if intent not in ["billing", "technical", "retention"]:
            intent = "billing"  # Safe default
        
        # Log routing decision to trace
        mlflow.update_current_trace(
            tags={"routing_decision": intent}
        )
        
        return intent
    
    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor_predict")
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Main prediction method - routes to specialist and returns response.
        """
        # Extract query from request
        query = request.input[0].content if request.input else ""
        
        # Route to appropriate agent (traced separately)
        selected_agent_name = self.route_query(query)
        selected_agent = self.agents[selected_agent_name]
        
        # Log routing decision to supervisor trace
        mlflow.update_current_trace(
            tags={
                "supervisor.selected_agent": selected_agent_name,
                "supervisor.query": query,
                "supervisor.architecture": "multi_agent"
            }
        )
        
        print(f"🎯 Supervisor routing to: {selected_agent_name.upper()} Agent")
        
        # Create a traced wrapper for specialist call to ensure it's captured
        @mlflow.trace(span_type=SpanType.AGENT, name=f"specialist_{selected_agent_name}_call")
        def call_specialist():
            return selected_agent.predict(request)
        
        # Delegate to specialist agent (will be traced as child span)
        specialist_response = call_specialist()
        
        # Add supervisor metadata to response
        if hasattr(specialist_response, 'custom_outputs'):
            if specialist_response.custom_outputs is None:
                specialist_response.custom_outputs = {}
            specialist_response.custom_outputs['routed_to'] = selected_agent_name
            specialist_response.custom_outputs['supervisor'] = True
        
        return specialist_response
    
    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor_predict_stream")
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Streaming prediction - routes to specialist agent's stream.
        """
        query = request.input[0].content if request.input else ""
        
        # Route to appropriate agent
        selected_agent_name = self.route_query(query)
        selected_agent = self.agents[selected_agent_name]
        
        print(f"🎯 Supervisor streaming from: {selected_agent_name.upper()} Agent")
        
        # Stream from specialist agent
        yield from selected_agent.predict_stream(request)
    
    def get_resources(self):
        """
        Aggregate all resources needed by sub-agents.
        """
        all_resources = []
        for agent in self.agents.values():
            all_resources.extend(agent.get_resources())
        
        # Deduplicate resources
        unique_resources = []
        seen = set()
        for resource in all_resources:
            resource_key = f"{type(resource).__name__}:{resource.name}"
            if resource_key not in seen:
                seen.add(resource_key)
                unique_resources.append(resource)
        
        return unique_resources
    
    @mlflow.trace(span_type=SpanType.AGENT, name="supervisor_multi_agent_predict")
    def predict_multi_agent(self, request) -> dict[str, Any]:
        """
        Advanced: Handle queries requiring multiple agents.
        
        Example: "My bill is high and my internet is slow"
            → Query billing agent AND technical agent
            → Aggregate both responses
        """
        # Handle both dict and ResponsesAgentRequest inputs
        if isinstance(request, dict):
            query = request.get("input", [{}])[0].get("content", "")
        else:
            query = request.input[0].content if request.input else ""
        
        # Detect if multiple agents needed (simplified logic)
        needs_billing = any(kw in query.lower() for kw in ['bill', 'payment', 'charge', 'subscription'])
        needs_technical = any(kw in query.lower() for kw in ['error', 'not working', 'slow', 'broken'])
        needs_retention = any(kw in query.lower() for kw in ['cancel', 'terrible', 'frustrated', 'complaint'])
        
        agents_needed = []
        if needs_billing:
            agents_needed.append("billing")
        if needs_technical:
            agents_needed.append("technical")
        if needs_retention:
            agents_needed.append("retention")
        
        # Log multi-agent coordination to trace
        mlflow.update_current_trace(
            tags={
                "supervisor.multi_agent": True,
                "supervisor.agents_needed": ",".join(agents_needed),
                "supervisor.num_agents": len(agents_needed)
            }
        )
        
        # If multiple agents needed, coordinate them
        if len(agents_needed) > 1:
            print(f"🔀 Multi-agent query detected: {', '.join(agents_needed)}")
            
            responses = {}
            for agent_name in agents_needed:
                agent = self.agents[agent_name]
                
                # Create traced wrapper for each specialist call
                @mlflow.trace(span_type=SpanType.AGENT, name=f"multi_agent_specialist_{agent_name}_call")
                def call_agent(agent_instance, req):
                    # Convert dict to ResponsesAgentRequest if needed
                    if isinstance(req, dict):
                        from mlflow.types.responses import ResponsesAgentRequest
                        agent_request = ResponsesAgentRequest(
                            input=req.get("input", [])
                        )
                        return agent_instance.predict(agent_request)
                    else:
                        return agent_instance.predict(req)
                
                response = call_agent(agent, request)
                responses[agent_name] = response.output[-1].content[-1]['text']
            
            # Aggregate responses
            aggregated = "Based on multiple specialist consultations:\n\n"
            for agent_name, response_text in responses.items():
                aggregated += f"**{agent_name.title()} Team:**\n{response_text}\n\n"
            
            return {
                "output": [{
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": aggregated
                    }]
                }],
                "multi_agent": True,
                "agents_used": list(agents_needed)
            }
        else:
            # Single agent route
            if isinstance(request, dict):
                from mlflow.types.responses import ResponsesAgentRequest
                agent_request = ResponsesAgentRequest(
                    input=request.get("input", [])
                )
                response = self.predict(agent_request)
            else:
                response = self.predict(request)
            # Convert ResponsesAgentResponse to dict format
            output_items = []
            for item in response.output:
                output_dict = item.model_dump()
                output_items.append(output_dict)
            return {
                "output": output_items,
                "multi_agent": False,
                "agents_used": []
            }


# Load configuration values from YAML
# This runs at module level so the agent is available when Model Serving imports it
import yaml

try:
    with open('./configs/supervisor_config.yaml', 'r') as f:
        supervisor_config = yaml.safe_load(f)
except:
    # Fallback for build job
    supervisor_config = {}

model_config = mlflow.models.ModelConfig(development_config='./configs/supervisor_config.yaml')

# Instantiate supervisor agent
# Note: For deployment, specialist agent configs must be in the configs/ directory
SUPERVISOR = SupervisorAgent(
    catalog=model_config.get("catalog"),
    schema=model_config.get("schema"),
    llm_endpoint=model_config.get("llm_endpoint_name")
)

# Register agent with MLflow for inference (Models-from-Code pattern)
mlflow.models.set_model(SUPERVISOR)
