"""
Billing Agent - Specialized for billing, payments, and subscriptions
"""
import mlflow
import sys
import os

# Add path to access base agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '02-agent-eval'))

from agent import LangGraphResponsesAgent
from mlflow.models import ModelConfig


class BillingAgent(LangGraphResponsesAgent):
    """Specialized agent for billing and subscription queries"""
    
    def __init__(self, catalog: str, schema: str):
        # Construct config path
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'configs', 
            'billing_agent_config.yaml'
        )
        
        # Load billing-specific configuration
        try:
            config = ModelConfig(development_config=config_path)
        except:
            # Fallback to default config if file not found
            import yaml
            config_dict = {
                "uc_tool_names": [
                    f"{catalog}.{schema}.get_customer_by_email",
                    f"{catalog}.{schema}.get_customer_billing_and_subscriptions",
                    f"{catalog}.{schema}.calculate_math_expression"
                ],
                "llm_endpoint_name": "databricks-claude-3-7-sonnet",
                "system_prompt": "You are a billing specialist. Be concise and accurate with billing information.",
                "retriever_config": None,
                "max_history_messages": 20
            }
            config = type('obj', (object,), {
                'get': lambda self, key: config_dict.get(key)
            })()
        
        super().__init__(
            uc_tool_names=config.get("uc_tool_names"),
            llm_endpoint_name=config.get("llm_endpoint_name"),
            system_prompt=config.get("system_prompt"),
            retriever_config=config.get("retriever_config"),
            max_history_messages=config.get("max_history_messages")
        )
        
        self.agent_name = "Billing Agent"
        self.specialization = "billing_and_subscriptions"
    
    def get_capabilities(self) -> list[str]:
        """Return list of capabilities this agent handles"""
        return [
            "billing_inquiries",
            "payment_history",
            "subscription_management",
            "invoice_explanations",
            "price_calculations",
            "payment_disputes"
        ]

