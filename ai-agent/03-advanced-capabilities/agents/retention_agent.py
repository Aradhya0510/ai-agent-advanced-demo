"""
Retention Agent - Specialized for customer retention and satisfaction
"""
import mlflow
import sys
import os

# Add path to access base agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '02-agent-eval'))

from agent import LangGraphResponsesAgent
from mlflow.models import ModelConfig


class RetentionAgent(LangGraphResponsesAgent):
    """Specialized agent for customer retention and churn prevention"""
    
    def __init__(self, catalog: str, schema: str):
        # Construct config path
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'configs', 
            'retention_agent_config.yaml'
        )
        
        # Load retention-specific configuration
        try:
            config = ModelConfig(development_config=config_path)
        except:
            # Fallback to default config if file not found
            import yaml
            config_dict = {
                "uc_tool_names": [
                    f"{catalog}.{schema}.get_customer_by_email",
                    f"{catalog}.{schema}.get_customer_billing_and_subscriptions"
                ],
                "llm_endpoint_name": "databricks-claude-3-7-sonnet",
                "system_prompt": "You are a customer retention specialist. Be empathetic and solution-oriented.",
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
        
        self.agent_name = "Retention Agent"
        self.specialization = "customer_retention"
    
    def get_capabilities(self) -> list[str]:
        """Return list of capabilities this agent handles"""
        return [
            "churn_prevention",
            "cancellation_requests",
            "complaint_resolution",
            "service_upgrades",
            "loyalty_offers",
            "vip_customer_handling",
            "satisfaction_improvement"
        ]

