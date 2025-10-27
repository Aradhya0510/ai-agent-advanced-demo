"""
Technical Agent - Specialized for technical support and troubleshooting
"""
import mlflow
import sys
import os

# Add path to access base agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '02-agent-eval'))

from agent import LangGraphResponsesAgent
from mlflow.models import ModelConfig


class TechnicalAgent(LangGraphResponsesAgent):
    """Specialized agent for technical support and troubleshooting"""
    
    def __init__(self, catalog: str, schema: str):
        # Load technical-specific configuration directly with yaml
        # (Don't use ModelConfig here as it conflicts with supervisor's model_config)
        import yaml
        
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'configs', 
            'technical_agent_config.yaml'
        )
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except:
            # Fallback to default config if file not found
            config_dict = {
                "uc_tool_names": [
                    f"{catalog}.{schema}.web_search_simulation",
                    f"{catalog}.{schema}.get_weather_by_city",
                    f"{catalog}.{schema}.calculate_distance"
                ],
                "llm_endpoint_name": "databricks-claude-3-7-sonnet",
                "system_prompt": "You are a senior technical support engineer. Provide step-by-step troubleshooting guidance.",
                "retriever_config": {
                    "index_name": f"{catalog}.{schema}.knowledge_base_vs_index",
                    "tool_name": "product_technical_docs_retriever",
                    "num_results": 3,
                    "description": "Technical documentation and troubleshooting guides"
                },
                "max_history_messages": 20
            }
        
        # Create simple config object
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
        
        self.agent_name = "Technical Agent"
        self.specialization = "technical_support"
    
    def get_capabilities(self) -> list[str]:
        """Return list of capabilities this agent handles"""
        return [
            "troubleshooting",
            "error_codes",
            "connectivity_issues",
            "device_configuration",
            "firmware_updates",
            "network_problems",
            "router_modem_issues",
            "technician_dispatch"
        ]

