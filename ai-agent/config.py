# Databricks notebook source
# MAGIC %md
# MAGIC ## Configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2Fconfig&demo_name=ai-agent&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fai-agent%2Fconfig&version=1&user_hash=849103abf30cff95ada66078de7ea959dddec3173bd69b9ad64853593258328d">

# COMMAND ----------

#Note: we do not recommend to change the catalog here as it won't impact all the demo resources such as DLT pipeline and Dashboards.
#Instead, please re-install the demo with a specific catalog and schema using dbdemos.install("lakehouse-retail-c360", catalog="..", schema="...")

catalog = "main"
schema = dbName = db = "dbdemos_ai_agent"

volume_name = "raw_data"
VECTOR_SEARCH_ENDPOINT_NAME="dbdemos_vs_endpoint"

MODEL_NAME = "dbdemos_ai_agent_demo"
ENDPOINT_NAME = f'{MODEL_NAME}_{catalog}_{db}'[:60]

# This must be a tool-enabled model
LLM_ENDPOINT_NAME = 'databricks-claude-3-7-sonnet'

# Advanced capabilities configurations
# MCP (Model Context Protocol) configurations
MCP_ENDPOINT = f"https://{{workspace_host}}/api/2.0/mcp/functions/{catalog}/{dbName}"
MCP_SECRETS_SCOPE = "mcp-apis"  # Scope for storing external API keys

# Multi-agent configurations
SUPERVISOR_MODEL_NAME = f"{MODEL_NAME}_supervisor"
SUPERVISOR_ENDPOINT_NAME = f'{SUPERVISOR_MODEL_NAME}_{catalog}_{db}'[:60]

# Specialized agent names
BILLING_AGENT_NAME = "billing_agent"
TECHNICAL_AGENT_NAME = "technical_agent"
RETENTION_AGENT_NAME = "retention_agent"

# Prompt registry table
PROMPT_REGISTRY_TABLE = f"{catalog}.{dbName}.prompt_registry"