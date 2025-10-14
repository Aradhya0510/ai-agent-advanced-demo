# Databricks notebook source
# MAGIC %md
# MAGIC ## MCP API Key Setup (Optional)
# MAGIC
# MAGIC This notebook helps you setup API keys for external MCP tools.
# MAGIC
# MAGIC **Note:** The main demo uses simulated/mock APIs. Use this setup only if you want to connect to real external APIs.
# MAGIC
# MAGIC ### Required API Keys (Optional):
# MAGIC
# MAGIC 1. **Weather API** (OpenWeatherMap): https://openweathermap.org/api
# MAGIC 2. **Maps API** (Google Maps Distance Matrix): https://developers.google.com/maps/documentation/distance-matrix
# MAGIC 3. **Web Search** (Tavily): https://tavily.com/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Secrets Scope
# MAGIC
# MAGIC First, create a Databricks secrets scope to store your API keys securely.

# COMMAND ----------

# DBTITLE 1,Create Secrets Scope (Run Once)
# Uncomment to create secrets scope

# from databricks.sdk import WorkspaceClient
# 
# w = WorkspaceClient()
# 
# # Create secrets scope for MCP APIs
# try:
#     w.secrets.create_scope(scope="mcp-apis")
#     print("✅ Created secrets scope: mcp-apis")
# except Exception as e:
#     if "already exists" in str(e):
#         print("ℹ️  Secrets scope 'mcp-apis' already exists")
#     else:
#         raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add API Keys to Secrets
# MAGIC
# MAGIC Use Databricks CLI or UI to add your API keys:
# MAGIC
# MAGIC ### Option 1: Using Databricks CLI
# MAGIC
# MAGIC ```bash
# MAGIC # Install Databricks CLI
# MAGIC pip install databricks-cli
# MAGIC
# MAGIC # Configure CLI
# MAGIC databricks configure --token
# MAGIC
# MAGIC # Add API keys
# MAGIC databricks secrets put --scope mcp-apis --key weather-api-key
# MAGIC databricks secrets put --scope mcp-apis --key maps-api-key
# MAGIC databricks secrets put --scope mcp-apis --key tavily-api-key
# MAGIC ```
# MAGIC
# MAGIC ### Option 2: Using Databricks UI
# MAGIC
# MAGIC 1. Go to Settings → Secrets
# MAGIC 2. Select scope: `mcp-apis`
# MAGIC 3. Add secrets:
# MAGIC    - `weather-api-key`: Your OpenWeatherMap API key
# MAGIC    - `maps-api-key`: Your Google Maps API key
# MAGIC    - `tavily-api-key`: Your Tavily API key

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup
# MAGIC
# MAGIC Test that secrets are accessible:

# COMMAND ----------

# DBTITLE 1,Verify Secrets Access
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    # List secrets in scope (won't show values, just keys)
    secrets = w.secrets.list_secrets(scope="mcp-apis")
    
    print("✅ Secrets scope accessible!")
    print(f"\nFound {len(secrets)} secrets:")
    for secret in secrets:
        print(f"  • {secret.key}")
    
    if len(secrets) == 0:
        print("\n⚠️  No secrets found. Add API keys using CLI or UI (see above)")
    
except Exception as e:
    print(f"❌ Error accessing secrets: {e}")
    print("\n💡 Make sure you've created the 'mcp-apis' scope first")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get API Keys (Example Services)
# MAGIC
# MAGIC ### 1. OpenWeatherMap API
# MAGIC
# MAGIC - Website: https://openweathermap.org/api
# MAGIC - Plan: Free tier (1000 calls/day)
# MAGIC - Steps:
# MAGIC   1. Sign up for free account
# MAGIC   2. Navigate to API keys section
# MAGIC   3. Generate new API key
# MAGIC   4. Add to secrets as `weather-api-key`
# MAGIC
# MAGIC ### 2. Google Maps Distance Matrix API
# MAGIC
# MAGIC - Website: https://developers.google.com/maps/documentation/distance-matrix
# MAGIC - Plan: $200 free credit monthly
# MAGIC - Steps:
# MAGIC   1. Create Google Cloud Project
# MAGIC   2. Enable Distance Matrix API
# MAGIC   3. Create API key with Distance Matrix API enabled
# MAGIC   4. Add to secrets as `maps-api-key`
# MAGIC
# MAGIC ### 3. Tavily Search API
# MAGIC
# MAGIC - Website: https://tavily.com/
# MAGIC - Plan: Free tier (1000 searches/month)
# MAGIC - Steps:
# MAGIC   1. Sign up for free account
# MAGIC   2. Get API key from dashboard
# MAGIC   3. Add to secrets as `tavily-api-key`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Use Free Mock APIs
# MAGIC
# MAGIC If you don't want to setup API keys, the demo notebooks use mock/simulated versions:
# MAGIC
# MAGIC - **Weather**: Simulated responses for common cities
# MAGIC - **Distance**: Calculated estimates for major city pairs  
# MAGIC - **Web Search**: Mock search results for common telco issues
# MAGIC
# MAGIC These work without any API keys and demonstrate the multi-agent architecture!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test API Connection (Optional)
# MAGIC
# MAGIC If you've setup real APIs, test them here:

# COMMAND ----------

# DBTITLE 1,Test Weather API (Uncomment to Test)
# import requests
# 
# try:
#     api_key = w.secrets.get_secret(scope="mcp-apis", key="weather-api-key")
#     
#     # Test API call
#     city = "San Francisco"
#     url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=imperial"
#     
#     response = requests.get(url)
#     response.raise_for_status()
#     
#     data = response.json()
#     print(f"✅ Weather API working!")
#     print(f"   Temperature in {city}: {data['main']['temp']}°F")
#     print(f"   Conditions: {data['weather'][0]['description']}")
#     
# except Exception as e:
#     print(f"❌ Weather API test failed: {e}")

# COMMAND ----------

# DBTITLE 1,Test Tavily Search API (Uncomment to Test)
# from tavily import TavilyClient
# 
# try:
#     api_key = w.secrets.get_secret(scope="mcp-apis", key="tavily-api-key")
#     
#     tavily = TavilyClient(api_key=api_key)
#     
#     # Test search
#     results = tavily.search("router troubleshooting", max_results=2)
#     
#     print("✅ Tavily Search API working!")
#     print(f"   Found {len(results['results'])} results")
#     for i, result in enumerate(results['results'][:2], 1):
#         print(f"   {i}. {result['title']}")
#     
# except Exception as e:
#     print(f"❌ Tavily API test failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **For the demo**: Mock APIs work out of the box - no setup needed!
# MAGIC
# MAGIC 🔧 **For production**: Follow the steps above to setup real API keys for:
# MAGIC - Real-time weather data
# MAGIC - Actual distance/routing calculations
# MAGIC - Live web search results
# MAGIC
# MAGIC All MCP tools use the same Unity Catalog function interface, so switching from mock to real APIs is seamless!

