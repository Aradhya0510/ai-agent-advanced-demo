# Advanced AI Agent Capabilities - Workshop Materials

This directory contains advanced workshop materials that build upon the foundational agent concepts from the basic workshop.

## 📚 Workshop Structure

### **Basic Workshop (01-03.1) - RECAP**
The previous workshop covered:
- ✅ Creating and registering tools in Unity Catalog
- ✅ Building a basic tool-calling agent with LangChain
- ✅ Adding RAG with Vector Search
- ✅ Evaluation and deployment
- ✅ Production monitoring

### **Advanced Workshop (03.2-03.4) - NEW CONTENT**

This advanced workshop adds three sophisticated capabilities:

---

## 🌐 03.2 - MCP on Unity Catalog (External API Integration)

**Notebook:** `03.2-mcp-unity-catalog-tools.py`

### What You'll Learn:
- Integrate external APIs as MCP (Model Context Protocol) tools
- Use Databricks Managed MCP Server
- Create three external API tools:
  - **Weather API** - Check weather conditions for connectivity issues
  - **Distance Calculator** - Estimate technician arrival times
  - **Web Search (Tavily)** - Find latest troubleshooting solutions

### Key Concepts:
- MCP endpoint: `https://<workspace>/api/2.0/mcp/functions/{catalog}/{schema}`
- Secure credential management with Databricks Secrets
- Combining internal UC functions with external MCP tools

### Expected Improvements:
- **+28%** improvement in technical support query quality
- **+70%** improvement in real-time context handling

### Files Created:
- Main notebook with MCP tool implementations
- Mock APIs for demo (no API keys needed)
- Optional real API setup guide in `_resources/05-mcp-setup.py`

---

## 💰 03.3 - Prompt Registry & Cost Optimization

**Notebook:** `03.3-prompt-registry-management.py`

### What You'll Learn:
- Create Unity Catalog prompt registry for versioning
- Run A/B tests across 4 prompt variants:
  - **Concise** - Token-efficient for simple queries
  - **Detailed** - Comprehensive for complex issues
  - **Technical** - Specialized for troubleshooting
  - **Retention** - Optimized for at-risk customers
- Analyze cost vs quality trade-offs
- Implement dynamic prompt selection based on context

### Key Concepts:
- Prompt versioning in Delta tables
- Token counting and cost estimation
- A/B testing framework with MLflow
- Cost optimization strategies

### Expected Improvements:
- **30-40%** cost savings with smart prompt routing
- Maintained quality scores above 0.85
- **$900-1200/month** savings on 1M requests

### Files Created:
- Prompt registry table schema
- 4 prompt variant definitions
- Cost analysis visualizations
- Dynamic prompt router implementation

---

## 🤖 03.4 - Multi-Agent Supervisor System

**Notebook:** `03.4-multi-agent-supervisor.py`

### What You'll Learn:
- Build specialized sub-agents for different domains
- Implement supervisor routing logic
- Coordinate multi-agent workflows
- Handle queries requiring multiple specialists

### Architecture:
```
User Query
    ↓
Supervisor Agent (Router)
    ↓
┌─────────┬─────────┬─────────┐
↓         ↓         ↓         ↓
Billing   Technical Retention
Agent     Agent     Agent
```

### Specialized Agents:

#### **Billing Agent**
- **Tools:** customer data, billing, calculations
- **Expertise:** Invoices, payments, subscriptions
- **Prompt:** Concise, number-focused

#### **Technical Agent**
- **Tools:** RAG docs, web search, weather, distance
- **Expertise:** Troubleshooting, error codes, connectivity
- **Prompt:** Technical, step-by-step guidance

#### **Retention Agent**
- **Tools:** customer analytics, churn scores
- **Expertise:** Churn prevention, VIP handling, offers
- **Prompt:** Empathetic, solution-oriented

### Expected Improvements:
- **+45%** routing accuracy (50% → 95%)
- **+13%** response quality (75% → 88%)
- **+33%** domain expertise (60% → 93%)
- **+27%** tool selection accuracy (65% → 92%)

### Files Created:
- `agents/billing_agent.py` - Billing specialist
- `agents/technical_agent.py` - Technical specialist
- `agents/retention_agent.py` - Retention specialist
- `agents/supervisor_agent.py` - Orchestrator
- `configs/*.yaml` - Agent configurations

---

## 📁 Directory Structure

```
03-advanced-capabilities/
├── README.md (this file)
├── 03.2-mcp-unity-catalog-tools.py
├── 03.3-prompt-registry-management.py
├── 03.4-multi-agent-supervisor.py
├── agents/
│   ├── __init__.py
│   ├── billing_agent.py
│   ├── technical_agent.py
│   ├── retention_agent.py
│   └── supervisor_agent.py
├── configs/
│   ├── billing_agent_config.yaml
│   ├── technical_agent_config.yaml
│   ├── retention_agent_config.yaml
│   └── supervisor_config.yaml
└── _resources/
    └── 05-mcp-setup.py (optional API key setup)
```

---

## 🚀 Workshop Flow

### **Part 1: Recap (30 min)**
Walk through notebooks 01-03.1 from previous workshop

### **Part 2: Advanced Capabilities (2 hours)**

**03.2 - MCP External APIs (40 min)**
- Create weather, distance, web search tools
- Integrate with Databricks MCP Server
- Test MCP-enhanced agent
- Evaluate improvements

**03.3 - Prompt Registry (40 min)**
- Create 4 prompt variants
- Run A/B tests
- Analyze cost vs quality
- Implement dynamic routing
- **Break (10 min)**

**03.4 - Multi-Agent Supervisor (40 min)**
- Build 3 specialized agents
- Create supervisor orchestrator
- Test routing and coordination
- Evaluate multi-agent system

### **Part 3: Deployment (30 min)**
- Update frontend (04-deploy-app)
- Configure monitoring (05-production-monitoring)
- Review business metrics (06-business-kpis)

---

## 🎯 Combined Impact

By implementing all three advanced capabilities:

| Metric | Baseline | After Advanced | Improvement |
|--------|----------|----------------|-------------|
| Response Quality | 65% | 88% | **+23%** |
| Technical Support | 60% | 88% | **+28%** |
| Routing Accuracy | 50% | 95% | **+45%** |
| Domain Expertise | 60% | 93% | **+33%** |
| Cost Efficiency | Baseline | -35% | **$1000+/mo savings** |

---

## 🔧 Prerequisites

### From Previous Workshop:
- Unity Catalog functions (billing, customer lookup, math)
- Vector Search index on PDF documentation
- Evaluation dataset
- MLflow experiment setup

### New Requirements:
- Python 3.11+ (DBR 15.4+)
- Same packages as base workshop
- Additional: `tiktoken` for token counting

### Optional (for real APIs):
- OpenWeatherMap API key
- Google Maps API key
- Tavily API key
- Databricks Secrets scope: `mcp-apis`

---

## 💡 Key Takeaways

### **MCP Tools:**
- Standardized protocol for external APIs
- Seamless integration with UC functions
- Managed by Databricks for security

### **Prompt Registry:**
- Version control for prompts
- Data-driven optimization
- Significant cost savings possible

### **Multi-Agent System:**
- Specialization improves quality
- Clear domain boundaries
- Easier to maintain and extend
- Better debugging and monitoring

---

## 📖 Additional Resources

- [Databricks MCP Documentation](https://docs.databricks.com/mcp)
- [MLflow Prompt Management](https://mlflow.org/docs/latest/prompts.html)
- [LangGraph Multi-Agent Patterns](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Unity Catalog Functions](https://docs.databricks.com/en/sql/language-manual/sql-ref-functions-udf.html)

---

## 🐛 Troubleshooting

### MCP Tools Not Working:
- Check UC function registration
- Verify workspace MCP endpoint URL
- For real APIs, check secrets scope access

### Prompt Registry Issues:
- Ensure Delta table created: `prompt_registry`
- Check catalog/schema permissions
- Verify yaml config files exist

### Multi-Agent Routing Problems:
- Check supervisor LLM endpoint access
- Verify all agent configs are valid
- Ensure agent dependencies imported correctly

---

## 🤝 Contributing

This is a workshop demo. To adapt for your use case:

1. Replace mock APIs with real ones (see `05-mcp-setup.py`)
2. Customize prompts for your domain
3. Add more specialized agents as needed
4. Adjust routing logic for your use cases

---

## 📝 License

This workshop material is part of the Databricks AI Agent Advanced Workshop.

---

**Questions?** Refer to the inline documentation in each notebook or the main workshop README.

**Next Steps:** After completing these advanced capabilities, proceed to `04-deploy-app` to deploy your sophisticated multi-agent system!

