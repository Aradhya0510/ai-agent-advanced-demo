# Databricks AI Agents - Advanced Workshop

Advanced workshop materials for building production-grade AI agent systems on Databricks with MLflow 3.0.

[![Workshop](https://img.shields.io/badge/Workshop-Advanced%20AI%20Agents-blue)](https://github.com/Aradhya0510/ai-agent-advanced-demo)
[![Databricks](https://img.shields.io/badge/Databricks-MLflow%203.0-orange)](https://docs.databricks.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

## 🎯 Overview

This repository contains comprehensive workshop materials for building advanced AI agent systems on Databricks, featuring:

- 🌐 **MCP Integration** - External API tools via Model Context Protocol
- 💰 **Cost Optimization** - Prompt registry with 30-40% cost savings
- 🤖 **Multi-Agent Systems** - Specialized agents with supervisor orchestration
- 📊 **Production Monitoring** - End-to-end MLflow tracking and evaluation
- 🚀 **Deployment** - Lakehouse Apps with Gradio frontend

## 📚 Workshop Structure

### **Part 1: Foundation (Basic Workshop)**

| Notebook | Topic | Duration |
|----------|-------|----------|
| 01 | Tool Creation & UC Registration | 15 min |
| 02 | Agent Building & Evaluation | 20 min |
| 03.1 | RAG with Vector Search | 20 min |

### **Part 2: Advanced Capabilities (NEW)**

| Notebook | Topic | Duration | Key Improvement |
|----------|-------|----------|-----------------|
| 03.2 | MCP External APIs | 40 min | +28% technical support quality |
| 03.3 | Prompt Registry | 40 min | 30-40% cost reduction |
| 03.4 | Multi-Agent Supervisor | 40 min | +22% overall quality |

### **Part 3: Production Deployment**

| Notebook | Topic | Duration |
|----------|-------|----------|
| 04 | Frontend Deployment | 20 min |
| 05 | Production Monitoring | 15 min |
| 06 | Business KPIs Dashboard | 10 min |

## 🚀 Quick Start

### Prerequisites

- Databricks workspace (DBR 15.4+)
- Unity Catalog enabled
- MLflow 3.0+
- Python 3.11+

### Setup

1. Clone this repository to your Databricks workspace:
```bash
git clone https://github.com/Aradhya0510/ai-agent-advanced-demo.git
```

2. Navigate to `ai-agent/01-ai-agent-introduction.py` and follow the workshop progression

3. For advanced capabilities only:
   - Ensure basic setup (01-03.1) is complete
   - Jump to `ai-agent/03-advanced-capabilities/03.2-mcp-unity-catalog-tools.py`

## 📁 Repository Structure

```
ai-agent/
├── 01-ai-agent-introduction.py          # Workshop overview
├── 01-create-tools/                      # UC function creation
├── 02-agent-eval/                        # Agent building & evaluation
│   ├── agent.py                         # LangGraph agent implementation
│   └── agent_config.yaml                # Agent configuration
├── 03-knowledge-base-rag/               # RAG with Vector Search
├── 03-advanced-capabilities/            # 🆕 ADVANCED CONTENT
│   ├── 03.2-mcp-unity-catalog-tools.py # External API integration
│   ├── 03.3-prompt-registry-management.py # Cost optimization
│   ├── 03.4-multi-agent-supervisor.py  # Multi-agent orchestration
│   ├── agents/                          # Specialized agent implementations
│   │   ├── billing_agent.py
│   │   ├── technical_agent.py
│   │   ├── retention_agent.py
│   │   └── supervisor_agent.py
│   ├── configs/                         # Agent configurations
│   └── README.md                        # Advanced capabilities guide
├── 04-deploy-app/                       # Lakehouse Apps deployment
├── 05-production-monitoring/            # MLflow monitoring
├── 06-improving-business-kpis/          # Business metrics
└── config.py                            # Global configuration
```

## 🎓 Learning Objectives

### Basic Workshop
- ✅ Create and register tools in Unity Catalog
- ✅ Build tool-calling agents with LangChain
- ✅ Add RAG capabilities with Vector Search
- ✅ Evaluate and deploy agents with MLflow

### Advanced Workshop
- ✅ Integrate external APIs via MCP
- ✅ Optimize prompts for cost and quality
- ✅ Build multi-agent systems with specialization
- ✅ Monitor production agent performance

## 🔑 Key Features

### 1. MCP External API Integration (03.2)
- Weather API for network troubleshooting
- Distance Calculator for technician dispatch
- Web Search (Tavily) for latest solutions
- Databricks Managed MCP Server integration

### 2. Prompt Registry & Optimization (03.3)
- Unity Catalog prompt versioning
- A/B testing framework
- Cost vs quality analysis
- Dynamic prompt selection

### 3. Multi-Agent Supervisor (03.4)
- **Billing Agent** - Payments & subscriptions
- **Technical Agent** - Troubleshooting & support
- **Retention Agent** - Churn prevention
- **Supervisor** - Intelligent routing & orchestration

## 📊 Expected Improvements

| Metric | Baseline | With Advanced | Improvement |
|--------|----------|---------------|-------------|
| Response Quality | 65% | 88% | **+23%** |
| Technical Support | 60% | 88% | **+28%** |
| Routing Accuracy | 50% | 95% | **+45%** |
| Monthly Cost | $3,000 | $1,950 | **-35%** |

## 🛠️ Technologies

- **Platform:** Databricks (Unity Catalog, Vector Search, Model Serving, Lakehouse Apps)
- **ML Framework:** MLflow 3.0 (tracing, evaluation, deployment)
- **Agent Framework:** LangChain + LangGraph
- **LLM:** Claude 3.7 Sonnet (`databricks-claude-3-7-sonnet`)
- **Embeddings:** `databricks-gte-large-en`
- **Frontend:** Gradio + FastAPI
- **Data:** Delta Lake with Change Data Feed

## 📖 Documentation

- **Main README:** This file
- **Advanced Capabilities:** [03-advanced-capabilities/README.md](ai-agent/03-advanced-capabilities/README.md)
- **Databricks Docs:** [MLflow 3.0 GenAI](https://docs.databricks.com/en/mlflow3/genai/)
- **MCP Protocol:** [Model Context Protocol](https://modelcontextprotocol.io/)

## 🤝 Workshop Format

### Audience
- Data scientists and ML engineers
- Previous basic workshop completion recommended
- Familiar with LLMs and prompt engineering

### Prerequisites
- Databricks workspace access
- Basic Python knowledge
- Understanding of LLMs and agents (or completed basic workshop)

### Delivery
- **Hands-on labs:** Run notebooks interactively
- **Live demos:** See agents in action
- **Best practices:** Production deployment patterns

## 🔍 Use Case: Telco Customer Support

The workshop demonstrates building a customer support agent for a telecommunications company:

- **Customer Queries:** Billing, technical issues, cancellations
- **Data Sources:** Customer DB, billing records, product manuals (PDFs)
- **External APIs:** Weather, distance calculation, web search
- **Outcome:** Faster resolution, higher satisfaction, lower costs

## 💡 Key Takeaways

1. **MCP enables seamless external API integration** while maintaining security via Unity Catalog
2. **Systematic prompt management** can reduce costs by 30-40% without sacrificing quality
3. **Multi-agent specialization** significantly improves domain expertise and response quality
4. **MLflow 3.0 provides end-to-end observability** from development through production

## 🚦 Next Steps

After completing this workshop:

1. **Adapt for your use case:** Replace telco scenario with your domain
2. **Add more specialists:** Create agents for your specific needs
3. **Integrate real APIs:** Move from mock to production APIs
4. **Deploy to production:** Use Lakehouse Apps or Model Serving
5. **Monitor & improve:** Use MLflow monitoring to continuously optimize

## 📝 License

This project is licensed under the Apache License 2.0 - see LICENSE file for details.

## 🙏 Acknowledgments

Built on Databricks platform capabilities:
- Unity Catalog for governance
- Vector Search for RAG
- MLflow 3.0 for agent lifecycle
- Lakehouse Apps for deployment

## 📧 Contact

For questions or feedback about this workshop:
- GitHub Issues: [Create an issue](https://github.com/Aradhya0510/ai-agent-advanced-demo/issues)
- Databricks Community: [Join discussions](https://community.databricks.com/)

---

**Happy Building!** 🚀

Built with ❤️ on Databricks

