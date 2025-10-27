# MLflow Prompt Registry - Quick Start Guide

## 🚀 Getting Started

### 1. Register a Prompt

```python
import mlflow

prompt = mlflow.genai.register_prompt(
    name="telco_support_technical",
    template="You are a technical support engineer...",
    commit_message="Initial version - basic troubleshooting",
    tags={
        "use_case": "technical_support",
        "author": "team-ai",
        "environment": "development",
        "token_count": "250"
    }
)

print(f"✅ Registered {prompt.name} version {prompt.version}")
```

### 2. Load a Prompt

```python
# Load by version number
prompt_v1 = mlflow.genai.load_prompt("prompts:/telco_support_technical", version=1)

# Load by alias
prod_prompt = mlflow.genai.load_prompt("prompts:/telco_support_technical@production")
latest_prompt = mlflow.genai.load_prompt("prompts:/telco_support_technical@latest")

# Use the prompt
agent_config["system_prompt"] = prod_prompt.template
```

### 3. Set Aliases

```python
from mlflow import MlflowClient

client = MlflowClient()

# Set production alias
client.set_registered_prompt_alias("telco_support_technical", "production", version=1)

# Set staging alias for testing
client.set_registered_prompt_alias("telco_support_technical", "staging", version=2)
```

### 4. Search Prompts

```python
# Search by tags
prompts = mlflow.genai.search_prompts(filter_string="tags.use_case = 'technical_support'")

# List all prompts with 'telco' in name
prompts = mlflow.genai.search_prompts(filter_string="name LIKE 'telco%'")

for prompt in prompts:
    print(f"{prompt.name} v{prompt.version} - {prompt.tags.get('use_case')}")
```

---

## 📋 Common Workflows

### Workflow 1: Update Production Prompt (Zero Downtime)

```python
# Step 1: Register improved version
improved_prompt = mlflow.genai.register_prompt(
    name="telco_support_technical",
    template="You are a senior technical support engineer...",
    commit_message="v2: Added 5G troubleshooting and IoT support",
    tags={"improvement": "5g_iot", "use_case": "technical_support"}
)

# Step 2: Test in staging
client.set_registered_prompt_alias("telco_support_technical", "staging", improved_prompt.version)
staging_prompt = mlflow.genai.load_prompt("prompts:/telco_support_technical@staging")
# ... run evaluations ...

# Step 3: Promote to production (if tests pass)
client.set_registered_prompt_alias("telco_support_technical", "production", improved_prompt.version)

# ✅ All systems loading @production now use the new version automatically!
```

### Workflow 2: A/B Testing

```python
# Set up two variants
client.set_registered_prompt_alias("telco_support_technical", "variant_a", version=1)
client.set_registered_prompt_alias("telco_support_technical", "variant_b", version=2)

# Load and test both
prompt_a = mlflow.genai.load_prompt("prompts:/telco_support_technical@variant_a")
prompt_b = mlflow.genai.load_prompt("prompts:/telco_support_technical@variant_b")

# ... run evaluation suite on both ...

# Promote winner to production
if quality_score_b > quality_score_a:
    client.set_registered_prompt_alias("telco_support_technical", "production", version=2)
```

### Workflow 3: Instant Rollback

```python
# Something went wrong? Rollback to previous version
client.set_registered_prompt_alias("telco_support_technical", "production", previous_version)

# ✅ Systems loading @production now use the old stable version
```

### Workflow 4: Compare Versions

```python
# Load two versions for comparison
v1 = mlflow.genai.load_prompt("prompts:/telco_support_technical", version=1)
v2 = mlflow.genai.load_prompt("prompts:/telco_support_technical", version=2)

print(f"Version 1 - Tokens: {v1.tags.get('token_count')}")
print(f"Version 2 - Tokens: {v2.tags.get('token_count')}")
print(f"Change: {v2.commit_message}")

# Side-by-side diff available in MLflow UI
```

---

## 🏷️ Alias Best Practices

| Alias | Purpose | Who Sets It | When to Update |
|-------|---------|-------------|----------------|
| `@production` | Currently deployed | Ops/Release team | After validation |
| `@staging` | Testing environment | Dev team | When testing new version |
| `@champion` | Best performing | Auto (after A/B test) | When quality improves |
| `@candidate` | Under evaluation | Dev team | When testing variant |
| `@latest` | Most recent | MLflow (automatic) | On every registration |

---

## 🎯 Use Cases by Industry

### Customer Support
```python
# Route by query type
if is_billing_query:
    prompt = mlflow.genai.load_prompt("prompts:/support_billing@production")
elif is_technical_query:
    prompt = mlflow.genai.load_prompt("prompts:/support_technical@production")
else:
    prompt = mlflow.genai.load_prompt("prompts:/support_general@production")
```

### Content Generation
```python
# Route by content type
if content_type == "blog":
    prompt = mlflow.genai.load_prompt("prompts:/content_blog@production")
elif content_type == "social":
    prompt = mlflow.genai.load_prompt("prompts:/content_social@production")
```

### Code Generation
```python
# Route by programming language
prompt_name = f"code_gen_{language}"
prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@production")
```

---

## 📊 Tracking Prompt Performance

### Log Metrics with Prompt Version

```python
with mlflow.start_run():
    # Load production prompt
    prompt = mlflow.genai.load_prompt("prompts:/telco_support_technical@production")
    
    # Log prompt metadata
    mlflow.log_param("prompt_name", prompt.name)
    mlflow.log_param("prompt_version", prompt.version)
    mlflow.log_param("prompt_alias", "production")
    
    # Log model with prompt
    mlflow.pyfunc.log_model(
        name="agent",
        python_model=agent,
        # ... rest of config ...
    )
    
    # Run evaluation
    eval_results = mlflow.evaluate(...)
    
    # Metrics automatically linked to prompt version
```

### Query Performance by Prompt Version

```sql
-- In MLflow UI or via API
SELECT 
  params.prompt_version,
  AVG(metrics.quality_score) as avg_quality,
  AVG(metrics.latency) as avg_latency,
  COUNT(*) as num_runs
FROM mlflow.runs
WHERE params.prompt_name = 'telco_support_technical'
GROUP BY params.prompt_version
ORDER BY params.prompt_version DESC
```

---

## 🔍 Debugging & Troubleshooting

### Check Current Aliases

```python
# Get prompt with aliases
prompt = mlflow.genai.load_prompt("prompts:/telco_support_technical@latest")
print(f"Version: {prompt.version}")
print(f"Aliases: {prompt.aliases}")  # List of aliases pointing to this version

# Check what version an alias points to
prod_prompt = mlflow.genai.load_prompt("prompts:/telco_support_technical@production")
print(f"Production is version: {prod_prompt.version}")
```

### List All Versions

```python
# Search returns all versions
all_versions = mlflow.genai.search_prompts(filter_string="name = 'telco_support_technical'")

for prompt in sorted(all_versions, key=lambda p: p.version):
    print(f"v{prompt.version} - {prompt.commit_message}")
    print(f"  Aliases: {', '.join(prompt.aliases) if prompt.aliases else 'None'}")
    print(f"  Tags: {prompt.tags}")
```

### Verify Prompt Content

```python
# Load and inspect
prompt = mlflow.genai.load_prompt("prompts:/telco_support_technical@production")
print(f"Template length: {len(prompt.template)} chars")
print(f"First 200 chars:\n{prompt.template[:200]}...")
```

---

## 🔗 Integration Patterns

### Pattern 1: Config-Driven Prompt Selection

```python
# config.yaml
prompts:
  billing: "prompts:/support_billing@production"
  technical: "prompts:/support_technical@production"
  retention: "prompts:/support_retention@production"

# Load in code
prompt = mlflow.genai.load_prompt(config['prompts'][query_type])
```

### Pattern 2: Dynamic Routing

```python
def get_prompt_for_query(query: str) -> str:
    """Select optimal prompt based on query characteristics."""
    
    if "billing" in query.lower() or "payment" in query.lower():
        return mlflow.genai.load_prompt("prompts:/support_billing@production")
    elif "connection" in query.lower() or "error" in query.lower():
        return mlflow.genai.load_prompt("prompts:/support_technical@production")
    else:
        return mlflow.genai.load_prompt("prompts:/support_general@production")
```

### Pattern 3: Cost-Optimized Routing

```python
def get_cost_optimized_prompt(query_complexity: str) -> str:
    """Use cheaper prompts for simple queries."""
    
    if query_complexity == "simple":
        # Use concise, low-token prompt
        return mlflow.genai.load_prompt("prompts:/support_concise@production")
    elif query_complexity == "complex":
        # Use detailed, high-quality prompt
        return mlflow.genai.load_prompt("prompts:/support_detailed@production")
    else:
        return mlflow.genai.load_prompt("prompts:/support_standard@production")
```

---

## 📚 Additional Resources

- [MLflow Prompt Registry Documentation](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [Databricks MLflow Guide](https://docs.databricks.com/mlflow/index.html)

---

## ⚡ Quick Command Reference

```python
# Register
prompt = mlflow.genai.register_prompt(name, template, commit_message, tags)

# Load
prompt = mlflow.genai.load_prompt("prompts:/name@alias")
prompt = mlflow.genai.load_prompt("prompts:/name", version=1)

# Search
prompts = mlflow.genai.search_prompts(filter_string="tags.use_case = 'support'")

# Set Alias
client.set_registered_prompt_alias(name, alias, version)

# Delete Alias
client.delete_registered_prompt_alias(name, alias)

# Get Version by Alias
prompt = mlflow.genai.load_prompt("prompts:/name@alias")
version = prompt.version
```

---

## 🎓 Best Practices Summary

✅ **Always use aliases** (`@production`, `@staging`) - never hardcode version numbers  
✅ **Write descriptive commit messages** - explain what changed and why  
✅ **Tag prompts with metadata** - use case, cost, performance, author  
✅ **Test in staging first** - validate before moving @production alias  
✅ **Track prompt version in MLflow runs** - log as parameters for lineage  
✅ **Use cost-aware routing** - match prompt complexity to query complexity  
✅ **Version prompts like code** - treat prompt changes as deployments  
✅ **Monitor prompt performance** - track quality/cost/latency per version  

---

**Ready to get started?** Check out the full example in:
`ai-agent/03-advanced-capabilities/03.3-prompt-registry-management.py`

