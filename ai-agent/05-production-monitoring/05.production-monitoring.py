# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC # Production Monitoring: Automated Quality at Scale
# MAGIC
# MAGIC MLflow's production monitoring automatically runs quality assessments on a sample of your production traffic, ensuring your GenAI app maintains high quality standards without manual intervention. MLflow lets you use the same metrics you defined for offline evaluation in production, enabling you to have consistent quality evaluation across your entire application lifecycle - dev to prod.
# MAGIC
# MAGIC **Key benefits:** 
# MAGIC
# MAGIC - Automated evaluation - Run LLM judges on production traces with configurable sampling rates
# MAGIC - Continuous quality assessment - Monitor quality metrics in real-time without disrupting user experience
# MAGIC - Cost-effective monitoring - Smart sampling strategies to balance coverage with computational cost
# MAGIC
# MAGIC Production monitoring enables you to deploy confidently, knowing that you will proactively detect issues so you can address them before they cause a major impact to your users.
# MAGIC
# MAGIC For more details on generative AI monitoring refer to the [Monitor served models using AI Gateway-enabled inference tables](https://docs.databricks.com/gcp/en/ai-gateway/inference-tables) and [Production quality monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/run-scorer-in-prod) documentation.
# MAGIC
# MAGIC <img src="https://i.imgur.com/wv4p562.gif">
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F05-production-monitoring%2F05.production-monitoring&demo_name=ai-agent&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fai-agent%2F05-production-monitoring%2F05.production-monitoring&version=1&user_hash=849103abf30cff95ada66078de7ea959dddec3173bd69b9ad64853593258328d">

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow[databricks]>=3.1.1 databricks-agents databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Let's create our production grade monitor
# MAGIC
# MAGIC You can easily create your monitor using the UI, or directly the SDK:
# MAGIC

# COMMAND ----------

# MLflow 3: Use scorers for production monitoring
import mlflow
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    RetrievalGroundedness,
    Guidelines,
    ScorerSamplingConfig,
)

# Let's re-use an existing experiment
xp_name = os.getcwd().rsplit("/", 1)[0]+"/03-knowledge-base-rag/03.1-pdf-rag-tool"
mlflow.set_experiment(xp_name)

accuracy_guidelines = [
  """
  The response correctly references all factual information from the provided_info based on these rules:
    - All factual information must be directly sourced from the provided data with NO fabrication
    - Names, dates, numbers, and company details must be 100% accurate with no errors
    - Meeting discussions must be summarized with the exact same sentiment and priority as presented in the data
    - Support ticket information must include correct ticket IDs, status, and resolution details when available
    - All product usage statistics must be presented with the same metrics provided in the data
    - No references to CloudFlow features, services, or offerings unless specifically mentioned in the customer data
    - AUTOMATIC FAIL if any information is mentioned that is not explicitly provided in the data
  """,
]

steps_and_reasoning_guidelines = [
  """
  Reponse must be done without showing reasoning.
    - don't mention that you need to look up things
    - do not mention tools or function used
    - do not tell your intermediate steps or reasoning
  """,
]

# COMMAND ----------

# MLflow 3: Register and start scorers for production monitoring
# Built-in scorers (replacing BuiltinJudge)
safety_scorer = Safety().register(name="safety")
groundedness_scorer = RetrievalGroundedness().register(name="groundedness")
relevance_scorer = RelevanceToQuery().register(name="relevance_to_query")

# Guidelines scorers (replacing GuidelinesJudge)
accuracy_scorer = Guidelines(
    name="accuracy",
    guidelines=accuracy_guidelines
).register(name="accuracy_guidelines")

reasoning_scorer = Guidelines(
    name="steps_and_reasoning",
    guidelines=steps_and_reasoning_guidelines
).register(name="steps_and_reasoning_guidelines")

# COMMAND ----------

def start_production_monitoring():
  """
  Start production monitoring by activating scorers with sampling configs.
  Scorers will automatically evaluate traces in the experiment.
  """
  # Start each scorer with its sampling configuration
  # Note: groundedness has a lower sample rate (0.4) to reduce cost
  safety_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
  groundedness_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.4))
  relevance_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
  accuracy_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
  reasoning_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))
  
  print("Production monitoring started successfully!")
  print(f"Monitoring experiment: {xp_name}")
  print("Active scorers:")
  print("  - Safety (100% sample rate)")
  print("  - Groundedness (40% sample rate)")
  print("  - Relevance to Query (100% sample rate)")
  print("  - Accuracy Guidelines (100% sample rate)")
  print("  - Steps and Reasoning Guidelines (100% sample rate)")

# COMMAND ----------

# Scorers will automatically evaluate traces in the experiment (small cost incurs). 
# Uncomment to start production monitoring!
# start_production_monitoring()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How MLflow 3 Production Monitoring Works
# MAGIC
# MAGIC Once you start the scorers, they will automatically evaluate production traces:
# MAGIC
# MAGIC 1. **Scorer Registration**: Each scorer is registered with a name in the experiment
# MAGIC 2. **Sampling Configuration**: When started, each scorer is configured with a sample rate (e.g., 40% for groundedness to reduce cost)
# MAGIC 3. **Automatic Evaluation**: The monitoring job runs every 15 minutes, evaluating traces based on the configured sample rates
# MAGIC   - If you have different sampling rates per scorer, the job attempts to score the same traces when possible
# MAGIC   - For example, if scorer A has 20% and scorer B has 40%, the same 20% of traces will be used for both
# MAGIC 4. **Results Storage**: 
# MAGIC   - Scorer feedback is attached to each evaluated trace in the MLflow Experiment
# MAGIC   - All traces (not just sampled ones) are written to the Delta Table named `trace_logs_<MLflow_experiment_id>`
# MAGIC 5. **Viewing Results**: Check the Trace tab in the MLflow Experiment UI, or query the Delta Table directly with SQL/Spark
# MAGIC
# MAGIC **Note**: The first monitoring run may take 15-30 minutes. Subsequent runs occur every 15 minutes. Large traffic volumes may require additional processing time.