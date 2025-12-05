# Build & Track ML Pipelines with DVC

raw → processed → features → model → evaluation → MLflow/DagsHub tracking



## DVC Commands

git init

dvc init

dvc repro

dvc dag

dvc metrics show

-- to run dvc experiements without polluting the git



dvc repro: 
- Purpose: Reproduce the pipeline from scratch (or from the last changed stage).
- When to use:
- If you’ve changed code or data and want to rebuild the pipeline outputs.
- It runs through all dependent stages in order.

dvc exp run:
- Purpose: Reproduce the pipeline from scratch (or from the last changed stage).
- When to use:
- If you’ve changed code or data and want to rebuild the pipeline outputs.
- It runs through all dependent stages in order.

dvc exp run
dvc exp show

Override experiements:

dvc exp run -S logreg.max_iter=200 -S gb.n_estimators=300

dvc exp apply = pipeline promotion (choosing the winner and making it official in code + Git).


Local vs Production:
Local (what you’re doing now):
- Run dvc exp run and dvc exp apply manually.
- Push artifacts with dvc push.
- Log experiments in MLflow locally.
- Great for prototyping, debugging, and reproducibility.

Production (enterprise setup):
- Pipelines are automated and event-driven.
- Data landing in S3 (or another data lake) triggers the pipeline.
- Orchestration tools (Airflow, Prefect, Dagster, Kubeflow, Databricks Workflows) manage scheduling, retries, and dependencies.
- Monitoring dashboards (MLflow, Prometheus/Grafana, DagsHub, Datadog) track metrics, drift, and failures.

Typical Enterprise Flow
1. 	Data Ingestion->
• 	New data arrives in S3 (e.g., raw logs, CSVs, parquet).
• 	An event notification (SNS/SQS, Lambda, or Databricks Auto Loader) triggers the pipeline.
2. 	Pipeline Orchestration->
• 	Orchestrator (Airflow DAG, Prefect flow, Databricks job) kicks off:
• 	Preprocessing
• 	Feature engineering
• 	Model training
• 	Evaluation
3. 	Artifact Management->
• 	DVC or MLflow ensures datasets, models, and metrics are versioned.
• 	Artifacts are stored in S3/DagsHub for reproducibility.
4. 	Monitoring & Alerts->
• 	MLflow tracks experiments and model registry.
• 	Monitoring tools detect anomalies (e.g., data drift, accuracy drop).
• 	Alerts are sent to Slack/Teams/Email.
5. 	Deployment->
• 	Approved models are promoted from MLflow registry → production endpoints (via Docker/Kubernetes/Databricks MLflow serving)


dvc push -r dagshub -> to use dagshub default s3 compatible storage for all dvc files(data, model, metrics..)

dvc push -r s3remote -> to use  aws s3 storage for all dvc files(data, model, metrics..) 

dvc remote default s3remote

Typical workflow
- Make changes to code/configs → git add, git commit, git push.
- Generate new data/models locally → dvc add, dvc commit.
- Sync artifacts → dvc push -r s3remote.
- Collaborators can reproduce → git pull + dvc pull -r s3remote.

- Git push → code/configs to GitHub/DagsHub. 
- DVC push → data/models/metrics to S3



✅ Best Practice Workflow
- Code release → push Git once, orchestrate with Airflow.
- Daily runs → Airflow executes dvc repro with new data.
- Artifact sync → Airflow runs dvc push so S3 has the new blobs.
- Git commit optional → if you want to version daily runs, you can commit updated dvc.lock. But many teams skip this in production and just rely on S3 + experiment tracking.


⚡ Two Modes of Operation
- Research/Dev → commit dvc.lock daily to Git, so you can diff experiments.
- Production → keep Git pinned, but still push artifacts daily to S3 for reproducibility and audit.
