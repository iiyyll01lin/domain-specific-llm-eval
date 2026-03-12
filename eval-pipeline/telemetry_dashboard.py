import glob
import json
import os

import pandas as pd
import streamlit as st

from src.ui.dashboard_actions import create_dashboard_job, list_dashboard_jobs

st.set_page_config(page_title="Telemetry Dashboard", layout="wide")
st.title("Domain-Specific LLM Evaluation Telemetry")


@st.cache_data
def load_telemetry_data():
    files = glob.glob("../outputs/*/telemetry/runtime_telemetry.json") + glob.glob(
        "outputs/*/telemetry/runtime_telemetry.json"
    )
    data = []
    for f in files:
        if os.path.exists(f):
            with open(f, "r") as fp:
                try:
                    content = json.load(fp)
                    content["run_path"] = os.path.dirname(os.path.dirname(f))
                    data.append(content)
                except json.JSONDecodeError:
                    continue
    return data


data = load_telemetry_data()
if not data:
    st.warning("No telemetry data found in outputs/ directory.")
else:
    st.subheader(f"Found {len(data)} Pipeline Runs")

    # Overview Table
    summary = []
    for run in data:
        metrics = run.get("metrics", {})
        summary.append(
            {
                "Run ID": run.get("run_id", "Unknown"),
                "Execution Time (s)": metrics.get("execution_time_seconds", 0),
                "Docs Processed": metrics.get("documents_processed", 0),
                "Samples Generated": metrics.get("samples_generated", 0),
                "Failed Syntheses": metrics.get("failed_syntheses", 0),
                "Status": (
                    "✅ Success"
                    if metrics.get("generation_success_rate", 0) > 0
                    else "❌ Failed"
                ),
            }
        )
    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True)

# Task 5 Orchestration
st.subheader("Generate new Testset")
doc_count = st.number_input("Documents to process:", min_value=1, max_value=50, value=5)
sample_count = st.number_input(
    "Samples to generate:", min_value=1, max_value=50, value=2
)
config_override = st.text_input("Optional config path:", value="")
if st.button("Trigger Pipeline Action"):
    job = create_dashboard_job(
        docs=int(doc_count),
        samples=int(sample_count),
        config_path=config_override or None,
    )
    st.success(f"Queued background job {job['job_id']} (pid={job['pid']}).")

jobs_payload = list_dashboard_jobs()
if jobs_payload["jobs"]:
    st.subheader("Dashboard Jobs")
    jobs_df = pd.DataFrame(
        [
            {
                "Job ID": job["job_id"],
                "Status": job["status"],
                "Docs": job["docs"],
                "Samples": job["samples"],
                "Progress %": job.get("progress", {}).get("percentage", 0.0),
                "Latest Stage": job.get("progress", {}).get("latest_stage"),
                "Telemetry Status": job.get("telemetry_status"),
                "Updated At": job.get("updated_at"),
            }
            for job in jobs_payload["jobs"]
        ]
    )
    st.dataframe(jobs_df, use_container_width=True)

    selected_job_id = st.selectbox(
        "Inspect job details:",
        options=[job["job_id"] for job in jobs_payload["jobs"]],
    )
    selected_job = next(job for job in jobs_payload["jobs"] if job["job_id"] == selected_job_id)
    progress = selected_job.get("progress", {})
    st.progress(min(max(progress.get("percentage", 0.0) / 100.0, 0.0), 1.0))
    st.caption(
        f"Stage: {progress.get('latest_stage') or 'pending'} | "
        f"Completed stages: {progress.get('completed_stages', 0)}/{progress.get('total_stages', 0)}"
    )
    if selected_job.get("stdout_tail"):
        st.text_area("Recent stdout", selected_job["stdout_tail"], height=180)
    if selected_job.get("stderr_tail"):
        st.text_area("Recent stderr", selected_job["stderr_tail"], height=180)
