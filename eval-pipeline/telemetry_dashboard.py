from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui.dashboard_actions import create_dashboard_job, list_dashboard_jobs
from src.ui.dashboard_data import (build_observability_retention_index,
                                   build_observability_views,
                                   load_telemetry_data)
from src.ui.reviewer_actions import (get_reviewer_summary, list_pending_reviews,
                                     submit_reviewer_feedback)

st.set_page_config(page_title="Telemetry Dashboard", layout="wide")
st.title("Domain-Specific LLM Evaluation Telemetry")


BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_dashboard_telemetry():
    return load_telemetry_data(BASE_DIR)


data = load_dashboard_telemetry()
if not data:
    st.warning("No telemetry data found in outputs/ directory.")
else:
    st.subheader(f"Found {len(data)} Pipeline Runs")

    # Overview Table
    summary = []
    for run in data:
        document_processing = run.get("document_processing", {})
        summary_metrics = run.get("summary", {})
        hardware = run.get("evaluation_metadata", {}).get(
            "hardware_acceleration_telemetry", {}
        )
        gpu_saturation = hardware.get("gpu_saturation", {})
        summary.append(
            {
                "Run ID": run.get("run_id", "Unknown"),
                "Docs Processed": document_processing.get("total_files", 0),
                "Chunks": document_processing.get("chunks", 0),
                "Samples Generated": summary_metrics.get("total_generated", 0),
                "Failed Syntheses": summary_metrics.get("total_failed", 0),
                "GPU Saturation": gpu_saturation.get("saturation_level", "n/a"),
                "Status": run.get("status", "unknown"),
            }
        )
    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True)

    observability = build_observability_views(data)
    latency_df = pd.DataFrame(observability["latency_trends"])
    fallback_df = pd.DataFrame(observability["fallback_trends"])
    saturation_df = pd.DataFrame(observability["saturation_trends"])
    error_modes_df = pd.DataFrame(observability["error_mode_trends"])

    st.subheader("Hardware Observability")
    if not latency_df.empty:
        st.caption("Latency percentile trend across runs")
        st.line_chart(
            latency_df.set_index("run_id")[["p50_latency_seconds", "p95_latency_seconds"]]
        )
    if not fallback_df.empty:
        st.caption("Fallback ratio over runs")
        st.dataframe(fallback_df, use_container_width=True)
    if not saturation_df.empty:
        st.caption("GPU saturation trend")
        st.line_chart(
            saturation_df.set_index("run_id")[["gpu_utilization", "kv_cache_utilization"]]
        )
        st.dataframe(saturation_df, use_container_width=True)
    if not error_modes_df.empty:
        st.caption("Per-error-mode trend")
        st.dataframe(error_modes_df, use_container_width=True)

    retention_index = build_observability_retention_index(BASE_DIR)
    st.subheader("Retention Index")
    st.caption(f"Retention artifact: {retention_index['retention_index_path']}")
    window_df = pd.DataFrame(
        [
            {"window": window, **values}
            for window, values in retention_index["window_comparisons"].items()
        ]
    )
    if not window_df.empty:
        st.dataframe(window_df, use_container_width=True)
    artifact_df = pd.DataFrame(retention_index["artifacts"])
    if not artifact_df.empty:
        st.caption("Artifact links and drill-down anchors")
        st.dataframe(artifact_df, use_container_width=True)
    if retention_index.get("error_drilldown"):
        st.caption("Cross-run error drill-down")
        st.dataframe(
            pd.DataFrame(
                [
                    {"error_mode": key, "count": value}
                    for key, value in retention_index["error_drilldown"].items()
                ]
            ),
            use_container_width=True,
        )
    diff_df = pd.DataFrame(retention_index.get("per_run_diffs", []))
    if not diff_df.empty:
        st.caption("Per-run diff against previous retained run")
        st.dataframe(diff_df, use_container_width=True)
    anomaly_df = pd.DataFrame(retention_index.get("anomaly_flags", []))
    if not anomaly_df.empty:
        st.caption("Retained anomaly flags")
        st.dataframe(anomaly_df, use_container_width=True)
    if retention_index.get("error_mode_artifacts"):
        st.caption("Error-mode to artifact cross-links")
        cross_links = []
        for mode, entries in retention_index["error_mode_artifacts"].items():
            for entry in entries:
                cross_links.append({"error_mode": mode, **entry})
        st.dataframe(pd.DataFrame(cross_links), use_container_width=True)
    searchable_df = pd.DataFrame(retention_index.get("searchable_artifacts", []))
    if not searchable_df.empty:
        st.caption("Searchable artifact index")
        st.dataframe(searchable_df, use_container_width=True)
    clusters = retention_index.get("issue_clusters", {})
    if clusters:
        st.caption("Retained issue clusters")
        st.dataframe(
            pd.DataFrame(
                [{"cluster": key, **value} for key, value in clusters.items()]
            ),
            use_container_width=True,
        )

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

st.subheader("Reviewer Queue")
review_summary = get_reviewer_summary(base_dir=BASE_DIR)
pending_reviews = list_pending_reviews(base_dir=BASE_DIR, status="pending")

st.caption(
    f"Pending: {review_summary['pending_reviews']} | "
    f"Resolved: {review_summary['resolved_reviews']} | "
    f"Ingested Results: {review_summary['ingested_reviewer_results']}"
)

if pending_reviews:
    review_df = pd.DataFrame(
        [
            {
                "Review ID": item["review_id"],
                "Priority": item["priority"],
                "Question": item["question"],
                "Reason": item["reason"],
                "Confidence": item.get("confidence"),
            }
            for item in pending_reviews
        ]
    )
    st.dataframe(review_df, use_container_width=True)

    selected_review = st.selectbox(
        "Select pending review",
        options=[item["review_id"] for item in pending_reviews],
        format_func=lambda review_id: next(
            item["question"] for item in pending_reviews if item["review_id"] == review_id
        ),
    )
    active_review = next(
        item for item in pending_reviews if item["review_id"] == selected_review
    )
    st.text_area("Candidate answer", active_review.get("answer", ""), height=140)

    with st.form("reviewer_submission_form"):
        approved = st.radio("Decision", options=[True, False], format_func=lambda value: "Approve" if value else "Reject")
        score = st.slider("Reviewer score", min_value=0.0, max_value=1.0, value=1.0 if active_review.get("priority") == "low" else 0.5, step=0.05)
        reviewer = st.text_input("Reviewer", value="streamlit-reviewer")
        notes = st.text_area("Notes", value=active_review.get("reason", ""))
        submitted = st.form_submit_button("Submit review decision")
        if submitted:
            result = submit_reviewer_feedback(
                {
                    "review_id": active_review["review_id"],
                    "index": active_review.get("index"),
                    "question": active_review.get("question", ""),
                    "approved": approved,
                    "score": score,
                    "notes": notes,
                    "reviewer": reviewer,
                },
                base_dir=BASE_DIR,
            )
            if result.get("submitted"):
                st.success("Review submitted.")
            else:
                st.warning("Review submission did not update any queue item.")
else:
    st.info("No pending reviewer decisions in the queue.")
