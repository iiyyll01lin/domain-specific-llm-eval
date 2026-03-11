import streamlit as st
import json
import glob
import os
import pandas as pd

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
                except:
                    pass
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
