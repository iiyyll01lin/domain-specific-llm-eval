import glob
import json
import os

import pandas as pd
import streamlit as st

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

# Task 5 Orchestration
st.subheader("Generate new Testset")
doc_count = st.number_input("Documents to process:", min_value=1, max_value=50, value=5)
sample_count = st.number_input(
    "Samples to generate:", min_value=1, max_value=50, value=2
)
if st.button("Trigger Pipeline Action"):
    st.info(
        f"Will execute `python3 run_pure_ragas_pipeline.py --docs {doc_count} --samples {sample_count}`."
    )
    import subprocess

    result = subprocess.run(
        [
            "python3",
            "run_pure_ragas_pipeline.py",
            "--docs",
            str(int(doc_count)),
            "--samples",
            str(int(sample_count)),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        st.success("Successfully completed!")
        st.rerun()
    else:
        st.error(f"Failed with code {result.returncode}:\n{result.stderr}")
