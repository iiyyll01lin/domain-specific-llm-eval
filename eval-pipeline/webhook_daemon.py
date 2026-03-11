import os
import subprocess

from fastapi import BackgroundTasks, FastAPI

app = FastAPI(title="RAGAS Evaluation Webhook Daemon")


def run_evaluation_pipeline():
    """Trigger the pipeline asynchronously on incoming git pushes / hooks"""
    try:
        print("Starting CI/CD background task evaluation run...")
        subprocess.run(
            ["python3", "run_pure_ragas_pipeline.py", "--docs", "5", "--samples", "50"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        print("CI/CD evaluation run complete.")
    except Exception as e:
        print(f"CI/CD Webhook pipeline failure: {e}")


@app.post("/webhook")
async def trigger_webhook(background_tasks: BackgroundTasks):
    """
    Medium/Low Priority: Run pipeline on git pushes or other webhooks.
    """
    background_tasks.add_task(run_evaluation_pipeline)
    return {"message": "Evaluation pipeline triggered.", "status": "queued"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
