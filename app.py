"""Gradio web app: upload a lift video -> annotated video + report. Local-first (Phase 2).

Run:  .\.venv\Scripts\python.exe app.py   then open the printed http://127.0.0.1:7860 URL.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from src import pipeline

OUT_DIR = "output"


def analyze(video_path: str, lift: str, progress=gr.Progress()):
    """Run the pipeline on an uploaded video and return (annotated video, report markdown)."""
    if not video_path:
        raise gr.Error("Upload a lift video first.")
    progress(0.1, desc="Loading video...")
    try:
        result = pipeline.analyze(
            video_path, lift=lift, out_dir=OUT_DIR,
            progress=lambda f: progress(0.5, desc="Analysing frames..."),
        )
    except ValueError as exc:
        raise gr.Error(f"Couldn't read that video: {exc}")
    except NotImplementedError as exc:
        raise gr.Error(str(exc))
    report_md = Path(result["paths"]["report"]).read_text(encoding="utf-8")
    return result["paths"]["annotated_video"], report_md


with gr.Blocks(title="Form Lab") as demo:
    gr.Markdown("# Form Lab — lift analysis")
    with gr.Row():
        video_in = gr.Video(label="Your lift (side-on)")
        lift_in = gr.Radio(["squat", "deadlift"], value="squat", label="Lift")
    run_btn = gr.Button("Analyse", variant="primary")
    video_out = gr.Video(label="Annotated")
    report_out = gr.Markdown()
    run_btn.click(analyze, inputs=[video_in, lift_in], outputs=[video_out, report_out])

if __name__ == "__main__":
    demo.launch()
