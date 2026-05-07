"""
Export readable BaGLM predictions for custom data.

Outputs:
  - JSON with per-segment VSG/BaGLM predictions and probabilities
  - CSV with one row per time segment
  - HTML timeline for visual inspection
"""

import argparse
import csv
import html
import json
import os
from typing import Any

import torch

from bayes_filter import run_bayes_filter


def key_to_graph_path(activity: str, variation: str, llm_prereq_dir: str) -> str:
    return os.path.join(llm_prereq_dir, f"{activity}_{variation}.pt")


def top_prediction(scores: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    idx = int(torch.argmax(scores).item())
    return {
        "index": idx,
        "label": labels[idx],
        "score": float(scores[idx].item()),
    }


def progress_expectation(progress_scores: torch.Tensor) -> list[float]:
    bins = torch.arange(10, device=progress_scores.device, dtype=progress_scores.dtype)
    values = (progress_scores * bins).sum(dim=1) / 9.0
    return [float(v.item()) for v in values]


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "video_uid",
        "start_sec",
        "end_sec",
        "vsg_label",
        "vsg_score",
        "baglm_label",
        "baglm_score",
        "none_score",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def write_html(path: str, video_uid: str, rows: list[dict[str, Any]]) -> None:
    blocks = []
    for row in rows:
        label = html.escape(row["baglm_label"])
        vsg_label = html.escape(row["vsg_label"])
        confidence = max(0.05, min(1.0, row["baglm_score"]))
        color = f"rgba(52, 120, 246, {confidence:.2f})"
        blocks.append(
            f"""
            <div class="row">
              <div class="time">{row["start_sec"]:.1f}-{row["end_sec"]:.1f}s</div>
              <div class="bar" style="background:{color}">
                <b>{label}</b>
                <span>BaGLM={row["baglm_score"]:.3f} | raw VSG: {vsg_label} ({row["vsg_score"]:.3f})</span>
              </div>
            </div>
            """
        )

    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BaGLM Predictions - {html.escape(video_uid)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; }}
    h1 {{ font-size: 22px; }}
    .row {{ display: flex; align-items: stretch; margin: 6px 0; }}
    .time {{ width: 90px; font-family: monospace; color: #444; padding: 8px 0; }}
    .bar {{ flex: 1; border-radius: 8px; padding: 8px 12px; color: white; }}
    .bar span {{ display: block; font-size: 12px; opacity: 0.9; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>BaGLM Predictions: {html.escape(video_uid)}</h1>
  {''.join(blocks)}
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_annots_file", required=True)
    parser.add_argument("--model", default="internvl2.5-8b")
    parser.add_argument("--lmm_vsg_dir", required=True)
    parser.add_argument("--lmm_prog_dir", required=True)
    parser.add_argument("--llm_prereq_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--segment_duration", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.video_annots_file) as f:
        video_annots = json.load(f)

    all_rows = []
    for video_annot in video_annots:
        video_uid = video_annot["video_uid"]
        activity = video_annot["activity"]
        variation = video_annot.get("variation", "none")
        steps = video_annot["step_headline"]
        labels = steps + ["None of the above."]

        vsg_path = os.path.join(args.lmm_vsg_dir, args.model, f"{video_uid}.pt")
        prog_path = os.path.join(args.lmm_prog_dir, args.model, f"{video_uid}.pt")
        prereq_path = key_to_graph_path(activity, variation, args.llm_prereq_dir)

        lmm_vsg_scores = torch.load(vsg_path)
        lmm_prog_scores = torch.load(prog_path)
        dependency_matrix = torch.load(prereq_path)
        beliefs = run_bayes_filter(lmm_vsg_scores, lmm_prog_scores, dependency_matrix)

        video_rows = []
        json_rows = []
        for t in range(beliefs.shape[0]):
            start_sec = t * args.segment_duration
            end_sec = (t + 1) * args.segment_duration
            vsg_pred = top_prediction(lmm_vsg_scores[t], labels)
            baglm_pred = top_prediction(beliefs[t], labels)

            row = {
                "video_uid": video_uid,
                "time_index": t,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "vsg_label": vsg_pred["label"],
                "vsg_score": vsg_pred["score"],
                "baglm_label": baglm_pred["label"],
                "baglm_score": baglm_pred["score"],
                "none_score": float(beliefs[t, -1].item()),
                "progress": progress_expectation(lmm_prog_scores[t]),
            }
            video_rows.append(row)
            json_rows.append(row)

        all_rows.extend(video_rows)

        json_path = os.path.join(args.output_dir, f"{video_uid}_predictions.json")
        csv_path = os.path.join(args.output_dir, f"{video_uid}_timeline.csv")
        html_path = os.path.join(args.output_dir, f"{video_uid}_timeline.html")

        with open(json_path, "w") as f:
            json.dump(
                {
                    "video_uid": video_uid,
                    "activity": activity,
                    "variation": variation,
                    "steps": steps,
                    "rows": json_rows,
                },
                f,
                indent=2,
            )
        write_csv(csv_path, video_rows)
        write_html(html_path, video_uid, video_rows)

        print(f"Saved: {json_path}")
        print(f"Saved: {csv_path}")
        print(f"Saved: {html_path}")

    write_csv(os.path.join(args.output_dir, "all_videos_timeline.csv"), all_rows)


if __name__ == "__main__":
    main()
