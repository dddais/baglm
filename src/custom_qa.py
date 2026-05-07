"""
Generate prerequisite dependency matrices for custom data.
Reads step_headline directly from video_annots.json — no taxonomy file needed.
Produces one .pt per (activity, variation) group.
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import product

import torch
from tqdm import tqdm

import t2v_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_annots_file", required=True, type=str)
    parser.add_argument("--result_dir", required=True, type=str)
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model", default="llama-3.3-70b-instruct", type=str)
    parser.add_argument("--question_file", default=None, type=str)
    parser.add_argument("--system_file", default=None, type=str)
    parser.add_argument("--oai_key_path", default=None, type=str)
    parser.add_argument("--openai_base_url", default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = os.path.join(args.result_dir, args.model)
    os.makedirs(result_dir, exist_ok=True)

    print(f"Model: {args.model}")
    if "gpt" in args.model:
        if args.oai_key_path is None:
            raise ValueError("--oai_key_path is required for GPT models")
        with open(args.oai_key_path) as f:
            key = f.read().strip()
        score_func = t2v_metrics.get_score_model(
            model=args.model, api_key=key, base_url=args.openai_base_url
        )
    else:
        score_func = t2v_metrics.get_score_model(
            model=args.model, device=args.device, cache_dir=args.cache_dir
        )

    kwargs = {}
    if args.system_file is not None:
        with open(args.system_file) as f:
            kwargs["system_prompt"] = f.read()
    if args.question_file is not None:
        with open(args.question_file) as f:
            kwargs["question_template"] = f.read()
        print(f"Using question template from: {args.question_file}")

    with open(args.video_annots_file) as f:
        video_annots = json.load(f)

    groups = defaultdict(list)
    for v in video_annots:
        key = (v["activity"], v.get("variation", "none"))
        groups[key].append(v)

    print(f"Found {len(groups)} (activity, variation) groups")

    for (activity, variation), videos in tqdm(groups.items(), total=len(groups)):
        save_name = f"{activity}_{variation}.pt"
        save_path = os.path.join(result_dir, save_name)

        if os.path.exists(save_path):
            print(f"Already exists, skipping: {save_path}")
            continue

        steps = videos[0]["step_headline"]
        batch = [
            {"goal": activity, "step_a": a, "step_b": b}
            for a, b in product(steps, repeat=2)
        ]

        try:
            scores = score_func.batch_forward_text(batch, args.batch_size, **kwargs).cpu()
            torch.save(scores, save_path)
            print(f"Saved: {save_path}  shape={list(scores.shape)}")
        except Exception as e:
            print(f"Failed ({activity}, {variation}): {e}")


if __name__ == "__main__":
    main()
