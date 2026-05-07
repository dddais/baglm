"""
Run LMM inference (VSG or Progress) on custom video data.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import t2v_metrics
from constants import PROMPT_PROGRESS, PROMPT_VSG
from custom_dataset import CustomDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, type=str, help="Directory containing video files (flat)")
    parser.add_argument("--video_annots_file", required=True, type=str)
    parser.add_argument("--result_dir", required=True, type=str)
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--decode_device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Video decoding device. Use cpu to avoid GPU decoder OOM.",
    )
    parser.add_argument("--model", default="internvl2.5-8b", type=str)
    parser.add_argument("--visual_batch_size", default=1, type=int)
    parser.add_argument("--text_batch_size", default=32, type=int)
    parser.add_argument("--question_file", default=None, type=str)
    parser.add_argument("--segment_duration", type=int, default=2)
    parser.add_argument("--sampling_fps", type=int, default=2)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000000)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prompt_type", choices=[PROMPT_VSG, PROMPT_PROGRESS], required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    result_dir = os.path.join(args.result_dir, args.model)
    os.makedirs(result_dir, exist_ok=True)

    print(f"Model: {args.model}")
    score_func = t2v_metrics.get_score_model(
        model=args.model, device=args.device, cache_dir=args.cache_dir
    )
    preprocess_fn = score_func.model.get_preprocessor()

    dataset = CustomDataset(
        video_dir=args.video_dir,
        video_annots_file=args.video_annots_file,
        result_dir=result_dir,
        preprocess_fn=preprocess_fn,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        segment_duration=args.segment_duration,
        sampling_fps=args.sampling_fps,
        decode_device=args.decode_device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    kwargs = {}
    if args.question_file is not None:
        with open(args.question_file) as f:
            kwargs["question_template"] = f.read()
        print(f"Using question template from: {args.question_file}")

    for batch_data in tqdm(dataloader, total=len(dataloader)):
        for video_data in batch_data:
            video_uid = video_data["video_uid"]
            result_path = os.path.join(result_dir, f"{video_uid}.pt")

            if os.path.exists(result_path):
                print(f"Already exists, skipping: {result_path}")
                continue

            if args.prompt_type == PROMPT_VSG:
                scores = score_func.forward_vsg(
                    video_data, args.visual_batch_size, **kwargs
                ).cpu()
            elif args.prompt_type == PROMPT_PROGRESS:
                scores = score_func.forward_progress(
                    video_data, args.visual_batch_size, args.text_batch_size, **kwargs
                ).cpu()
            else:
                raise ValueError(f"Unknown prompt_type: {args.prompt_type}")

            scores = scores.repeat_interleave(args.segment_duration, dim=0)
            torch.save(scores, result_path)
            print(f"Saved: {result_path}  shape={list(scores.shape)}")


if __name__ == "__main__":
    main()
