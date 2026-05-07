"""
Recall computation for custom data.
Same logic as htstep_recall.py but uses 'activity' as grouping key.
Used by bayes_filter.py via dynamic import.
"""

import argparse
import json
import math
import os

import numpy as np
import torch


def get_Y_pred(scores):
    binary_matrix = np.zeros_like(scores, dtype=int)
    binary_matrix[np.argmax(scores, axis=0), np.arange(scores.shape[1])] = 1
    return binary_matrix


def get_Y_true(T, K, video_annot):
    Y_true = np.zeros([T, K], dtype=np.uint8)
    steps = video_annot["step_headline"]
    step_to_k = {step: k for k, step in enumerate(steps)}

    for clip in video_annot["clips"]:
        for annotation in clip["annotations"]:
            for lang_query in annotation["language_queries"]:
                step = lang_query["query"]
                if step not in step_to_k:
                    continue
                k = step_to_k[step]
                start_t = lang_query["clip_start_sec"]
                end_t = lang_query["clip_end_sec"]
                Y_true[math.floor(start_t) : math.ceil(end_t) + 1, k] = 1
    return Y_true


def get_recall(Y_true, Y_pred):
    recall = []
    for task in Y_true:
        ys_true = Y_true[task]
        ys_pred = Y_pred[task]
        for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
            y_true = ys_true[vid]
            y_pred = ys_pred[vid]
            K = y_true.shape[1]
            for k in range(K):
                if y_true[:, k].sum() > 0:
                    pred_t = np.argmax(y_pred[:, k])
                    if y_true[pred_t, k] == 1:
                        recall.append(1)
                    else:
                        recall.append(0)
    return np.mean(recall) if recall else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="internvl2.5-8b", type=str)
    parser.add_argument("--lmm_vsg_dir", required=True, type=str)
    parser.add_argument("--video_annots_file", required=True, type=str)
    args = parser.parse_args()

    lmm_vsg_dir = os.path.join(args.lmm_vsg_dir, args.model)

    with open(args.video_annots_file) as f:
        video_annots = json.load(f)

    video_annots = [
        v for v in video_annots
        if os.path.exists(os.path.join(lmm_vsg_dir, f"{v['video_uid']}.pt"))
    ]
    print(f"Evaluating {len(video_annots)} videos")

    Y_true, Y_pred = {}, {}
    for video_annot in video_annots:
        video_uid = video_annot["video_uid"]
        task = video_annot["activity"]
        scores = torch.load(os.path.join(lmm_vsg_dir, f"{video_uid}.pt"))

        y_pred = get_Y_pred(scores.numpy())
        Y_pred.setdefault(task, {})[video_uid] = y_pred
        Y_true.setdefault(task, {})[video_uid] = get_Y_true(*y_pred.shape, video_annot)

    recall = get_recall(Y_true, Y_pred)
    print(f"Recall@1: {recall:.4f}")


if __name__ == "__main__":
    main()
