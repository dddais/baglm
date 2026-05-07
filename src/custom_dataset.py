"""
Minimal Dataset for custom video data.
Reads video_annots.json directly; videos are flat in video_dir.
No taxonomy / annotation files needed.
"""

import json
import os

import torch
from torch.utils.data import Dataset

from utils.video_utils import get_frame_indices, load_video


class CustomDataset(Dataset):
    def __init__(
        self,
        video_dir,
        video_annots_file,
        result_dir,
        preprocess_fn=None,
        start_idx=0,
        end_idx=1000000,
        segment_duration=1,
        sampling_fps=2,
        decode_device="cpu",
    ):
        self.video_dir = video_dir
        self.preprocess_fn = preprocess_fn
        self.segment_duration = segment_duration
        self.sampling_fps = sampling_fps
        self.decode_device = decode_device

        with open(video_annots_file) as f:
            self.video_annots = json.load(f)

        self.video_annots = self.video_annots[start_idx:end_idx]
        self.video_annots = [
            v
            for v in self.video_annots
            if not os.path.exists(os.path.join(result_dir, f"{v['video_uid']}.pt"))
        ]

    def __len__(self):
        return len(self.video_annots)

    def __getitem__(self, idx):
        item = self.video_annots[idx]
        video_uid = item["video_uid"]
        video_path = os.path.join(self.video_dir, f"{video_uid}{item['video_ext']}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        num_frames, fps = item["video_num_frames"], item["video_fps"]
        frame_indices = get_frame_indices(num_frames, fps, self.sampling_fps)

        all_frames = load_video(video_path, frame_indices, device=self.decode_device)
        frames_per_segment = int(self.segment_duration * self.sampling_fps)

        if self.preprocess_fn is not None:
            all_frames = [
                self.preprocess_fn(frame.data, nframes=frames_per_segment)
                for frame in all_frames
            ]

        video_segments = []
        for i in range(0, len(all_frames), frames_per_segment):
            seg = all_frames[i : i + frames_per_segment]
            if len(seg) == frames_per_segment:
                video_segments.append(torch.cat(seg))

        texts = [
            {"goal": item["activity"], "step": step}
            for step in item["step_headline"]
        ]

        return {
            "video_uid": video_uid,
            "videos": video_segments,
            "task_id": item.get("activity", video_uid),
            "texts": texts,
        }
