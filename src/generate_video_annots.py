"""
Scan a video directory and generate a video_annots.json template.

Usage:
    python src/generate_video_annots.py \
        --video_dir /path/to/videos \
        --output /path/to/video_annots.json

The script:
  1. Scans video_dir for all video files (.mp4, .webm, .mkv, .avi)
  2. Extracts metadata (num_frames, fps, duration) via ffprobe
  3. Writes a video_annots.json with placeholder activity / step_headline
     for you to fill in manually

After running, open the JSON and fill in:
  - "activity":       task description, e.g. "Restock Cola"
  - "step_headline":  list of candidate step strings
  - "clips" (optional): ground-truth annotations for evaluation
"""

import argparse
import json
import os
import subprocess


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def get_video_metadata(video_path: str) -> dict | None:
    """Use ffprobe to extract num_frames, fps, and duration."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"  [WARN] ffprobe failed for {video_path}: {e}")
        return None

    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue

        nb_frames = stream.get("nb_frames")
        if nb_frames and nb_frames != "N/A":
            num_frames = int(nb_frames)
        else:
            num_frames = None

        r_frame_rate = stream.get("r_frame_rate", "0/1")
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0

        duration = stream.get("duration")
        if duration and duration != "N/A":
            duration = float(duration)
        else:
            fmt_duration = info.get("format", {}).get("duration")
            duration = float(fmt_duration) if fmt_duration else None

        if num_frames is None and duration and fps > 0:
            num_frames = int(duration * fps)

        return {
            "num_frames": num_frames or 0,
            "fps": round(fps, 2),
            "duration": round(duration, 2) if duration else 0.0,
        }

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate video_annots.json template from video files")
    parser.add_argument("--video_dir", required=True, help="Directory containing video files")
    parser.add_argument("--output", required=True, help="Output path for video_annots.json")
    parser.add_argument("--variation", default="none", help="Default variation name (default: none)")
    args = parser.parse_args()

    video_dir = args.video_dir
    if not os.path.isdir(video_dir):
        print(f"Error: {video_dir} is not a directory")
        return

    video_files = sorted(
        f for f in os.listdir(video_dir)
        if os.path.isfile(os.path.join(video_dir, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video(s) in {video_dir}")

    annots = []
    for fname in video_files:
        video_path = os.path.join(video_dir, fname)
        uid, ext = os.path.splitext(fname)

        print(f"  Processing: {fname} ...", end=" ")
        meta = get_video_metadata(video_path)
        if meta is None:
            print("SKIPPED (ffprobe failed)")
            continue
        print(f"frames={meta['num_frames']}, fps={meta['fps']}, duration={meta['duration']}s")

        annots.append({
            "video_uid": uid,
            "video_ext": ext,
            "video_num_frames": meta["num_frames"],
            "video_fps": meta["fps"],
            "video_duration": meta["duration"],
            "activity": "TODO: fill in task name",
            "variation": args.variation,
            "step_headline": [
                "TODO: step 1",
                "TODO: step 2",
            ],
            "clips": [
                {
                    "annotations": [
                        {
                            "language_queries": []
                        }
                    ]
                }
            ],
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(annots, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(annots)} entries to {args.output}")
    print("Next: open the JSON and fill in 'activity' and 'step_headline' for each video.")
    print("      Optionally add ground-truth time ranges in 'clips.annotations.language_queries'.")


if __name__ == "__main__":
    main()
