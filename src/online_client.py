"""
BaGLM Online Test Client.

Connects to the BaGLM inference server and sends camera frames for real-time
step grounding. Supports two modes:

1. **Video file mode** (default): Reads frames from a local video file and
   sends them segment-by-segment to the server.

2. **Webcam mode** (`--webcam`): Captures frames from a local camera device
   and sends them in real-time.

Protocol (matches online_server.py):
    Send: 4-byte big-endian length + JSON header
          raw frame data (N * H * W * 3 bytes)
    Recv: 4-byte big-endian length + JSON response

Usage:
    # From a video file:
    python src/online_client.py --video_path /path/to/video.mp4

    # From webcam (device 0):
    python src/online_client.py --webcam --webcam_id 0

    # Custom server address:
    python src/online_client.py --host 192.168.1.100 --port 9999 \
        --video_path /path/to/video.mp4
"""

import argparse
import json
import struct
import sys
import time

import numpy as np


# ------------------------------------------------------------------
# Networking helpers (shared protocol with server)
# ------------------------------------------------------------------

def recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server disconnected")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock) -> bytes:
    raw_len = recv_exact(sock, 4)
    msg_len = struct.unpack("!I", raw_len)[0]
    return recv_exact(sock, msg_len)


def send_msg(sock, data: bytes):
    sock.sendall(struct.pack("!I", len(data)) + data)


def send_frames(sock, frames_np: np.ndarray):
    """Send a batch of frames to the server.

    Args:
        sock: Connected TCP socket.
        frames_np: [N, H, W, 3] uint8 numpy array.
    """
    n, h, w, c = frames_np.shape
    assert c == 3, f"Expected 3 channels, got {c}"

    header = json.dumps({
        "n_frames": n,
        "frame_h": h,
        "frame_w": w,
    }).encode("utf-8")

    send_msg(sock, header)
    sock.sendall(frames_np.tobytes())


def recv_response(sock) -> dict:
    """Receive and parse one response from the server."""
    resp_raw = recv_msg(sock)
    return json.loads(resp_raw)


# ------------------------------------------------------------------
# Frame source: video file
# ------------------------------------------------------------------

def iterate_video_segments(video_path: str, segment_duration: int, sampling_fps: int):
    """Yield segments of frames from a video file using decord.

    Yields:
        frames_np: [frames_per_segment, H, W, 3] uint8 numpy array.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    fps = float(vr.get_avg_fps())

    frames_per_segment = int(segment_duration * sampling_fps)

    # Sampled indices
    total_sampled = int(num_frames * (sampling_fps / fps))
    if total_sampled < 1:
        total_sampled = 1
    indices = np.linspace(0, num_frames - 1, total_sampled).astype(int).tolist()

    print(f"Video: {num_frames} frames, {fps:.1f} fps, "
          f"sampling {total_sampled} frames, "
          f"{frames_per_segment} frames/segment")

    # Read sampled frames (NHWC uint8)
    all_frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3]

    for start in range(0, len(all_frames), frames_per_segment):
        end = start + frames_per_segment
        if end > len(all_frames):
            break
        yield all_frames[start:end]


# ------------------------------------------------------------------
# Frame source: webcam
# ------------------------------------------------------------------

def iterate_webcam_segments(webcam_id: int, segment_duration: int, sampling_fps: int,
                            target_size: tuple = (640, 480)):
    """Yield segments of frames from a webcam.

    Yields:
        frames_np: [frames_per_segment, H, W, 3] uint8 numpy array.
    """
    import cv2

    cap = cv2.VideoCapture(webcam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])

    frames_per_segment = segment_duration * sampling_fps
    interval = 1.0 / sampling_fps  # Time between sampled frames

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {webcam_id}")

    print(f"Webcam opened (device {webcam_id}), "
          f"capturing {frames_per_segment} frames at {sampling_fps} fps")

    try:
        while True:
            frames = []
            next_capture = time.time()

            for _ in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    print("Webcam read failed.")
                    return
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                next_capture += interval
                sleep_time = next_capture - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            yield np.stack(frames)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BaGLM Online Test Client")
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=9999, type=int)
    parser.add_argument("--video_path", default=None, type=str,
                        help="Path to a video file to send.")
    parser.add_argument("--webcam", action="store_true",
                        help="Use webcam instead of video file.")
    parser.add_argument("--webcam_id", default=0, type=int)
    parser.add_argument("--segment_duration", default=2, type=int)
    parser.add_argument("--sampling_fps", default=2, type=int)
    parser.add_argument("--max_segments", default=0, type=int,
                        help="Stop after N segments (0 = unlimited).")
    parser.add_argument("--log_dir", default="logs/online", type=str,
                        help="Directory to save session summary logs.")
    args = parser.parse_args()

    if not args.webcam and args.video_path is None:
        parser.error("Either --video_path or --webcam must be specified.")

    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to {args.host}:{args.port} ...")
    sock.connect((args.host, args.port))
    print("Connected!")

    try:
        if args.video_path:
            frame_iter = iterate_video_segments(
                args.video_path, args.segment_duration, args.sampling_fps,
            )
        else:
            frame_iter = iterate_webcam_segments(
                args.webcam_id, args.segment_duration, args.sampling_fps,
            )

        seg_count = 0
        all_results = []
        for frames_np in frame_iter:
            seg_count += 1
            if args.max_segments > 0 and seg_count > args.max_segments:
                break

            t0 = time.time()
            send_frames(sock, frames_np)
            response = recv_response(sock)
            elapsed = (time.time() - t0) * 1000

            if "error" in response:
                print(f"[Segment {seg_count}] ERROR: {response['error']}")
                continue

            step = response["step"]
            conf = response["confidence"]
            latency = response.get("latency_ms", 0)
            belief = response.get("belief", {})

            # Find top-3 beliefs
            sorted_belief = sorted(belief.items(), key=lambda x: x[1], reverse=True)
            top3 = ", ".join(f"{k}={v:.2f}" for k, v in sorted_belief[:3])

            print(
                f"[Seg {seg_count:3d}] "
                f"Step: {step:<30s} "
                f"Conf: {conf:.3f} "
                f"Round-trip: {elapsed:.0f}ms "
                f"(inference: {latency:.0f}ms) "
                f"| Top3: {top3}"
            )

            # Collect for summary
            all_results.append({
                "seg": seg_count,
                "time": time.time(),
                "step": step,
                "step_idx": response.get("step_idx"),
                "confidence": conf,
                "belief": belief,
                "round_trip_ms": round(elapsed, 1),
                "inference_ms": round(latency, 1),
            })

        # Save session summary
        if all_results:
            _save_session_summary(args, all_results)

    except ConnectionError as e:
        print(f"Connection lost: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        sock.close()
        print("Connection closed.")


def _save_session_summary(args, results: list):
    """Save session results to a JSON summary file and print a recap table."""
    import os
    from datetime import datetime

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = os.path.basename(args.video_path) if args.video_path else "webcam"
    summary_path = os.path.join(args.log_dir, f"session_{timestamp}_{source}.json")

    # Compute summary stats
    confidences = [r["confidence"] for r in results]
    latencies = [r["inference_ms"] for r in results]

    # Step transitions
    transitions = []
    prev_step = None
    for r in results:
        if r["step"] != prev_step:
            transitions.append({"seg": r["seg"], "from": prev_step, "to": r["step"]})
            prev_step = r["step"]

    summary = {
        "timestamp": timestamp,
        "source": args.video_path or "webcam",
        "server": f"{args.host}:{args.port}",
        "segment_duration": args.segment_duration,
        "sampling_fps": args.sampling_fps,
        "total_segments": len(results),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "max_confidence": round(max(confidences), 4),
        "min_confidence": round(min(confidences), 4),
        "avg_inference_ms": round(sum(latencies) / len(latencies), 1),
        "step_transitions": transitions,
        "results": results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" Session Summary: {summary_path}")
    print(f"{'='*60}")
    print(f"  Segments:     {summary['total_segments']}")
    print(f"  Avg Conf:     {summary['avg_confidence']:.3f}")
    print(f"  Max Conf:     {summary['max_confidence']:.3f}")
    print(f"  Avg Latency:  {summary['avg_inference_ms']:.0f}ms")
    print(f"  Transitions:")
    for tr in transitions:
        arrow = f"{tr['from'] or 'START'} → {tr['to']}"
        print(f"    Seg {tr['seg']:3d}: {arrow}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
