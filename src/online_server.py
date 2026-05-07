"""
BaGLM Online Inference Server.

Receives camera frames from a robot via TCP socket, runs LMM inference
(VSG + Progress) and incremental Bayesian filtering, and returns the
current step belief back to the robot.

Protocol (length-prefixed JSON + binary):
    1. Client sends: 4-byte big-endian length + JSON header
       Header: {"n_frames": N, "frame_h": H, "frame_w": W, "frame_dtype": "uint8"}
    2. Client sends: raw frame data (N * H * W * 3 bytes)
    3. Server replies: 4-byte big-endian length + JSON response
       Response: {"step": "...", "step_idx": 0, "confidence": 0.85,
                  "belief": {"step_a": 0.1, ...}, "timestamp": 1234.5}

Usage:
    python src/online_server.py \
        --task_config configs/restock_cola.json \
        --prereq_dir /path/to/results/prereq \
        --port 9999

    # Or with pre-computed dependency matrix:
    python src/online_server.py \
        --task_config configs/restock_cola.json \
        --dep_matrix_path /path/to/matrix.pt \
        --port 9999
"""

import argparse
import io
import json
import logging
import os
import struct
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F

import t2v_metrics
from bayes_filter import safe_normalize
from online_bayes import IncrementalBayesFilter
from utils.text_utils import format_questions

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ------------------------------------------------------------------
# Networking helpers
# ------------------------------------------------------------------

def recv_exact(conn, n: int) -> bytes:
    """Receive exactly *n* bytes from a socket connection."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(conn) -> bytes:
    """Receive one length-prefixed message."""
    raw_len = recv_exact(conn, 4)
    msg_len = struct.unpack("!I", raw_len)[0]
    return recv_exact(conn, msg_len)


def send_msg(conn, data: bytes):
    """Send one length-prefixed message."""
    conn.sendall(struct.pack("!I", len(data)) + data)


# ------------------------------------------------------------------
# Frame -> LMM scores
# ------------------------------------------------------------------

class OnlineInferenceEngine:
    """Holds the LMM model and runs VSG + Progress on raw frames."""

    def __init__(
        self,
        model_name: str = "internvl2.5-8b",
        cache_dir: Optional[str] = None,
        vsg_prompt_file: Optional[str] = None,
        prog_prompt_file: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name

        logger.info(f"Loading LMM model: {model_name} ...")
        self.score_func = t2v_metrics.get_score_model(
            model=model_name, device=device, cache_dir=cache_dir,
        )
        self.preprocess_fn = self.score_func.model.get_preprocessor()
        logger.info("LMM model loaded.")

        self.vsg_template = self._load_prompt(vsg_prompt_file, "prompts/vsg/question_robot.txt")
        self.prog_template = self._load_prompt(prog_prompt_file, "prompts/prog/question_robot.txt")
        self.verify_template = self._load_prompt(None, "prompts/verify/question_robot.txt")

    @staticmethod
    def _load_prompt(path: Optional[str], default: str) -> str:
        if path and os.path.isfile(path):
            with open(path) as f:
                return f.read()
        fallback = os.path.join(os.path.dirname(__file__), "..", default)
        if os.path.isfile(fallback):
            with open(fallback) as f:
                return f.read()
        raise FileNotFoundError(
            f"Prompt file not found: tried {path} and {fallback}"
        )

    def frames_to_segments(
        self,
        frames_np: np.ndarray,
        segment_duration: int = 2,
        sampling_fps: int = 2,
    ) -> list:
        """Convert raw numpy frames to pre-processed tensor segments.

        Args:
            frames_np: [N, H, W, 3] uint8 numpy array from the robot camera.
            segment_duration: Number of seconds per segment (determines segment length).
            sampling_fps: How many frames per second to use.

        Returns:
            List of [frames_per_segment, C, H', W'] tensors (one per segment).
        """
        n_frames = frames_np.shape[0]

        # Sub-sample to target sampling_fps (assume input is ~30fps or similar)
        # We take evenly spaced frames to match n_frames * (sampling_fps / input_fps),
        # but since we don't know input fps exactly, we just use all frames and let
        # the caller control how many they send.
        # A simple approach: the client sends exactly `segment_duration * sampling_fps`
        # frames per request, so n_frames IS the segment.

        frames_per_segment = segment_duration * sampling_fps

        if n_frames < frames_per_segment:
            logger.warning(
                f"Only {n_frames} frames received, need {frames_per_segment}. "
                "Padding by repeating last frame."
            )
            pad = np.tile(frames_np[-1:], (frames_per_segment - n_frames, 1, 1, 1))
            frames_np = np.concatenate([frames_np, pad], axis=0)
            n_frames = frames_np.shape[0]

        # Convert each frame: HWC uint8 numpy -> CHW tensor -> preprocess
        all_preprocessed = []
        for i in range(n_frames):
            img_tensor = torch.from_numpy(frames_np[i].copy()).permute(2, 0, 1)  # CHW
            processed = self.preprocess_fn(img_tensor, nframes=frames_per_segment)
            all_preprocessed.append(processed)

        # Build segments
        segments = []
        for i in range(0, len(all_preprocessed), frames_per_segment):
            seg = all_preprocessed[i : i + frames_per_segment]
            if len(seg) == frames_per_segment:
                segments.append(torch.cat(seg))

        return segments

    @torch.no_grad()
    def infer_vsg(
        self,
        segments: list,
        goal: str,
        step_headline: list,
        visual_batch_size: int = 1,
    ) -> torch.Tensor:
        """Run VSG inference on segments.

        Returns:
            scores: [T, S+1] tensor.
        """
        texts = [{"goal": goal, "step": step} for step in step_headline]
        batch = {
            "videos": segments,
            "texts": texts,
        }
        scores = self.score_func.forward_vsg(
            batch, visual_batch_size, question_template=self.vsg_template,
        )
        return scores.cpu()

    @torch.no_grad()
    def infer_progress(
        self,
        segments: list,
        goal: str,
        step_headline: list,
        visual_batch_size: int = 1,
        text_batch_size: int = 1,
    ) -> torch.Tensor:
        """Run Progress inference on segments.

        Returns:
            scores: [T, S, 10] tensor.
        """
        texts = [{"goal": goal, "step": step} for step in step_headline]
        batch = {
            "videos": segments,
            "texts": texts,
        }
        scores = self.score_func.forward_progress(
            batch,
            visual_batch_size,
            text_batch_size,
            question_template=self.prog_template,
        )
        return scores.cpu()

    @torch.no_grad()
    def infer_verify(
        self,
        segments: list,
        success_criteria: str,
    ) -> dict:
        """Run step verification on segments.

        Uses LMM to judge whether the step was completed successfully
        based on the provided success criteria.

        Args:
            segments: List of pre-processed video segment tensors.
            success_criteria: Description of what a successful completion looks like.

        Returns:
            dict with keys:
                - "success": bool, whether the step was completed successfully.
                - "confidence": float, probability of success.
                - "answer": str, raw answer from the model ("A" or "B").
        """
        # Use the last segment (most recent frames) for verification
        last_segment = segments[-1:]

        # Build the full question with choices embedded
        question = self.verify_template.format(success_criteria=success_criteria)

        # Use forward_multi_choice with progress=True path.
        # That path does: questions = [question_template.format(**fields)]
        # and answer_labels = [[str(n) for n in range(10)]]
        # We repurpose it by passing our verify question via {step} field,
        # and manually extracting A/B probabilities from the 10-bin output.
        #
        # However, a cleaner approach: use the model's forward() method
        # which returns the probability of a given answer token.
        # We call it twice: once for "A", once for "B".

        prob_a = self.score_func.model.forward(
            last_segment,
            [{"success_criteria": success_criteria}],
            question_template=self.verify_template,
            answer_template="A",
        )[0].item()

        prob_b = self.score_func.model.forward(
            last_segment,
            [{"success_criteria": success_criteria}],
            question_template=self.verify_template,
            answer_template="B",
        )[0].item()

        total = prob_a + prob_b
        success_prob = prob_a / total if total > 0 else 0.5

        answer = "A" if success_prob >= 0.5 else "B"

        return {
            "success": answer == "A",
            "confidence": round(success_prob, 4),
            "answer": answer,
        }


# ------------------------------------------------------------------
# Main server loop
# ------------------------------------------------------------------

def handle_client(conn, engine: OnlineInferenceEngine, bayes: IncrementalBayesFilter,
                  task_config: dict, args, session_log_path: Optional[str] = None):
    """Handle one client connection in a loop until disconnect."""
    goal = task_config["activity"]
    step_headline = task_config["step_headline"]
    step_success_criteria = task_config.get("step_success_criteria", None)
    verify_enabled = step_success_criteria is not None and len(step_success_criteria) == len(step_headline)
    segment_duration = args.segment_duration
    sampling_fps = args.sampling_fps
    frames_per_segment = segment_duration * sampling_fps

    if verify_enabled:
        logger.info("Step verification ENABLED")
    else:
        logger.info("Step verification DISABLED (no step_success_criteria in config)")

    # Open session log file (JSONL)
    log_file = None
    if session_log_path:
        os.makedirs(os.path.dirname(session_log_path), exist_ok=True)
        log_file = open(session_log_path, "a")

    logger.info(
        f"Session started — goal: {goal}, steps: {step_headline}, "
        f"frames_per_segment: {frames_per_segment}"
    )

    while True:
        try:
            # 1. Receive header
            header_raw = recv_msg(conn)
            header = json.loads(header_raw)
            n_frames = header["n_frames"]
            frame_h = header.get("frame_h", 480)
            frame_w = header.get("frame_w", 640)

            expected_bytes = n_frames * frame_h * frame_w * 3

            # 2. Receive frame data
            frame_data = recv_exact(conn, expected_bytes)
            frames_np = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                n_frames, frame_h, frame_w, 3
            )

            recv_ts = time.time()
            logger.info(f"Received {n_frames} frames ({frame_h}x{frame_w})")

            # 3. Run inference
            segments = engine.frames_to_segments(
                frames_np, segment_duration, sampling_fps,
            )

            vsg_scores = engine.infer_vsg(
                segments, goal, step_headline, args.visual_batch_size,
            )
            prog_scores = engine.infer_progress(
                segments, goal, step_headline,
                args.visual_batch_size, args.text_batch_size,
            )

            # 4. Update Bayesian filter (process each segment)
            T = vsg_scores.shape[0]
            verify_result = None
            for t in range(T):
                bayes.update(vsg_scores[t], prog_scores[t])

                # Check if a step transition occurred -> verify the previous step
                if verify_enabled and bayes.last_transition is not None:
                    trans = bayes.last_transition
                    prev_idx = trans["from_idx"]
                    prev_step = trans["from_step"]
                    criteria = step_success_criteria[prev_idx]

                    logger.info(
                        f"Step transition detected: {prev_step} → {trans['to_step']}, "
                        f"verifying '{prev_step}' ..."
                    )
                    try:
                        verify_result = engine.infer_verify(segments, criteria)
                        verify_result["verified_step"] = prev_step
                        verify_result["verified_step_idx"] = prev_idx
                        verify_result["transition_to"] = trans["to_step"]

                        logger.info(
                            f"  Verify '{prev_step}': "
                            f"{'SUCCESS' if verify_result['success'] else 'FAILED'} "
                            f"(conf={verify_result['confidence']:.3f})"
                        )
                    except Exception as e:
                        logger.warning(f"  Verification failed with error: {e}")
                        verify_result = None

            # 5. Build response
            response = {
                "step": bayes.current_step(),
                "step_idx": bayes.current_step_index(),
                "confidence": bayes.current_confidence(),
                "belief": bayes.belief_dict(),
                "segment_count": T,
                "total_segments": bayes.step_count,
                "timestamp": time.time(),
                "latency_ms": (time.time() - recv_ts) * 1000,
            }

            # Add verification result if triggered
            if verify_result is not None:
                response["step_verification"] = verify_result

            send_msg(conn, json.dumps(response).encode("utf-8"))

            # Write to session log
            if log_file:
                log_entry = {
                    "seg": bayes.step_count,
                    "time": datetime.now().isoformat(),
                    "step": response["step"],
                    "step_idx": response["step_idx"],
                    "confidence": response["confidence"],
                    "belief": response["belief"],
                    "latency_ms": round(response["latency_ms"], 1),
                }
                if verify_result is not None:
                    log_entry["verification"] = verify_result
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

            logger.info(
                f"Step: {response['step']}  "
                f"Conf: {response['confidence']:.3f}  "
                f"Latency: {response['latency_ms']:.0f}ms"
            )

        except ConnectionError:
            logger.info("Client disconnected.")
            break
        except struct.error:
            logger.info("Client disconnected (struct error).")
            break
        except Exception:
            logger.exception("Error during inference")
            try:
                error_resp = json.dumps({"error": "inference_failed"}).encode()
                send_msg(conn, error_resp)
            except Exception:
                pass
            break

    if log_file:
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description="BaGLM Online Inference Server")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=9999, type=int)
    parser.add_argument("--task_config", required=True, type=str,
                        help="Path to task config JSON")
    parser.add_argument("--dep_matrix_path", default=None, type=str,
                        help="Direct path to dependency matrix .pt")
    parser.add_argument("--prereq_dir", default=None, type=str,
                        help="Dir with prereq .pt files (used if --dep_matrix_path not set)")
    parser.add_argument("--model", default="internvl2.5-8b", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--segment_duration", default=2, type=int)
    parser.add_argument("--sampling_fps", default=2, type=int)
    parser.add_argument("--visual_batch_size", default=1, type=int)
    parser.add_argument("--text_batch_size", default=1, type=int)
    parser.add_argument("--vsg_prompt_file", default=None, type=str)
    parser.add_argument("--prog_prompt_file", default=None, type=str)
    parser.add_argument("--log_dir", default="logs/online", type=str,
                        help="Directory to save session logs (JSONL format)")
    args = parser.parse_args()

    # Load task config
    with open(args.task_config) as f:
        task_config = json.load(f)

    # Load or create dependency matrix
    if args.dep_matrix_path:
        dep_matrix = torch.load(args.dep_matrix_path, map_location="cpu", weights_only=True)
    elif args.prereq_dir:
        activity = task_config["activity"]
        variation = task_config.get("variation", "none")
        prereq_path = os.path.join(
            args.prereq_dir, args.model, f"{activity}_{variation}.pt"
        )
        dep_matrix = torch.load(prereq_path, map_location="cpu", weights_only=True)
    else:
        logger.warning(
            "No dependency matrix provided — using identity matrix "
            "(no step-order constraints)."
        )
        S = len(task_config["step_headline"])
        dep_matrix = torch.eye(S)

    # Initialize engine + filter
    engine = OnlineInferenceEngine(
        model_name=args.model,
        cache_dir=args.cache_dir,
        vsg_prompt_file=args.vsg_prompt_file,
        prog_prompt_file=args.prog_prompt_file,
        device=args.device,
    )

    bayes = IncrementalBayesFilter(
        dependency_matrix=dep_matrix,
        step_headline=task_config["step_headline"],
        device="cpu",
    )

    # Start TCP server
    import socket

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    logger.info(f"BaGLM server listening on {args.host}:{args.port}")
    logger.info(f"Task: {task_config['activity']}")
    logger.info(f"Steps: {task_config['step_headline']}")

    try:
        while True:
            conn, addr = srv.accept()
            logger.info(f"Connection from {addr}")
            # Reset belief for new session
            bayes.reset()

            # Generate log path for this session
            session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_log_path = os.path.join(
                args.log_dir,
                task_config["activity"].replace(" ", "_"),
                f"{session_tag}.jsonl",
            )
            logger.info(f"Session log: {session_log_path}")

            handle_client(conn, engine, bayes, task_config, args, session_log_path)
            conn.close()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
