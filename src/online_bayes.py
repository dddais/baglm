"""
Incremental Bayesian filter for online step grounding.

Maintains belief state across time steps. Each call to `update()` processes
one time segment and returns the updated belief. Suitable for real-time use
where frames arrive sequentially from a robot camera.
"""

import json
from typing import List, Optional

import torch

from bayes_filter import (
    build_successor_matrix,
    compute_readiness_validity,
    safe_normalize,
)


class IncrementalBayesFilter:
    """Stateful Bayesian filter that updates incrementally, one segment at a time."""

    def __init__(
        self,
        dependency_matrix: torch.Tensor,
        step_headline: List[str],
        device: str = "cpu",
    ):
        """
        Args:
            dependency_matrix: [S, S] prerequisite dependency matrix for the task.
            step_headline: Ordered list of step names (length S).
            device: Torch device.
        """
        self.device = device
        self.step_headline = step_headline
        self.S = len(step_headline)

        self.dependency_matrix = dependency_matrix.to(device)
        self.successor_matrix = build_successor_matrix(dependency_matrix).to(device)
        self.static_transition = self.successor_matrix / self.successor_matrix.sum(
            dim=1, keepdim=True
        )

        # State: belief over S steps + 1 "none" column
        self.belief = torch.ones(self.S + 1, device=device) / float(self.S + 1)

        # Cumulative trackers (monotonically non-decreasing)
        self.max_vsg_scores = torch.zeros(self.S, device=device)
        self.monotonic_progress = torch.zeros(self.S, device=device)

        self.step_count = 0
        self.history: List[torch.Tensor] = [self.belief.clone()]
        self.last_transition = None  # Set by update() when step changes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        vsg_scores: torch.Tensor,
        prog_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Process one time segment and return the posterior belief.

        Args:
            vsg_scores: [S+1] raw LMM scores for this segment.
            prog_scores: [S, 10] progress probability distribution per step.

        Returns:
            belief: [S+1] posterior belief after this segment.

        Side-effect:
            Sets self.last_transition to a dict if a step transition occurred,
            or None otherwise. Check self.last_transition after calling update().
        """
        # Record previous step before updating
        prev_step_idx = self.current_step_index()

        vsg_scores = vsg_scores.to(self.device)
        prog_scores = prog_scores.to(self.device)

        # --- replicate the logic from run_bayes_filter one time-step ---

        # Monotonic VSG: keep running maximum
        self.max_vsg_scores = torch.maximum(self.max_vsg_scores, vsg_scores[:-1])

        # Expected progress from 10-bin distribution
        bins = torch.arange(10, device=self.device, dtype=prog_scores.dtype)
        expected_progress = (prog_scores * bins).sum(dim=1) / 9.0
        self.monotonic_progress = torch.maximum(
            self.monotonic_progress, expected_progress
        )

        step_readiness, step_validity = compute_readiness_validity(
            self.dependency_matrix, self.monotonic_progress
        )

        # Build time-dependent transition matrix
        transition_t = self.static_transition.clone()
        transition_t[:, :-1] = transition_t[:, :-1] * (step_readiness * step_validity)
        transition_t = safe_normalize(transition_t, dim=1)

        # Predict
        prior = self.belief @ transition_t

        # Update
        posterior = safe_normalize(prior * vsg_scores)

        self.belief = posterior
        self.step_count += 1
        self.history.append(self.belief.clone())

        # Detect step transition
        new_step_idx = self.current_step_index()
        if new_step_idx != prev_step_idx:
            self.last_transition = {
                "from_idx": prev_step_idx,
                "from_step": self.step_headline[prev_step_idx],
                "to_idx": new_step_idx,
                "to_step": self.step_headline[new_step_idx],
                "seg": self.step_count,
            }
        else:
            self.last_transition = None

        return self.belief

    def current_step(self) -> str:
        """Return the most likely step name (excluding 'none')."""
        idx = self.belief[:-1].argmax().item()
        return self.step_headline[idx]

    def current_step_index(self) -> int:
        """Return the most likely step index (0-based, excluding 'none')."""
        return self.belief[:-1].argmax().item()

    def current_confidence(self) -> float:
        """Return belief probability of the most likely step."""
        return self.belief[:-1].max().item()

    def belief_dict(self) -> dict:
        """Return current belief as a human-readable dict."""
        d = {name: round(self.belief[i].item(), 4) for i, name in enumerate(self.step_headline)}
        d["<none>"] = round(self.belief[-1].item(), 4)
        return d

    def reset(self):
        """Reset all internal state."""
        self.belief = torch.ones(self.S + 1, device=self.device) / float(self.S + 1)
        self.max_vsg_scores = torch.zeros(self.S, device=self.device)
        self.monotonic_progress = torch.zeros(self.S, device=self.device)
        self.step_count = 0
        self.history = [self.belief.clone()]
        self.last_transition = None

    @staticmethod
    def from_task_config(
        task_config_path: str,
        prereq_dir: str,
        model: str = "internvl2.5-8b",
        device: str = "cpu",
    ) -> "IncrementalBayesFilter":
        """Construct from a task config JSON and pre-computed prerequisite dir.

        Args:
            task_config_path: Path to task config JSON (see configs/ for format).
            prereq_dir: Directory containing `<activity>_<variation>.pt`.
            model: Model subdirectory inside prereq_dir (for path resolution).
            device: Torch device.

        Returns:
            Initialized IncrementalBayesFilter.
        """
        with open(task_config_path) as f:
            cfg = json.load(f)

        step_headline = cfg["step_headline"]
        activity = cfg["activity"]
        variation = cfg.get("variation", "none")

        prereq_path = f"{prereq_dir}/{model}/{activity}_{variation}.pt"
        dep_matrix = torch.load(prereq_path, map_location=device, weights_only=True)

        return IncrementalBayesFilter(
            dependency_matrix=dep_matrix,
            step_headline=step_headline,
            device=device,
        )
