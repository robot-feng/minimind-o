"""Small, dependency-free helpers for validating training progress."""

import math


class LossStabilityTracker:
    """Require consecutive finite loss reports at the configured log cadence."""

    def __init__(self, required_observations=5, log_interval=100):
        if required_observations < 1 or log_interval < 1:
            raise ValueError("required_observations and log_interval must be positive")
        self.required_observations = required_observations
        self.log_interval = log_interval
        self.last_step = None
        self.observations = 0

    def observe(self, step, total_steps, loss):
        if not math.isfinite(loss):
            raise ValueError(f"loss must be finite, got {loss}")
        if step < 1 or total_steps < step:
            raise ValueError("step must be within the training run")

        expected_step = self.last_step + self.log_interval if self.last_step is not None else None
        self.observations = self.observations + 1 if step == expected_step else 1
        self.last_step = step
        return self.observations >= self.required_observations
