"""Small, dependency-free reward calibration summaries for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


def roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    """Return tie-aware ROC-AUC, or None when either class is absent."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None

    ranked = sorted(zip(scores, labels), key=lambda sample: sample[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ranked[start:end])
        start = end

    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


@dataclass
class RewardProbeStats:
    count: int = 0
    target_sum: float = 0.0
    one_step_prior_prediction_sum: float = 0.0
    posterior_prediction_sum: float = 0.0
    one_step_prior_absolute_error_sum: float = 0.0
    posterior_absolute_error_sum: float = 0.0
    zero_absolute_error_sum: float = 0.0

    def record(
        self,
        target: float,
        one_step_prior_prediction: float,
        posterior_prediction: float,
    ) -> None:
        self.count += 1
        self.target_sum += target
        self.one_step_prior_prediction_sum += one_step_prior_prediction
        self.posterior_prediction_sum += posterior_prediction
        self.one_step_prior_absolute_error_sum += abs(
            one_step_prior_prediction - target
        )
        self.posterior_absolute_error_sum += abs(posterior_prediction - target)
        self.zero_absolute_error_sum += abs(target)

    def summary(self) -> dict[str, object]:
        if not self.count:
            return {
                "count": 0,
                "target_mean": None,
                "one_step_prior_prediction_mean": None,
                "posterior_prediction_mean": None,
                "one_step_prior_mae": None,
                "posterior_mae": None,
                "zero_predictor_mae": None,
            }
        count = float(self.count)
        return {
            "count": self.count,
            "target_mean": self.target_sum / count,
            "one_step_prior_prediction_mean": (
                self.one_step_prior_prediction_sum / count
            ),
            "posterior_prediction_mean": self.posterior_prediction_sum / count,
            "one_step_prior_mae": (self.one_step_prior_absolute_error_sum / count),
            "posterior_mae": self.posterior_absolute_error_sum / count,
            "zero_predictor_mae": self.zero_absolute_error_sum / count,
        }


@dataclass
class RewardProbe:
    overall: RewardProbeStats = field(default_factory=RewardProbeStats)
    by_reward_sign: dict[str, RewardProbeStats] = field(
        default_factory=lambda: {
            "positive": RewardProbeStats(),
            "zero": RewardProbeStats(),
            "negative": RewardProbeStats(),
        }
    )
    samples: list[tuple[float, float, float]] = field(default_factory=list)

    def record(
        self,
        target: float,
        one_step_prior_prediction: float,
        posterior_prediction: float,
    ) -> None:
        if not all(
            math.isfinite(value)
            for value in (target, one_step_prior_prediction, posterior_prediction)
        ):
            raise ValueError("reward probe requires finite targets and predictions")
        if target > 0.0:
            sign = "positive"
        elif target < 0.0:
            sign = "negative"
        else:
            sign = "zero"
        self.overall.record(target, one_step_prior_prediction, posterior_prediction)
        self.by_reward_sign[sign].record(
            target, one_step_prior_prediction, posterior_prediction
        )
        self.samples.append((target, one_step_prior_prediction, posterior_prediction))

    def summary(self) -> dict[str, object]:
        overall = self.overall.summary()
        return {
            "samples": self.overall.count,
            "one_step_prior_mae": overall["one_step_prior_mae"],
            "posterior_mae": overall["posterior_mae"],
            "zero_predictor_mae": overall["zero_predictor_mae"],
            "by_reward_sign": {
                sign: stats.summary() for sign, stats in self.by_reward_sign.items()
            },
            "one_step_prior_ranking": self._ranking_summary(1),
            "posterior_ranking": self._ranking_summary(2),
        }

    def _ranking_summary(self, prediction_index: int) -> dict[str, float | None]:
        targets = [sample[0] for sample in self.samples]
        predictions = [sample[prediction_index] for sample in self.samples]
        nonzero = [target != 0.0 for target in targets]
        nonzero_predictions = [
            prediction
            for prediction, is_nonzero in zip(predictions, nonzero)
            if is_nonzero
        ]
        nonzero_targets = [target for target in targets if target != 0.0]
        return {
            "positive_vs_rest_roc_auc": roc_auc(
                [target > 0.0 for target in targets], predictions
            ),
            "negative_vs_rest_roc_auc": roc_auc(
                [target < 0.0 for target in targets],
                [-prediction for prediction in predictions],
            ),
            "reward_event_vs_zero_roc_auc": roc_auc(
                nonzero, [abs(prediction) for prediction in predictions]
            ),
            "positive_vs_negative_roc_auc": roc_auc(
                [target > 0.0 for target in nonzero_targets],
                nonzero_predictions,
            ),
        }
