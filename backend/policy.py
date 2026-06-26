from __future__ import annotations

from typing import Any

import numpy as np

from .modeling import ModelArtifact
from .schemas import PolicyRequest


def simulate_policy(artifact: ModelArtifact, policy: PolicyRequest) -> dict[str, Any]:
    probabilities = artifact.pipeline.predict_proba(artifact.test_frame)[:, 1]
    actual_default = artifact.test_target.to_numpy()
    loan_amounts = artifact.test_frame["loan_amount"].to_numpy(dtype=float)

    approved = probabilities < policy.approve_below
    rejected = probabilities > policy.reject_above
    manual_review = ~(approved | rejected)

    true_approved = int(((actual_default == 0) & approved).sum())
    false_approved = int(((actual_default == 1) & approved).sum())
    true_rejected = int(((actual_default == 1) & rejected).sum())
    false_rejected = int(((actual_default == 0) & rejected).sum())

    interest_revenue = float((loan_amounts[(actual_default == 0) & approved] * policy.interest_margin).sum())
    expected_loss = float((loan_amounts[approved] * probabilities[approved] * policy.loss_given_default).sum())
    realized_loss_proxy = float((loan_amounts[(actual_default == 1) & approved] * policy.loss_given_default).sum())
    manual_review_cost_total = float(policy.manual_review_cost * manual_review.sum())
    false_rejection_opportunity_cost = float(
        (loan_amounts[(actual_default == 0) & rejected] * policy.interest_margin).sum()
    )
    estimated_profit = interest_revenue - expected_loss - manual_review_cost_total

    return {
        "approval_rate": float(approved.mean()),
        "manual_review_rate": float(manual_review.mean()),
        "rejection_rate": float(rejected.mean()),
        "expected_default_rate_approved": float(actual_default[approved].mean()) if approved.any() else 0.0,
        "predicted_default_rate_approved": float(probabilities[approved].mean()) if approved.any() else 0.0,
        "estimated_profit": float(estimated_profit),
        "expected_loss": expected_loss,
        "manual_review_cost_total": manual_review_cost_total,
        "false_rejection_opportunity_cost": false_rejection_opportunity_cost,
        "confusion_matrix": {
            "true_approved": true_approved,
            "false_approved": false_approved,
            "true_rejected": true_rejected,
            "false_rejected": false_rejected,
        },
        "counts": {
            "approved": int(approved.sum()),
            "manual_review": int(manual_review.sum()),
            "rejected": int(rejected.sum()),
            "total": int(len(probabilities)),
            "observed_defaults": int(np.sum(actual_default)),
        },
        "realized_loss_proxy": realized_loss_proxy,
    }

