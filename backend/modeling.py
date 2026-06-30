from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    DATA_DICTIONARY,
    FEATURES,
    LEAKAGE_EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
    generate_synthetic_loan_data,
    reference_profile,
    training_ranges,
)


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Any
    training_frame: pd.DataFrame
    test_frame: pd.DataFrame
    test_target: pd.Series
    metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    reference_values: dict[str, Any]
    input_ranges: dict[str, Any]
    artifact_metadata: dict[str, Any]


ARTIFACT_METADATA_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model_metadata.json"


def stable_json_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-serializable metadata."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_frame_hash(frame: pd.DataFrame, target: pd.Series) -> str:
    """Hash the exact split used by a demo model artifact."""
    training_payload = frame.copy().reset_index(drop=True)
    training_payload["default"] = target.reset_index(drop=True).astype(int)
    records = training_payload.sort_index(axis=1).to_dict(orient="records")
    return stable_json_hash(records)


def build_artifact_metadata(
    *,
    selected_model: str,
    metrics: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    training_hash = training_frame_hash(x_train, y_train)
    holdout_hash = training_frame_hash(x_test, y_test)
    metadata = {
        "artifact_type": "in-memory sklearn pipeline with persisted metadata sidecar",
        "metadata_schema_version": "1.0",
        "model_version": metrics["model_version"],
        "selected_model": selected_model,
        "created_at_utc": "deterministic synthetic demo generated at application startup",
        "training_hash_sha256": training_hash,
        "holdout_hash_sha256": holdout_hash,
        "feature_schema_hash_sha256": stable_json_hash(
            {
                "features": FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "excluded_leakage_features": LEAKAGE_EXCLUDED_FEATURES,
            }
        ),
        "training_rows": int(len(x_train)),
        "holdout_rows": int(len(x_test)),
        "synthetic_data_generator": "backend.data.generate_synthetic_loan_data(n_rows=2400, seed=7)",
        "split": {"test_size": 0.25, "stratified": True, "random_state": 11},
        "libraries": {
            "python_requires": ">=3.10",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "disclaimer": "Synthetic demo artifact metadata only; not valid for real lending decisions.",
    }
    metadata["artifact_metadata_hash_sha256"] = stable_json_hash(
        {key: value for key, value in metadata.items() if key != "artifact_metadata_hash_sha256"}
    )
    return metadata


def persist_artifact_metadata(metadata: dict[str, Any], path: Path = ARTIFACT_METADATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def evaluate_classifier(model: Any, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.2) -> dict[str, Any]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "precision_at_20pct": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_at_20pct": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_at_20pct": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_at_20pct": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def selected_model_diagnostics(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    probabilities = model.predict_proba(x_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
    return {
        "calibration_curve": [
            {"mean_predicted_probability": float(pred), "observed_default_rate": float(obs)}
            for pred, obs in zip(mean_pred, frac_pos)
        ],
        "score_distribution": {
            "p05": float(np.quantile(probabilities, 0.05)),
            "p25": float(np.quantile(probabilities, 0.25)),
            "p50": float(np.quantile(probabilities, 0.5)),
            "p75": float(np.quantile(probabilities, 0.75)),
            "p95": float(np.quantile(probabilities, 0.95)),
        },
        "default_rate": float(y_test.mean()),
        "proxy_group_review": proxy_group_review(x_test, probabilities),
        "threshold_stress": threshold_stress(probabilities),
    }


def proxy_group_review(x_test: pd.DataFrame, probabilities: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """Summarize score behavior by non-protected synthetic proxy groups.

    These are not protected-class fairness results. They provide a repeatable check
    that score and approval rates are inspectable by application fields before any
    real fairness analysis is attempted.
    """
    review: dict[str, list[dict[str, Any]]] = {}
    scored = x_test.copy()
    scored["predicted_default_probability"] = probabilities
    scored["approved_at_8pct"] = probabilities < 0.08
    for feature in ["home_ownership", "loan_purpose"]:
        groups = []
        for value, group in scored.groupby(feature, sort=True):
            groups.append(
                {
                    "feature_value": str(value),
                    "count": int(len(group)),
                    "mean_predicted_default_probability": float(group["predicted_default_probability"].mean()),
                    "approval_rate_at_8pct": float(group["approved_at_8pct"].mean()),
                }
            )
        review[feature] = groups
    return review


def threshold_stress(probabilities: np.ndarray) -> list[dict[str, float]]:
    """Summarize deterministic approval/review/rejection rates over threshold pairs."""
    scenarios = [(0.05, 0.15), (0.08, 0.20), (0.10, 0.25), (0.12, 0.30)]
    return [
        {
            "approve_below": approve_below,
            "reject_above": reject_above,
            "approval_rate": float((probabilities < approve_below).mean()),
            "manual_review_rate": float(((probabilities >= approve_below) & (probabilities <= reject_above)).mean()),
            "rejection_rate": float((probabilities > reject_above).mean()),
        }
        for approve_below, reject_above in scenarios
    ]


def compute_feature_importance(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, Any]]:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=6,
        random_state=17,
        scoring="roc_auc",
    )
    ranked = sorted(
        zip(FEATURES, result.importances_mean, result.importances_std),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "feature": feature,
            "importance": float(max(mean, 0.0)),
            "std": float(std),
            "interpretation": feature_importance_note(feature),
        }
        for feature, mean, std in ranked
    ]


def feature_importance_note(feature: str) -> str:
    notes = {
        "debt_to_income": "Higher values generally indicate less repayment capacity.",
        "revolving_utilization": "Higher utilization is usually associated with credit stress.",
        "interest_rate": "Higher rates often reflect risk already priced at origination.",
        "annual_income": "Higher income usually improves repayment capacity.",
        "delinquencies_2y": "Recent delinquencies are associated with elevated default risk.",
        "term_months": "Longer terms can increase uncertainty and loss exposure.",
        "loan_purpose": "Some purposes carry different observed risk in the demo sample.",
        "home_ownership": "Housing status can proxy financial stability in the demo sample.",
    }
    return notes.get(feature, "Model-agnostic importance measured on the held-out demo set.")


def train_demo_model() -> ModelArtifact:
    data = generate_synthetic_loan_data()
    x = data[FEATURES]
    y = data["default"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=11,
    )

    baseline = build_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced"))
    forest = build_pipeline(
        RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=18,
            random_state=11,
            class_weight="balanced_subsample",
        )
    )
    calibrated_forest = CalibratedClassifierCV(estimator=forest, cv=3, method="isotonic")

    candidates = {
        "logistic_regression": baseline,
        "calibrated_random_forest": calibrated_forest,
    }
    scores: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        scores[name] = evaluate_classifier(candidate, x_test, y_test)

    best_name = max(scores, key=lambda name: (scores[name]["roc_auc"], -scores[name]["brier_score"]))
    best_model = candidates[best_name]
    diagnostics = selected_model_diagnostics(best_model, x_test, y_test)
    importance = compute_feature_importance(best_model, x_test, y_test)
    probabilities = best_model.predict_proba(x_test)[:, 1]
    reference_policy = {
        "approve_below": 0.08,
        "reject_above": 0.20,
        "approval_rate": float((probabilities < 0.08).mean()),
        "manual_review_rate": float(((probabilities >= 0.08) & (probabilities <= 0.20)).mean()),
        "rejection_rate": float((probabilities > 0.20).mean()),
    }
    metrics = {
        "model_version": f"credit-risk-{best_name}-v1-demo",
        "selected_model": best_name,
        "target_definition": "1 = default or charge-off equivalent; 0 = fully paid equivalent",
        "models": scores,
        "selected_model_diagnostics": diagnostics,
        "policy_reference": reference_policy,
        "data": {
            "dataset_source": "Synthetic LendingClub-like local demo sample.",
            "row_count": int(len(data)),
            "default_rate": float(y.mean()),
            "features": FEATURES,
            "data_dictionary": DATA_DICTIONARY,
            "excluded_leakage_features": LEAKAGE_EXCLUDED_FEATURES,
        },
    }
    artifact_metadata = build_artifact_metadata(
        selected_model=best_name,
        metrics=metrics,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    metrics["artifact_metadata"] = artifact_metadata
    persist_artifact_metadata(artifact_metadata)
    return ModelArtifact(
        pipeline=best_model,
        training_frame=x_train.copy(),
        test_frame=x_test.copy(),
        test_target=y_test.copy(),
        metrics=metrics,
        feature_importance=importance,
        reference_values=reference_profile(x_train),
        input_ranges=training_ranges(x_train),
        artifact_metadata=artifact_metadata,
    )
