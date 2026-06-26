from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
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
    }


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
    return ModelArtifact(
        pipeline=best_model,
        training_frame=x_train.copy(),
        test_frame=x_test.copy(),
        test_target=y_test.copy(),
        metrics=metrics,
        feature_importance=importance,
        reference_values=reference_profile(x_train),
        input_ranges=training_ranges(x_train),
    )
