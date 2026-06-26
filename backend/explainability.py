from __future__ import annotations

from typing import Any

import pandas as pd

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .modeling import ModelArtifact
from .schemas import Applicant


DISPLAY_NAMES = {
    "loan_amount": "Loan amount",
    "term_months": "Loan term",
    "interest_rate": "Interest rate",
    "annual_income": "Annual income",
    "debt_to_income": "Debt-to-income ratio",
    "employment_length_years": "Employment length",
    "home_ownership": "Home ownership",
    "loan_purpose": "Loan purpose",
    "revolving_utilization": "Revolving utilization",
    "open_accounts": "Open accounts",
    "delinquencies_2y": "Recent delinquencies",
}


def applicant_frame(applicant: Applicant) -> pd.DataFrame:
    return pd.DataFrame([applicant.model_dump()])[FEATURES]


def predict_probability(artifact: ModelArtifact, applicant: Applicant) -> float:
    return float(artifact.pipeline.predict_proba(applicant_frame(applicant))[0, 1])


def decision_band(probability: float) -> tuple[str, str]:
    if probability < 0.08:
        return "approve", "low"
    if probability <= 0.20:
        return "manual_review", "medium"
    return "reject", "high"


def local_drivers(artifact: ModelArtifact, applicant: Applicant, probability: float, limit: int = 5) -> list[dict[str, Any]]:
    row = applicant_frame(applicant)
    drivers: list[dict[str, Any]] = []
    for feature in FEATURES:
        reference_value = artifact.reference_values[feature]
        if row.iloc[0][feature] == reference_value:
            continue
        counterfactual = row.copy()
        counterfactual.loc[counterfactual.index[0], feature] = reference_value
        reference_probability = float(artifact.pipeline.predict_proba(counterfactual)[0, 1])
        impact = probability - reference_probability
        if abs(impact) < 0.003:
            continue
        drivers.append(
            {
                "feature": feature,
                "direction": "increases_risk" if impact > 0 else "decreases_risk",
                "explanation": explanation_for(feature, row.iloc[0][feature], reference_value, impact),
                "impact": float(round(impact, 4)),
            }
        )
    drivers.sort(key=lambda item: abs(item["impact"]), reverse=True)
    if drivers:
        return drivers[:limit]
    return [
        {
            "feature": "profile",
            "direction": "neutral",
            "explanation": "The applicant is close to the demo training reference profile across the strongest drivers.",
            "impact": 0.0,
        }
    ]


def explanation_for(feature: str, value: Any, reference_value: Any, impact: float) -> str:
    name = DISPLAY_NAMES.get(feature, feature)
    direction = "increased" if impact > 0 else "reduced"
    if feature in NUMERIC_FEATURES:
        if feature in {"interest_rate", "debt_to_income", "revolving_utilization"}:
            actual = f"{float(value):.1%}"
            reference = f"{float(reference_value):.1%}"
        elif feature in {"loan_amount", "annual_income"}:
            actual = f"${float(value):,.0f}"
            reference = f"${float(reference_value):,.0f}"
        else:
            actual = f"{float(value):.1f}"
            reference = f"{float(reference_value):.1f}"
        return f"{name} of {actual} {direction} risk versus the demo reference value of {reference}."
    if feature in CATEGORICAL_FEATURES:
        return f"{name} '{value}' {direction} risk versus the demo reference segment '{reference_value}'."
    return f"{name} {direction} estimated risk versus the demo reference profile."


def score_warnings(artifact: ModelArtifact, applicant: Applicant) -> list[str]:
    values = applicant.model_dump()
    warnings: list[str] = []
    for feature in NUMERIC_FEATURES:
        value = float(values[feature])
        bounds = artifact.input_ranges[feature]
        if value < bounds["p01"] or value > bounds["p99"]:
            warnings.append(
                f"{DISPLAY_NAMES.get(feature, feature)} is outside the central 98% range of the demo training sample."
            )
    for feature in CATEGORICAL_FEATURES:
        if values[feature] not in artifact.input_ranges[feature]:
            warnings.append(
                f"{DISPLAY_NAMES.get(feature, feature)} category '{values[feature]}' was not observed in the demo training sample."
            )
    if applicant.debt_to_income > 0.75 or applicant.revolving_utilization > 1.0:
        warnings.append("Applicant is in a high-stress credit range; local explanations may be less stable.")
    return warnings[:6]
