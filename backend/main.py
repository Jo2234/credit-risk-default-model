from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .data import (
    CATEGORICAL_FEATURES,
    DATA_DICTIONARY,
    FEATURES,
    LEAKAGE_EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
    generate_synthetic_loan_data,
)
from .explainability import decision_band, local_drivers
from .modeling import build_pipeline, train_demo_model
from .policy import simulate_policy
from .schemas import Applicant, BatchScoreRequest, MetricsResponse, PolicyRequest, PolicyResponse, ScoreResponse


app = FastAPI(
    title="Credit Risk Default Model",
    version="0.2.0",
    description="Local demo API for default scoring, model diagnostics, explanation drivers, and credit policy simulation.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


ARTIFACT = train_demo_model()
MODEL = ARTIFACT.pipeline
METRICS = ARTIFACT.metrics
X_TEST = ARTIFACT.test_frame
Y_TEST = ARTIFACT.test_target
DATA = generate_synthetic_loan_data()
NUMERIC = NUMERIC_FEATURES
CATEGORICAL = CATEGORICAL_FEATURES


def explain(applicant: Applicant, probability: float) -> list[dict[str, str]]:
    median = DATA[FEATURES].median(numeric_only=True)
    drivers = []
    if applicant.debt_to_income > median["debt_to_income"]:
        drivers.append(("debt_to_income", "increases_risk", "Debt-to-income ratio is above the training-set median."))
    if applicant.revolving_utilization > median["revolving_utilization"]:
        drivers.append(("revolving_utilization", "increases_risk", "Revolving utilization is elevated versus the sample."))
    if applicant.annual_income > median["annual_income"]:
        drivers.append(("annual_income", "decreases_risk", "Annual income is above the training-set median."))
    if applicant.delinquencies_2y > 0:
        drivers.append(("delinquencies_2y", "increases_risk", "Recent delinquencies are associated with higher observed default risk."))
    return [{"feature": f, "direction": d, "explanation": e} for f, d, e in drivers[:5]] or [
        {
            "feature": "profile",
            "direction": "neutral",
            "explanation": "No single feature is far from the demo training distribution.",
        }
    ]


def warnings_for(applicant: Applicant) -> list[str]:
    if applicant.debt_to_income > 0.75 or applicant.revolving_utilization > 1.0:
        return ["Applicant is near or outside the high-risk range of the demo training data."]
    return []


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> dict[str, Any]:
    row = pd.DataFrame([applicant.model_dump()])[FEATURES]
    probability = float(MODEL.predict_proba(row)[0, 1])
    band, grade = decision_band(probability)
    return {
        "default_probability": probability,
        "decision_band": band,
        "risk_grade": grade,
        "top_risk_drivers": explain(applicant, probability),
        "model_version": METRICS["model_version"],
        "warnings": warnings_for(applicant),
    }


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(payload: BatchScoreRequest) -> list[dict[str, Any]]:
    return [score(applicant) for applicant in payload.applicants]


@app.post("/policy/simulate", response_model=PolicyResponse)
def simulate(policy: PolicyRequest) -> dict[str, Any]:
    return simulate_policy(ARTIFACT, policy)


@app.get("/model/metrics", response_model=MetricsResponse)
def metrics() -> dict[str, Any]:
    return {
        "model_version": ARTIFACT.metrics["model_version"],
        "selected_model": ARTIFACT.metrics["selected_model"],
        "target_definition": ARTIFACT.metrics["target_definition"],
        "models": ARTIFACT.metrics["models"],
        "selected_model_diagnostics": ARTIFACT.metrics["selected_model_diagnostics"],
        "policy_reference": ARTIFACT.metrics["policy_reference"],
    }


@app.get("/model/explainability")
def model_explainability() -> dict[str, Any]:
    return {
        "method": "Permutation importance for global drivers and one-feature-at-a-time counterfactuals for local drivers.",
        "caution": "Drivers describe model behavior on synthetic demo data and should not be interpreted as causal credit findings.",
        "global_feature_importance": ARTIFACT.feature_importance,
        "reference_profile": ARTIFACT.reference_values,
    }


@app.get("/model/card")
def model_card() -> dict[str, Any]:
    return {
        "model_name": "Credit Risk Default Model",
        "model_version": ARTIFACT.metrics["model_version"],
        "intended_use": "Portfolio demo for estimating default probability, explaining model drivers, and simulating credit policy thresholds.",
        "not_intended_for": "Real lending decisions or legal compliance.",
        "dataset_source": ARTIFACT.metrics["data"]["dataset_source"],
        "features": FEATURES,
        "data_dictionary": DATA_DICTIONARY,
        "excluded_leakage_features": LEAKAGE_EXCLUDED_FEATURES,
        "metrics": metrics(),
        "explainability": {
            "global": "Permutation importance on the held-out demo set.",
            "local": "Counterfactual replacement of each applicant feature with a training reference value.",
        },
        "fairness_considerations": "Protected characteristics are not included; production use would require formal fairness and compliance review.",
        "limitations": [
            "Synthetic data; no claim of production validity.",
            "Probability calibration should be rechecked on real held-out data.",
            "Feature drivers describe model sensitivity, not causal effects.",
        ],
    }


@app.get("/demo/applicants")
def demo_applicants() -> dict[str, list[dict[str, Any]]]:
    return {
        "applicants": [
            {
                "loan_amount": 9000,
                "term_months": 36,
                "interest_rate": 0.091,
                "annual_income": 118000,
                "debt_to_income": 0.12,
                "employment_length_years": 8,
                "home_ownership": "mortgage",
                "loan_purpose": "home_improvement",
                "revolving_utilization": 0.21,
                "open_accounts": 9,
                "delinquencies_2y": 0,
            },
            {
                "loan_amount": 22000,
                "term_months": 60,
                "interest_rate": 0.184,
                "annual_income": 68000,
                "debt_to_income": 0.41,
                "employment_length_years": 2,
                "home_ownership": "rent",
                "loan_purpose": "debt_consolidation",
                "revolving_utilization": 0.76,
                "open_accounts": 13,
                "delinquencies_2y": 1,
            },
            {
                "loan_amount": 35000,
                "term_months": 60,
                "interest_rate": 0.247,
                "annual_income": 52000,
                "debt_to_income": 0.57,
                "employment_length_years": 0,
                "home_ownership": "rent",
                "loan_purpose": "small_business",
                "revolving_utilization": 0.94,
                "open_accounts": 17,
                "delinquencies_2y": 3,
            },
        ]
    }
