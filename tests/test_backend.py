from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import backend.main as backend


def test_feature_schema_excludes_post_origination_leakage_fields():
    leakage_fields = {
        "loan_status",
        "recoveries",
        "collection_recovery_fee",
        "last_payment_date",
        "total_payment_received",
    }

    assert set(backend.NUMERIC).isdisjoint(backend.CATEGORICAL)
    assert set(backend.FEATURES) == set(backend.NUMERIC + backend.CATEGORICAL)
    assert leakage_fields.isdisjoint(backend.FEATURES)


def test_preprocessing_pipeline_encodes_unknown_categories(sample_applicant):
    training_rows = pd.DataFrame(
        [
            sample_applicant,
            {**sample_applicant, "home_ownership": "mortgage", "loan_purpose": "credit_card", "debt_to_income": 0.61},
            {**sample_applicant, "home_ownership": "own", "loan_purpose": "home_improvement", "interest_rate": 0.08},
            {**sample_applicant, "loan_purpose": "small_business", "revolving_utilization": 0.91},
        ]
    )
    target = np.array([0, 1, 0, 1])
    pipeline = backend.build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(training_rows[backend.FEATURES], target)

    unknown_category_row = pd.DataFrame(
        [
            {
                **sample_applicant,
                "home_ownership": "family",
                "loan_purpose": "medical",
            }
        ]
    )
    probabilities = pipeline.predict_proba(unknown_category_row[backend.FEATURES])
    transformed = pipeline.named_steps["preprocess"].transform(unknown_category_row[backend.FEATURES])
    transformed_values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0, 1] <= 1
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > len(backend.NUMERIC)
    assert np.isfinite(transformed_values).all()


@pytest.mark.parametrize(
    ("probability", "expected_band", "expected_grade"),
    [
        (0.079, "approve", "low"),
        (0.080, "manual_review", "medium"),
        (0.200, "manual_review", "medium"),
        (0.201, "reject", "high"),
    ],
)
def test_score_threshold_boundaries(client, fixed_model, sample_applicant, probability, expected_band, expected_grade):
    fixed_model(probability)

    response = client.post("/score", json=sample_applicant)

    assert response.status_code == 200
    body = response.json()
    assert body["default_probability"] == pytest.approx(probability)
    assert body["decision_band"] == expected_band
    assert body["risk_grade"] == expected_grade


def test_score_real_model_returns_probability_and_contract_fields(client, sample_applicant):
    response = client.post("/score", json=sample_applicant)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["decision_band"] in {"approve", "manual_review", "reject"}
    assert body["risk_grade"] in {"low", "medium", "high"}
    assert body["model_version"] == backend.METRICS["model_version"]
    assert isinstance(body["warnings"], list)
    assert body["top_risk_drivers"]
    for driver in body["top_risk_drivers"]:
        assert set(driver) == {"feature", "direction", "explanation"}
        assert driver["direction"] in {"increases_risk", "decreases_risk", "neutral"}
        assert isinstance(driver["explanation"], str) and driver["explanation"]


def test_explainability_reports_directional_plain_english_drivers(client, fixed_model, high_risk_applicant):
    fixed_model(0.44)

    response = client.post("/score", json=high_risk_applicant)

    assert response.status_code == 200
    body = response.json()
    drivers = body["top_risk_drivers"]
    driver_by_feature = {driver["feature"]: driver for driver in drivers}
    assert driver_by_feature["debt_to_income"]["direction"] == "increases_risk"
    assert driver_by_feature["revolving_utilization"]["direction"] == "increases_risk"
    assert driver_by_feature["annual_income"]["direction"] == "decreases_risk"
    assert driver_by_feature["delinquencies_2y"]["direction"] == "increases_risk"
    assert all(driver["explanation"].endswith(".") for driver in drivers)
    assert body["warnings"] == ["Applicant is near or outside the high-risk range of the demo training data."]


def test_explainability_falls_back_to_neutral_profile_for_typical_applicant(sample_applicant):
    medians = backend.DATA[backend.FEATURES].median(numeric_only=True)
    typical = backend.Applicant(
        **{
            **sample_applicant,
            "annual_income": float(medians["annual_income"]) - 1,
            "debt_to_income": float(medians["debt_to_income"]) * 0.8,
            "revolving_utilization": float(medians["revolving_utilization"]) * 0.8,
            "delinquencies_2y": 0,
        }
    )

    assert backend.explain(typical, probability=0.12) == [
        {
            "feature": "profile",
            "direction": "neutral",
            "explanation": "No single feature is far from the demo training distribution.",
        }
    ]


def test_policy_simulation_rates_sum_to_one_and_counts_match_holdout(client):
    response = client.post(
        "/policy/simulate",
        json={"approve_below": 0.08, "reject_above": 0.20, "loss_given_default": 0.6, "manual_review_cost": 35},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["approval_rate"] + body["manual_review_rate"] + body["rejection_rate"] == pytest.approx(1.0)
    assert 0 <= body["expected_default_rate_approved"] <= 1
    assert isinstance(body["estimated_profit"], float)
    assert sum(body["confusion_matrix"].values()) <= len(backend.Y_TEST)
    assert set(body["confusion_matrix"]) == {"true_approved", "false_approved", "true_rejected", "false_rejected"}


@pytest.mark.parametrize(
    "payload",
    [
        {"approve_below": 0.2, "reject_above": 0.2},
        {"approve_below": 0.3, "reject_above": 0.2},
    ],
)
def test_policy_threshold_validation(client, payload):
    response = client.post("/policy/simulate", json=payload)

    assert response.status_code == 422
    assert "approve_below must be less than reject_above" in response.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("loan_amount", 0),
        ("term_months", 6),
        ("interest_rate", 1.2),
        ("annual_income", -1),
        ("debt_to_income", 2.1),
        ("revolving_utilization", 1.6),
        ("open_accounts", -1),
        ("delinquencies_2y", -1),
    ],
)
def test_score_rejects_invalid_applicant_inputs(client, sample_applicant, field, value):
    response = client.post("/score", json={**sample_applicant, field: value})

    assert response.status_code == 422
    assert field in response.text


def test_model_metrics_and_card_document_business_contract(client):
    metrics_response = client.get("/model/metrics")
    card_response = client.get("/model/card")

    assert metrics_response.status_code == 200
    assert card_response.status_code == 200
    metrics = metrics_response.json()
    card = card_response.json()
    selected = metrics["selected_model"]
    selected_metrics = metrics["models"][selected]
    assert metrics["model_version"].startswith(f"credit-risk-{selected}-v1")
    assert metrics["target_definition"] == "1 = default or charge-off equivalent; 0 = fully paid equivalent"
    assert {"roc_auc", "pr_auc", "brier_score", "precision_at_20pct", "recall_at_20pct", "f1_at_20pct"} <= set(selected_metrics)
    assert card["features"] == backend.FEATURES
    assert "loan_status" in card["excluded_leakage_features"]
    assert "Real lending decisions or legal compliance." == card["not_intended_for"]


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
