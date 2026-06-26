from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app  # noqa: E402


class FixedProbabilityModel:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, rows):
        return np.tile([[1 - self.probability, self.probability]], (len(rows), 1))


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture
def sample_applicant():
    return {
        "loan_amount": 15000,
        "term_months": 36,
        "interest_rate": 0.132,
        "annual_income": 85000,
        "debt_to_income": 0.24,
        "employment_length_years": 4,
        "home_ownership": "rent",
        "loan_purpose": "debt_consolidation",
        "revolving_utilization": 0.41,
        "open_accounts": 8,
        "delinquencies_2y": 0,
    }


@pytest.fixture
def high_risk_applicant(sample_applicant):
    applicant = sample_applicant.copy()
    applicant.update(
        {
            "interest_rate": 0.29,
            "annual_income": 180000,
            "debt_to_income": 0.92,
            "employment_length_years": 0,
            "home_ownership": "rent",
            "revolving_utilization": 1.18,
            "delinquencies_2y": 3,
        }
    )
    return applicant


@pytest.fixture
def fixed_model(monkeypatch):
    import backend.main as backend

    def apply(probability: float):
        model = FixedProbabilityModel(probability)
        monkeypatch.setattr(backend, "MODEL", model)
        return model

    return apply
