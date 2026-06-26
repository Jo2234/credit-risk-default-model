from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURES = [
    "loan_amount",
    "term_months",
    "interest_rate",
    "annual_income",
    "debt_to_income",
    "employment_length_years",
    "home_ownership",
    "loan_purpose",
    "revolving_utilization",
    "open_accounts",
    "delinquencies_2y",
]

NUMERIC_FEATURES = [
    "loan_amount",
    "term_months",
    "interest_rate",
    "annual_income",
    "debt_to_income",
    "employment_length_years",
    "revolving_utilization",
    "open_accounts",
    "delinquencies_2y",
]

CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]

LEAKAGE_EXCLUDED_FEATURES = [
    "loan_status",
    "recoveries",
    "collection_recovery_fee",
    "last_payment_date",
    "total_payment_received",
    "total_payment_received_to_date",
    "days_past_due",
    "hardship_status",
]

DATA_DICTIONARY: dict[str, dict[str, str]] = {
    "loan_amount": {"timing": "application", "description": "Requested loan principal."},
    "term_months": {"timing": "origination", "description": "Contract term in months."},
    "interest_rate": {"timing": "origination", "description": "Quoted annual interest rate."},
    "annual_income": {"timing": "application", "description": "Self-reported annual income."},
    "debt_to_income": {"timing": "application", "description": "Debt obligations divided by income."},
    "employment_length_years": {"timing": "application", "description": "Years at current employment."},
    "home_ownership": {"timing": "application", "description": "Rent, mortgage, own, or other."},
    "loan_purpose": {"timing": "application", "description": "Borrower-provided loan purpose."},
    "revolving_utilization": {"timing": "application", "description": "Revolving credit utilization ratio."},
    "open_accounts": {"timing": "application", "description": "Number of open credit accounts."},
    "delinquencies_2y": {"timing": "application", "description": "Credit delinquencies in the past two years."},
}


def generate_synthetic_loan_data(n_rows: int = 2400, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic LendingClub-like sample for local demos and tests."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(11.05, 0.48, n_rows).clip(22_000, 320_000)
    loan_amount = rng.gamma(4.0, 4500, n_rows).clip(1_000, 50_000)
    dti_base = rng.beta(2.2, 5.3, n_rows)
    loan_to_income = (loan_amount / income).clip(0, 1.5)
    debt_to_income = (dti_base + 0.18 * loan_to_income).clip(0.01, 0.95)
    delinquencies = rng.poisson(0.32 + 0.9 * (debt_to_income > 0.42), n_rows).clip(0, 8)
    revolving_utilization = (rng.beta(2.4, 4.0, n_rows) + 0.18 * (debt_to_income > 0.38)).clip(0.02, 1.25)
    term_months = rng.choice([36, 60], n_rows, p=[0.7, 0.3])
    home_ownership = rng.choice(["rent", "mortgage", "own", "other"], n_rows, p=[0.41, 0.45, 0.11, 0.03])
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical"],
        n_rows,
        p=[0.49, 0.2, 0.1, 0.08, 0.07, 0.06],
    )
    employment = rng.integers(0, 13, n_rows)
    open_accounts = rng.poisson(8.5 + 3.5 * revolving_utilization, n_rows).clip(1, 35)

    risk_premium = (
        0.035
        + 0.13 * debt_to_income
        + 0.045 * revolving_utilization
        + 0.011 * delinquencies
        + 0.018 * (term_months == 60)
        + 0.018 * np.isin(loan_purpose, ["small_business", "medical"])
        + 0.01 * (home_ownership == "rent")
    )
    interest_rate = (risk_premium + rng.normal(0.055, 0.018, n_rows)).clip(0.045, 0.34)

    logit = (
        -3.65
        + 4.7 * debt_to_income
        + 2.65 * revolving_utilization
        + 4.1 * interest_rate
        + 0.2 * delinquencies
        + 0.42 * (term_months == 60)
        + 0.55 * (loan_purpose == "small_business")
        + 0.24 * (loan_purpose == "medical")
        + 0.24 * (home_ownership == "rent")
        - 0.36 * np.log(income / 55_000)
        - 0.035 * employment
        + 0.38 * loan_to_income
        + rng.normal(0, 0.42, n_rows)
    )
    default_probability = 1 / (1 + np.exp(-logit))
    default = (rng.random(n_rows) < default_probability).astype(int)

    return pd.DataFrame(
        {
            "loan_amount": loan_amount.round(2),
            "term_months": term_months,
            "interest_rate": interest_rate.round(4),
            "annual_income": income.round(2),
            "debt_to_income": debt_to_income.round(4),
            "employment_length_years": employment,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "revolving_utilization": revolving_utilization.round(4),
            "open_accounts": open_accounts,
            "delinquencies_2y": delinquencies,
            "default": default,
        }
    )


def reference_profile(frame: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        profile[feature] = float(frame[feature].median())
    for feature in CATEGORICAL_FEATURES:
        profile[feature] = str(frame[feature].mode().iloc[0])
    return profile


def training_ranges(frame: pd.DataFrame) -> dict[str, Any]:
    ranges: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        ranges[feature] = {
            "p01": float(frame[feature].quantile(0.01)),
            "p50": float(frame[feature].quantile(0.5)),
            "p99": float(frame[feature].quantile(0.99)),
        }
    for feature in CATEGORICAL_FEATURES:
        ranges[feature] = sorted(str(value) for value in frame[feature].dropna().unique())
    return ranges

