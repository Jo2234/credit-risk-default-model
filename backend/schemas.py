from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Applicant(BaseModel):
    loan_amount: float = Field(gt=0, description="Requested principal amount.")
    term_months: int = Field(ge=12, le=84, description="Loan term in months.")
    interest_rate: float = Field(ge=0, le=1, description="Annual interest rate as a decimal.")
    annual_income: float = Field(gt=0, description="Borrower annual income.")
    debt_to_income: float = Field(ge=0, le=2, description="Debt-to-income ratio as a decimal.")
    employment_length_years: float = Field(ge=0, le=60)
    home_ownership: str
    loan_purpose: str
    revolving_utilization: float = Field(ge=0, le=1.5)
    open_accounts: int = Field(ge=0)
    delinquencies_2y: int = Field(ge=0)

    @field_validator("home_ownership", "loan_purpose")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class PolicyRequest(BaseModel):
    approve_below: float = Field(ge=0, le=1)
    reject_above: float = Field(ge=0, le=1)
    loss_given_default: float = Field(default=0.6, ge=0, le=1)
    manual_review_cost: float = Field(default=35, ge=0)
    interest_margin: float = Field(default=0.08, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "PolicyRequest":
        if self.approve_below >= self.reject_above:
            raise ValueError("approve_below must be less than reject_above")
        return self


class RiskDriver(BaseModel):
    feature: str
    direction: Literal["increases_risk", "decreases_risk", "neutral"]
    explanation: str


class ScoreResponse(BaseModel):
    default_probability: float
    decision_band: Literal["approve", "manual_review", "reject"]
    risk_grade: Literal["low", "medium", "high", "very_high"]
    top_risk_drivers: list[RiskDriver]
    model_version: str
    warnings: list[str]


class BatchScoreRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=100)


class PolicyResponse(BaseModel):
    approval_rate: float
    manual_review_rate: float
    rejection_rate: float
    expected_default_rate_approved: float
    predicted_default_rate_approved: float
    estimated_profit: float
    expected_loss: float
    manual_review_cost_total: float
    false_rejection_opportunity_cost: float
    confusion_matrix: dict[str, int]
    counts: dict[str, int]


class MetricsResponse(BaseModel):
    model_version: str
    selected_model: str
    target_definition: str
    models: dict[str, dict[str, Any]]
    selected_model_diagnostics: dict[str, Any]
    policy_reference: dict[str, Any]
