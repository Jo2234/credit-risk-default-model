# Credit Risk Default Model

A small credit-risk analytics demo that estimates borrower default probability, explains the main risk drivers, and shows how different approval and rejection thresholds change portfolio outcomes.

This is built as a portfolio project for banking, fintech, credit analytics, and applied ML roles. The point is not to pretend this is a production lending system. The point is to show the full shape of a practical risk workflow: probability scoring, leakage-aware feature design, model evaluation, explainable decisions, and policy tradeoffs.

## What It Does

- Scores a borrower through a FastAPI endpoint.
- Returns a default probability, decision band, risk grade, explanation, model version, and warnings.
- Simulates a credit policy with approve, manual-review, and reject bands.
- Reports model metrics for baseline and improved models.
- Exposes a model-card endpoint and a human-readable model card.
- Serves a lightweight browser demo for scoring and policy simulation.

## Current Demo Scope

The current implementation uses synthetic LendingClub-like data generated locally at app startup. It is intentionally small enough to run without downloading a large dataset and is suitable for demos, interviews, and code review.

The intended production-style extension is to replace the synthetic generator with a public loan dataset, such as LendingClub historical loans, while preserving the same controls:

- Define `default = 1` for default, charge-off, or equivalent terminal outcomes.
- Define `default = 0` for fully paid terminal outcomes.
- Exclude current loans and ambiguous outcomes.
- Exclude leakage fields that would not be known at application or origination time.
- Re-check model performance, calibration, fairness, and drift on held-out real data.

## Architecture

```text
frontend/index.html
  Static browser demo

backend/main.py
  FastAPI app
  Synthetic data generator
  Feature preprocessing pipeline
  Logistic regression baseline
  Random forest improved model
  Borrower scoring endpoint
  Policy simulation endpoint
  Metrics and model-card endpoints

tests/test_backend.py
  API behavior checks
```

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API:

```bash
cd projects/credit-risk-default-model
PYTHONPATH=. uvicorn backend.main:app --reload --port 8003
```

In a second terminal, serve the browser demo:

```bash
cd projects/credit-risk-default-model
python3 -m http.server 5175 -d frontend
```

Open:

```text
http://localhost:5175
```

The frontend expects the API at:

```text
http://localhost:8003
```

## API Workflows

Health check:

```bash
curl http://localhost:8003/health
```

Score a demo applicant:

```bash
curl -X POST http://localhost:8003/score \
  -H "Content-Type: application/json" \
  -d '{
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
    "delinquencies_2y": 0
  }'
```

Simulate a credit policy:

```bash
curl -X POST http://localhost:8003/policy/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "approve_below": 0.08,
    "reject_above": 0.20,
    "loss_given_default": 0.60,
    "manual_review_cost": 35
  }'
```

Inspect metrics and model documentation:

```bash
curl http://localhost:8003/model/metrics
curl http://localhost:8003/model/card
```

## Decision Policy

The demo converts model probability into an operational decision:

| Default probability | Decision | Risk grade |
| --- | --- | --- |
| Below 8% | Approve | Low |
| 8% to 20% | Manual review | Medium |
| Above 20% | Reject | High |

These thresholds are not recommendations. They are demo assumptions used to show the tradeoff between approval rate, expected default rate, false approvals, false rejections, manual-review cost, and estimated profit.

## Modeling Assumptions

- The model predicts probability of default, not borrower character or legal creditworthiness.
- Input features are treated as application-time or origination-time fields.
- Protected characteristics are not included.
- The synthetic target is generated from plausible credit-risk relationships, including debt-to-income, revolving utilization, delinquency count, interest rate, term, and income.
- The selected model is chosen by ROC-AUC among the trained candidates.
- Probability quality must be judged with calibration metrics, not ranking metrics alone.

## Leakage Controls

The project explicitly excludes fields that would leak future loan performance into the prediction:

- `loan_status`
- `recoveries`
- `collection_recovery_fee`
- `last_payment_date`
- `total_payment_received`

For a real LendingClub-style dataset, every candidate feature should be categorized as known at application time, known at origination, known after origination, or excluded due to leakage risk.

## Limitations

- The current model is trained on synthetic data, so the metrics are useful for software validation but not for real-world credit validation.
- The explanation logic is a lightweight demo based on feature comparisons, not a full SHAP implementation.
- The policy simulator uses simplified economics: average loan amount, assumed interest revenue, loss given default, and manual-review cost.
- The API does not persist scoring history, model artifacts, dataset versions, or experiment runs.
- The project is not compliant with lending regulations and must not be used for real credit decisions.

## Model Risk

Credit models can create real financial and legal harm if used casually. Before any real use, this project would need:

- A legally usable, versioned dataset.
- Formal target definition and rejected-feature register.
- Out-of-time validation.
- Probability calibration review.
- Fairness testing using appropriate protected or proxy group analysis.
- Monitoring for data drift, performance decay, and policy drift.
- Human review procedures for adverse-action, appeals, and exceptions.
- Legal and compliance review for the relevant jurisdiction.

## Tests

Run:

```bash
cd projects/credit-risk-default-model
PYTHONPATH=. pytest
```

The current tests cover:

- Score endpoint probability bounds.
- Policy threshold validation.
- Model-card leakage documentation.

## Portfolio Narrative

This project is designed to read as a risk analytics product, not a generic notebook. It connects machine learning to an actual credit-policy workflow: estimate risk, explain the estimate, choose thresholds, and understand the business cost of mistakes.

See [model_card.md](model_card.md) for the full model documentation.
