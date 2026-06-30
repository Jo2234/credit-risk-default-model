# Model Card: Credit Risk Default Model

## Summary

This model estimates the probability that a loan applicant will default. It is a portfolio demonstration of a credit-risk workflow: scoring, explanation, threshold policy, model metrics, leakage controls, and model-risk documentation.

The current implementation trains at application startup on synthetic LendingClub-like data generated locally in `backend/main.py`. It is not trained on real borrower records and must not be used for lending, underwriting, pricing, adverse-action notices, or legal compliance decisions.

## Model Details

| Field | Value |
| --- | --- |
| Model name | Credit Risk Default Model |
| Current version | Exposed at runtime as `credit-risk-<selected_model>-v1` |
| Model family | Logistic regression baseline and random forest improved model |
| Serving interface | FastAPI |
| Main scoring route | `POST /score` |
| Policy route | `POST /policy/simulate` |
| Metrics route | `GET /model/metrics` |
| Documentation route | `GET /model/card` |
| Training data | Synthetic LendingClub-like records generated locally |
| Artifact metadata | Persisted at `artifacts/model_metadata.json` and exposed in metrics/model-card responses |
| Intended audience | Credit analytics, fintech, banking, and applied ML portfolio reviewers |

## Intended Use

This model is intended to demonstrate how a lender-facing risk analytics system could:

- Estimate default probability for an applicant.
- Explain the strongest risk drivers in plain English.
- Assign a decision band: approve, manual review, or reject.
- Simulate how approval and rejection thresholds affect a portfolio.
- Compare model performance with more than one metric.
- Document model risk clearly instead of hiding it behind a dashboard.

## Out-of-Scope Uses

This model is not intended for:

- Real lending decisions.
- Automated underwriting.
- Credit pricing.
- Collections strategy.
- Adverse-action notices.
- Legal or regulatory compliance.
- Consumer-facing financial advice.
- Any workflow where an individual is materially affected by the model output.

## Data

### Current Demo Data

The application currently generates synthetic data with plausible credit-risk fields:

- `loan_amount`
- `term_months`
- `interest_rate`
- `annual_income`
- `debt_to_income`
- `employment_length_years`
- `home_ownership`
- `loan_purpose`
- `revolving_utilization`
- `open_accounts`
- `delinquencies_2y`

The target variable is generated from a synthetic risk function that increases default likelihood for higher debt-to-income, higher revolving utilization, recent delinquencies, higher interest rates, longer term loans, and lower income.

### Intended Real Dataset Extension

For a real public-data version, the recommended source is a legally usable historical loan dataset such as LendingClub or a smaller public credit-default dataset.

The target should be defined as:

- `default = 1` for default, charged-off, or equivalent terminal outcomes.
- `default = 0` for fully paid or equivalent terminal outcomes.

Rows should be excluded when:

- The loan is current and the final outcome is unknown.
- The status is ambiguous.
- The available fields cannot be cleanly mapped to a repayment/default outcome.

## Feature Timing and Leakage Controls

Credit-risk modeling is especially vulnerable to data leakage. A real implementation must classify every candidate feature by when it becomes known:

| Feature timing | Treatment |
| --- | --- |
| Known at application time | Candidate feature |
| Known at loan origination | Candidate feature if policy allows |
| Known after origination | Usually excluded |
| Known after default or repayment | Excluded |

The current project explicitly excludes these leakage-prone fields:

- `loan_status`
- `recoveries`
- `collection_recovery_fee`
- `last_payment_date`
- `total_payment_received`

Additional real-dataset fields would need the same review before training.

## Training Procedure

The current training flow:

1. Generate a synthetic loan dataset.
2. Split into train and test sets with stratification.
3. Preprocess numeric and categorical features using a scikit-learn pipeline.
4. Train a logistic regression baseline.
5. Train a random forest improved model.
6. Score both models on the held-out test set.
7. Select the model with the strongest ROC-AUC.
8. Serve the selected model through the API.

The preprocessing pipeline standardizes numeric fields and one-hot encodes categorical fields. The same fitted pipeline is used for scoring.

## Reproducibility and Artifact Metadata

Each deterministic demo training run writes a metadata sidecar to `artifacts/model_metadata.json`. The sidecar records:

- Model version and selected model family.
- Training and holdout row counts.
- Synthetic data generator and train/test split parameters.
- Feature schema hash and leakage-exclusion list hash inputs.
- SHA-256 hashes for the training split, holdout split, and metadata document.
- Library versions used to create the artifact.

The persisted sidecar describes the in-memory sklearn pipeline; it is not a production registry, not a serialized model binary, and not evidence of real-world model validity. It is included so reviewers can tie reported synthetic metrics to a reproducible training hash and feature schema.

## Evaluation

The API reports metrics for each trained candidate model:

- ROC-AUC
- PR-AUC
- Brier score
- Precision at the 20% default-probability threshold
- Recall at the 20% default-probability threshold
- F1 at the 20% default-probability threshold

Accuracy is intentionally not the main metric because default events are often imbalanced. ROC-AUC and PR-AUC describe ranking quality, while the Brier score helps evaluate probability quality.

## Calibration

The model returns probabilities, so calibration matters. In a real deployment, the project should add:

- Calibration curves.
- Out-of-time calibration checks.
- Brier score tracking by model version.
- Platt scaling or isotonic regression if probabilities are poorly calibrated.

The current synthetic demo exposes Brier score and an 8-bin calibration curve for the selected model. The random-forest candidate is calibrated with isotonic calibration before selection. These diagnostics support software review only and are not evidence of real-world calibration.

## Decision Policy

The demo uses a three-band policy:

| Default probability | Decision |
| --- | --- |
| Below 8% | Approve |
| 8% to 20% | Manual review |
| Above 20% | Reject |

These thresholds are illustrative. They are not a recommendation for any lender. Different businesses would choose thresholds based on risk appetite, funding costs, growth goals, regulatory requirements, and review capacity.

## Policy Simulation

The `POST /policy/simulate` endpoint estimates:

- Approval rate.
- Manual-review rate.
- Rejection rate.
- Expected default rate among approved borrowers.
- False approvals.
- False rejections.
- Estimated profit.

The profit estimate uses simplified assumptions:

- Average loan amount from the test sample.
- 8% assumed interest revenue on true approved loans.
- User-provided loss given default.
- User-provided manual-review cost.

This is a teaching device for policy tradeoffs, not a production finance model.

The selected model diagnostics also include a deterministic threshold stress table over several approve/reject threshold pairs. Tests assert that approval rates increase and rejection rates decrease as the illustrative thresholds loosen.

## Explainability

The scoring endpoint returns local explanations for the applicant. Current explanations are rule-based comparisons against the synthetic training distribution, for example:

- Debt-to-income above the sample median increases risk.
- Revolving utilization above the sample median increases risk.
- Annual income above the sample median decreases risk.
- Recent delinquencies increase risk.

For a real model version, local and global explainability should be upgraded to SHAP or another validated method, with care not to present feature importance as causal proof.

## Fairness and Protected Characteristics

The demo does not include protected characteristics such as race, ethnicity, religion, sex, marital status, age beyond legally permissible treatment, disability, or national origin.

Excluding protected characteristics is not enough to prove fairness. The demo exposes a non-protected proxy-group review by `home_ownership` and `loan_purpose` with counts, mean predicted default probabilities, and approval rates at the illustrative 8% threshold. This is a deterministic transparency check, not a legal fairness conclusion.

A real credit model would require:

- Protected-class or proxy-group analysis where legally and ethically appropriate.
- Disparate impact review.
- Error-rate comparison across groups.
- Adverse-action reason review.
- Compliance review for the relevant lending jurisdiction.

## Assumptions

- Synthetic data is sufficient for demonstrating software behavior.
- The applicant input fields are available before the lending decision.
- The target represents terminal repayment/default outcome.
- The model output is a probability estimate, not a binding decision.
- The manual-review band exists to keep uncertain cases out of fully automated treatment.
- Business economics in the simulator are simplified and configurable.

## Limitations

- Synthetic training data means model performance is not evidence of real-world accuracy.
- No external dataset version is currently tracked.
- No experiment tracking or model registry is included.
- A deterministic model metadata sidecar is persisted, but no full serialized model binary or production model registry is included.
- No out-of-time validation is included.
- No production drift monitoring is included.
- No formal SHAP explainability layer is implemented yet.
- No legal compliance workflow is implemented.
- No adverse-action notice generation is implemented.
- No authentication, audit log, or access control is implemented.

## Model Risk

Main model risks:

- Leakage risk: real datasets may contain post-origination fields that make backtests look unrealistically strong.
- Calibration risk: good ranking does not guarantee reliable probabilities.
- Dataset shift: borrower behavior, underwriting standards, macro conditions, and loan products can change.
- Fairness risk: neutral-looking variables may proxy protected characteristics.
- Policy risk: thresholds can optimize short-term profit while creating unacceptable customer, legal, or reputational harm.
- Automation risk: users may treat a demo score as a decision instead of a decision-support signal.

Required controls before real use:

- Versioned raw data and immutable processed datasets.
- Feature timing review and rejected-feature register.
- Train, validation, test, and out-of-time splits.
- Calibration report.
- Fairness report.
- Drift monitoring.
- Human review procedure.
- Audit trail for scores and policy versions.
- Legal and compliance sign-off.

## Ethical Considerations

Credit decisions affect access to capital. Any real deployment should be conservative, reviewable, and contestable. The model should support human judgment, not hide policy decisions inside a probability score.

Applicants should not be materially affected by this demo model. The current implementation is for education and portfolio review only.

## Maintenance Plan

If extended beyond the synthetic demo, each model release should record:

- Dataset source and version.
- Target definition.
- Feature list and rejected-feature list.
- Training date.
- Model version.
- Validation metrics.
- Calibration metrics.
- Fairness checks.
- Decision thresholds.
- Known limitations.
- Owner and review date.

## Recommended Next Improvements

- Replace synthetic data with a documented public dataset.
- Add a data dictionary.
- Save model artifacts and preprocessing artifacts.
- Add SHAP-based local and global explanations.
- Add calibration plots and threshold curves.
- Add batch scoring.
- Add drift and fairness reports.
- Add MLflow or another experiment tracker.
