# Risk Policy

## Credit Score Brackets

| Credit Score | Risk Level | Action        | Rationale               |
|-------------|------------|---------------|-------------------------|
| Poor        | High Risk  | Reject        | High probability of default |
| Standard    | Medium Risk| Manual Review | Human judgment required |
| Good        | Low Risk   | Approve       | Low default risk        |

## Low-Confidence Override
Any prediction with max class probability < **0.60** is overridden to **manual_review** regardless of predicted class.

## Model Governance
- Model must achieve macro F1 >= 0.70 on validation gate before deployment
- Retraining triggered when: PSI > 0.25 (drift) or F1 drops below 0.65
- All predictions logged for audit trail

## Monitoring Thresholds
| Metric               | Warning  | Alert    |
|---------------------|----------|----------|
| PSI max             | > 0.10   | > 0.20   |
| Model F1 macro      | < 0.70   | < 0.65   |
| API p95 latency     | > 300ms  | > 500ms  |
| Missing rate        | > 20%    | > 30%    |
