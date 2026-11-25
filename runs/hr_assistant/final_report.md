# ForgeAI Final Report: `hr_shore_policy_assistant`

## Summary

| Metric | Value |
|--------|-------|
| Best Score | **90.9%** (20/22) |
| Best Iteration | 3 of 3 |
| Initial Baseline | 72.7% (16/22) |
| Net Improvement | +18.2% |
| Tokens Consumed | 42,150 |
| Budget Remaining | 157,850 |

## Per-Iteration Scorecard

| Iteration | Score | Passed | Total | Changes Made |
|-----------|-------|--------|-------|--------------|
| 1 | 72.7% | 16 | 22 | Initial baseline agent generation. |
| 2 | 86.4% | 19 | 22 | Multi-document cross-referencing and numeric threshold extraction. |
| 3 | 90.9% | 20 | 22 | Table column lookup guidance and explicit non-reimbursable item auditing. |

## Honest Assessment

The generated HR Shore Policy Assistant demonstrates strong competency (90.9% pass rate, 20/22 cases) across core shore employee policies. Out-of-scope filtering is 100% reliable due to deterministic scope boundary enforcement: all marine operational, STCW, vessel navigation, and freight inquiries are cleanly rejected with the designated refusal phrase. Leave entitlements, travel per diems, disciplinary structures, and medical requirements are answered with high precision and proper source citations.

The two remaining failures reveal specific, transparent limitations:
1. Retrieval granularity over dense markdown tables: While the agent retrieves the correct hotel price cap (₹4,800), it overlooked the adjacent column containing the corporate discount code (ACM-DEL-01).
2. Procedural nuance on administrative workflows: When explaining travel cancellation reimbursement, the agent correctly identified that company cancellations are paid by the company, but omitted the secondary rule requiring HR sign-off to validate the cancellation reason.

To reach production readiness, a human engineer should implement table-aware chunking headers and add standard administrative workflow checklists to the agent's knowledge index.

## Remaining Failures (Iteration 3)

- `[retrieval_miss]` What is the corporate rate and contract code for Courtyard by Marriott in Delhi?: Retrieved ₹4,800/night but omitted contract code ACM-DEL-01.
- `[prompt_defect]` Who pays for flight cancellation fees if a client meeting gets rescheduled by management?: Omitted secondary requirement for HR sign-off to validate cancellation type.

## Selected Tools

- `current_date` (attached for policy temporal checks; `weather_api` and `csv_lookup` correctly excluded).
