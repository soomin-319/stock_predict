# Model Training Improvements Design

## Goal
Improve `docs/04_model.md` P1/P2 items with low-risk LightGBM early-stopping/regularization support and explicit sklearn fallback warnings.

## Scope
- Add optional training config fields for LightGBM regularization and early stopping.
- Keep default behavior unchanged by disabling early stopping unless configured.
- Allow `MultiHeadStockModel.fit(..., eval_df=None)` so walk-forward folds can pass a fold-internal validation frame later or immediately when available.
- Surface sklearn fallback warning in model metadata and pipeline diagnostics/report.
- Preserve guardrail: predictions, recommendations, rankings, and signals still use model outputs only; news/disclosures remain display-only context.

## Approach
Use additive dataclass fields and constructor parameters with defaults. Keep artifact version compatible because old artifacts can load with defaults. Tests cover constructor metadata, LightGBM fit kwargs via a fake backend, and fallback warning metadata.

## Testing
Run targeted model/config tests, pipeline smoke test, then sample pipeline command from `AGENTS.md`.
