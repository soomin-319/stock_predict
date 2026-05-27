# Roadmap

## Current Priorities

1. Keep recommendation-policy tests fixed around the 2% next-day expected-return thresholds, with no news/disclosure or score-based override of buy/sell/hold labels.
2. Preserve public CLI flags, console scripts, `run_pipeline(...)`, and output filenames while moving helper logic into focused modules.
3. Keep tests deterministic by using sample CSVs and repo-local pytest cache/temp paths.
4. Reduce fragmented feature-frame writes and duplicate formatter/cache paths.
5. Avoid Korean mojibake normalization in this cleanup; string literal normalization should be a separate change with fixture updates.

## Near-Term Work

- Remove compatibility wrappers from `src/pipeline.py` only after tests and external imports no longer reference them.
- Add focused tests around the new `src/reports/output.py`, `src/validation/support.py`, and `src/data/cli_refresh.py` helper modules.
- Review deprecated CLI options and mark a removal version before deleting them.
- Tighten display-only issue-summary cache invalidation rules for Kakao/Colab runs.
- Add a lightweight artifact schema check for `result_simple.csv` and `result_detail.csv`.

## Deferred

- Normalize mojibake Korean literals across source, docs, and fixtures.
- Split long chatbot state handling into smaller services.
- Add CI coverage for the console script entry points.
- Add a documented model/policy version field to report JSON outputs.
