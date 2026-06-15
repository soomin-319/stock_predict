# Feature Layer Hardening Design

## Goal

Implement every improvement listed in `docs/03_features.md` while preserving the
pipeline rule that news and disclosures remain display-only context and never
affect model features, expected-return ranking, or recommendations.

## Scope

The work covers:

- P0 correctness fixes for external-market look-ahead leakage, future-value
  backfilling, and incorrect feature documentation.
- P1 consistency and safety fixes for the 52-week-high threshold, non-finite
  technical-indicator values, and historical Korean price-limit detection.
- P2 maintainability, performance, missing-value policy, diagnostics, tests,
  and documentation.

## External-Market Timing

External daily observations must represent only information available by the
Korean market close for a row's date. Korean indexes (`^KS11`, `^KQ11`) may be
joined on the same calendar date. Overseas indexes, rates, FX, volatility
indexes, and futures are conservatively delayed by one published observation
before joining because daily yfinance data does not identify the value
available exactly at 15:30 KST.

The external frame may use forward fill after date alignment. It must never use
backfill, because backfill copies future observations into earlier rows. Leading
unavailable periods remain missing.

## Price and Technical Features

`src/features/technical_indicators.py` is the single implementation source for
RSI, MACD, ATR, stochastic, CCI, OBV, and rolling z-score calculations.
`build_features()` computes these indicators once per symbol and concatenates
the resulting blocks once, instead of maintaining duplicate formulas.

All generated numeric feature values are normalized so positive and negative
infinity become missing values. In particular, `obv_change_5d` must never emit
infinity when OBV crosses or starts from zero.

`vol_ratio_20` keeps its existing, accurate meaning: current volume divided by
20-day average volume. Documentation will no longer describe it as a volatility
ratio.

## Threshold Ownership

`InvestmentCriteriaConfig.near_52w_distance_threshold` is the only configurable
definition of "near the 52-week high." `build_features()` produces continuous
52-week-high inputs, while `add_investment_signal_features()` owns the
threshold-derived flag.

Price-limit flags use an explicit per-row limit percentage when supplied.
Otherwise, ordinary Korean equity rows use the historical default of 15% before
2015-06-15 and 30% on or after 2015-06-15. The implementation does not guess
special ETF, newly-listed, or relisted exceptions when metadata is unavailable.

## Missing-Value Policy and Diagnostics

Feature preparation applies explicit neutral defaults only where a neutral
meaning is well defined:

- RSI and stochastic oscillators: `50`
- directional/relative-return features and MACD family: `0`
- binary flags: `0`

History-dependent absolute levels and features without a defensible neutral
value remain missing for the model's existing final fallback. To make this
visible, the feature layer adds missing-indicator columns for selected
history-dependent features and exposes a missing-rate summary helper for
pipeline diagnostics. These diagnostic columns and summaries contain no future
information.

## Performance

Price rows are sorted by symbol and date before grouped calculations. Per-symbol
technical indicator blocks are computed in a single grouped pass and combined
once. Existing vectorized grouped transforms remain where they are clearer or
faster. No optional performance dependency such as numba or bottleneck is added.

## Compatibility and Error Handling

- Input row ordering is restored before `build_features()` returns.
- Empty data frames continue to work.
- External download failures retain current coverage reporting.
- Leading unavailable external values remain missing rather than being silently
  invented.
- News and disclosure columns remain excluded by
  `DISPLAY_ONLY_CONTEXT_COLUMNS`.

## Testing

Tests are added before each implementation change and must demonstrate:

- overseas features are delayed while Korean indexes remain same-date;
- external features never backfill leading dates;
- `vol_ratio_20` matches the documented volume ratio;
- 52-week-high flags obey configuration;
- OBV changes and all selected model features contain no infinity;
- historical price-limit thresholds and explicit overrides work;
- technical helpers and `build_features()` agree;
- neutral filling and missing diagnostics behave as documented;
- row ordering and existing display-only guards remain intact.

Run focused feature tests, `tests/test_pipeline_smoke.py`, then the complete
`pytest` suite. Capture a simple feature-build timing comparison to ensure the
refactor does not materially regress runtime.

## Documentation

Rewrite `docs/03_features.md` as valid UTF-8 Korean documentation. It must
describe the implemented timing, filling, threshold, indicator, missing-value,
and diagnostic behavior without promising unsupported special-market rules.
