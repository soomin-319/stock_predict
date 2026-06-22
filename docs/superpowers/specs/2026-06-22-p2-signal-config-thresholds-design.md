# P2 Signal Config Thresholds Design

## Goal

Move the recommendation buy/sell thresholds out of hard-coded `signal_policy.py` constants and into `SignalConfig`, while keeping the core guardrail: user-facing buy/sell/hold decisions depend only on `predicted_return`.

This PR also locks scalar and vectorized recommendation behavior together so future policy changes cannot drift between row-level helpers and the production vectorized path.

## Scope

In scope:

- Add configurable percent thresholds to `SignalConfig`.
- Validate threshold shape in `_validate_app_config`.
- Use the configured thresholds in scalar and vectorized recommendation helpers.
- Preserve default behavior:
  - `predicted_return > +2.0` means `"매수"`.
  - `predicted_return <= -2.0` means `"매도"`.
  - all other values and missing values mean `"관망"`.
- Add tests for defaults, overrides, validation failures, and scalar/vectorized equivalence.

Out of scope:

- Changing recommendation semantics beyond making thresholds configurable.
- Moving confidence-label or event-boost constants.
- Letting news, disclosures, signal score, probability, or uncertainty change the recommendation.
- Refactoring all row/vectorized helpers in `signal_policy.py`.

## Data Model

`src/config/settings.py` extends `SignalConfig`:

```python
@dataclass
class SignalConfig:
    return_weight: float = 0.65
    up_prob_weight: float = 0.35
    uncertainty_penalty: float = 0.25
    recommendation_buy_threshold_pct: float = 2.0
    recommendation_sell_threshold_pct: float = -2.0
```

Validation rules:

- Both values must be numeric `int` or `float`; `bool` is invalid.
- `recommendation_buy_threshold_pct` must be positive.
- `recommendation_sell_threshold_pct` must be negative.
- `recommendation_sell_threshold_pct < recommendation_buy_threshold_pct`.

These are percentages, matching existing `predicted_return` units.

## Signal Policy Changes

`src/domain/signal_policy.py` imports `SignalConfig` and adds a small default resolver:

```python
DEFAULT_SIGNAL_CONFIG = SignalConfig()

def _signal_cfg(cfg: SignalConfig | None) -> SignalConfig:
    return cfg or DEFAULT_SIGNAL_CONFIG
```

`recommendation_from_signal()` gains an optional keyword-only config parameter:

```python
def recommendation_from_signal(
    signal_score,
    predicted_return,
    up_probability=None,
    uncertainty_score=None,
    *,
    signal_cfg: SignalConfig | None = None,
) -> str:
```

The first four parameters stay for backward compatibility. They remain ignored except for `predicted_return`.

Vectorized recommendation becomes:

```python
def _recommendation_series(df: pd.DataFrame, signal_cfg: SignalConfig | None = None) -> pd.Series:
```

It uses the same thresholds and same inclusivity rules as scalar logic:

- buy: `predicted_return > buy_threshold`
- sell: `predicted_return <= sell_threshold`
- hold: otherwise or missing

`build_prediction_policy_frame()` accepts an optional `signal_cfg` while keeping the existing `cfg` investment criteria argument:

```python
def build_prediction_policy_frame(
    pred_df: pd.DataFrame,
    cfg: InvestmentCriteriaConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> pd.DataFrame:
```

Existing callers can omit it and keep defaults.

## Data Flow

Pipeline already has `cfg.signal` and passes `signal_config` into prediction-frame creation. This PR threads the same config into policy-frame creation so final recommendations use the validated runtime config.

Flow:

1. `load_app_config()` builds and validates `AppConfig`.
2. `run_pipeline()` passes `cfg.signal` through prediction assembly.
3. `build_prediction_policy_frame(..., signal_cfg=cfg.signal)` creates the final recommendation columns.
4. Reports/publish/chatbot consume the resulting recommendation text as before.

## Guardrails

Recommendation still ignores:

- news/disclosure fields,
- `news_impact_*`,
- issue summary fields,
- `signal_score`,
- `up_probability`,
- `uncertainty_score`,
- event boost fields.

Only `predicted_return` and validated `SignalConfig` thresholds control `"매수"`, `"매도"`, `"관망"`.

## Tests

Add or update deterministic pytest tests:

- `tests/test_settings.py` or existing config tests:
  - default thresholds are present in `app_config_to_dict()`;
  - override thresholds load successfully;
  - invalid threshold order/sign/type raises `ValueError`.
- `tests/test_signal_policy_recommendation.py`:
  - default boundary behavior remains unchanged;
  - custom thresholds change only the predicted-return cutoffs;
  - signal/probability/uncertainty inputs remain ignored.
- `tests/test_signal_policy.py`:
  - scalar `recommendation_from_signal(..., signal_cfg=custom)` equals vectorized `_recommendation_series(..., signal_cfg=custom)` for edge values including NaN.
- `tests/test_pipeline_smoke.py`:
  - custom `SignalConfig` reaches policy-frame recommendations in the pipeline smoke path.

Verification:

```powershell
pytest tests/test_signal_policy_recommendation.py tests/test_signal_policy.py tests/test_pipeline_smoke.py -q
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
pytest -q
```

## Compatibility

Backward compatibility is preserved:

- Existing positional calls to `recommendation_from_signal()` still work.
- Existing `build_prediction_policy_frame(pred_df, cfg=...)` calls still work.
- Default config reproduces current hard-coded `±2.0%` policy.
- Config schema version remains `1` because added dataclass defaults make old configs load without migration.

## Spec Self-Review

- No placeholders or deferred requirements.
- Scope is limited to recommendation thresholds plus scalar/vectorized recommendation equivalence.
- Guardrail remains explicit: recommendations use only `predicted_return` plus config thresholds.
- Threshold inclusivity is unambiguous and matches current behavior.
