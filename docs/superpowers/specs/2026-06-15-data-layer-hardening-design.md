# Data Layer Hardening Design

## Goal

Implement every improvement proposed in `docs/02_data.md` while preserving
existing pipeline interfaces and preventing display-only news or disclosure
context from affecting expected returns, rankings, recommendations, or signals.

## Scope

- Correct KOSPI/KOSDAQ yfinance symbol resolution.
- Load and save OHLCV CSV files consistently with UTF-8 BOM support.
- Use adjusted prices and retry transient yfinance failures.
- Report real-data fetch coverage and failed symbols.
- Make duplicate, zero-volume, and extreme-return cleaning policies explicit.
- Centralize the default real-data start date and document public contracts.
- Add deterministic tests without live network calls.

Broad data-configuration refactoring and automated trading behavior are out of
scope.

## Architecture

### Symbol Resolution and Market Fallback

Add cached KRX ticker-to-symbol lookup behavior based on the repository-managed
KRX and KOSPI200 symbol-name CSV files. A six-digit ticker resolves to the
mapped yfinance-style symbol when available, including `.KQ` for KOSDAQ.

For an unmapped six-digit ticker, normalization keeps the compatible `.KS`
default. During download, Korean equity candidates are attempted in this order:

1. the explicitly supplied or mapped symbol;
2. the alternate Korean market suffix (`.KS` or `.KQ`).

The first successful candidate becomes the `Symbol` written to fetched OHLCV
data. Explicit non-Korean or already-suffixed symbols remain supported.

### Adjusted Downloads and Retry Policy

Both real OHLCV downloads and external-market feature downloads use
`auto_adjust=True`. This avoids artificial return jumps from splits and
distributions.

Each provider candidate receives at most three attempts. Retry delay uses
exponential backoff with a one-second base delay. Tests inject or patch sleeping
so they remain fast and deterministic. Empty responses and provider exceptions
are retryable; exhausted attempts are reported as failures without provider
noise leaking to normal console output.

### Fetch Coverage

Introduce a structured fetch result containing the fetched frame and coverage
metadata while preserving `fetch_real_ohlcv(...) -> DataFrame` for existing
callers. Save and append helpers retain their current return type and expose the
latest coverage through a small module-level accessor.

Coverage records:

- requested, successful, and failed symbol counts;
- success ratio;
- failed input symbols;
- candidate fallback usage;
- retried symbols and total retry count;
- resolved input-to-provider symbol details.

CLI refresh operations capture this metadata and pass it to `run_pipeline`.
`pipeline_report.json` exposes it as `data_fetch_coverage`. Runs without a
refresh report a disabled/empty coverage structure.

### CSV Encoding

`load_ohlcv_csv` reads with `encoding="utf-8-sig"`, which accepts UTF-8 files
with or without a BOM. Data-layer save and append helpers write
`encoding="utf-8-sig"` for consistency with report artifacts and Windows Excel.
Existing optional columns remain preserved.

### Cleaning Policy

`clean_ohlcv` keeps its existing public signature and applies these policies:

- coerce dates and required numeric fields;
- fill a missing `Symbol` column with `"UNKNOWN"`;
- reject invalid or non-positive OHLC values and negative volume;
- enforce OHLC range consistency;
- resolve duplicate `(Date, Symbol)` rows by selecting the row with the highest
  numeric `Volume`, with the latest input row as a deterministic tie-breaker;
- preserve zero-volume rows and add `is_zero_volume`;
- compute symbol-local close-to-close returns and add
  `is_extreme_return` when absolute daily return exceeds 40%.

Flagged rows remain in cleaned history so legitimate events and continuity are
not silently lost. Before feature building, rows flagged by either
`is_extreme_return` or `is_zero_volume` are removed from the model input. This
prevents flagged observations from affecting model features, targets,
validation, or predictions.

The flags are operational data-quality fields only; they do not alter news or
disclosure policy.

### Defaults and Contracts

Define one `DEFAULT_REAL_START_DATE = "2020-01-01"` constant in the data layer.
Fetcher defaults and the CLI `--real-start` default use this constant.

Update `docs/02_data.md` to document:

- market-aware normalization and fallback;
- adjusted-price and retry behavior;
- `data_fetch_coverage`;
- BOM-safe loading and saving;
- cleaning flags and model exclusion policy;
- the optional `symbol=` loader argument;
- exact, partial, normalized-space, and fuzzy company-name matching behavior;
- the centralized default start date.

## Data Flow

1. CLI resolves raw user symbols using repository KRX mappings.
2. The fetcher tries market candidates with bounded retries and adjusted prices.
3. Save or append writes a BOM-safe CSV and retains fetch coverage.
4. The loader reads BOM-safe CSV and normalizes basic columns.
5. The cleaner selects deterministic duplicates and adds quality flags.
6. The pipeline excludes flagged rows before feature creation.
7. The artifact report includes refresh coverage when a refresh occurred.

## Error Handling

- Missing yfinance remains a clear runtime error on live-fetch paths.
- Total fetch failure remains fatal for full refresh and non-destructive for
  append, matching current behavior.
- Partial fetch failure succeeds but records failed symbols and reduced
  coverage.
- Unknown Korean tickers try both market suffixes before failing.
- Existing input CSV remains unchanged when append fetches no data.
- Empty data after quality-flag exclusion fails through the existing core
  pipeline validation path rather than producing misleading predictions.

## Testing

Use TDD for each behavior:

- BOM and non-BOM CSV inputs load successfully.
- Missing `Symbol` is filled and explicit `symbol=` remains supported.
- known KOSDAQ tickers normalize to `.KQ`;
- unknown Korean tickers fall back from `.KS` to `.KQ`;
- the successful fallback symbol is written to output;
- downloads use `auto_adjust=True`;
- transient failures retry and exhausted retries are reported;
- coverage counts, ratios, failed symbols, fallbacks, and retries are correct;
- save and append outputs use UTF-8 BOM and preserve optional columns;
- duplicate rows select maximum volume with deterministic ties;
- zero-volume and extreme-return rows receive flags and remain in cleaned data;
- flagged rows are excluded before feature creation;
- external-market downloads use adjusted prices and retry;
- CLI and fetcher share the centralized start-date constant;
- pipeline report contains `data_fetch_coverage`;
- impacted tests, pipeline smoke test, and full pytest suite pass.

## Compatibility

- Existing fetch, save, append, loader, cleaner, and pipeline call patterns
  remain valid.
- Explicit symbols and default KOSPI200 behavior remain supported.
- `fetch_real_ohlcv` continues returning a DataFrame.
- Save and append helpers continue returning their output path.
- News and disclosures remain display-only context.
