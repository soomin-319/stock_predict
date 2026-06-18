# 10. 설정 및 환경변수

`src/config/settings.py`는 파이프라인의 핵심 설정을 `AppConfig` dataclass로 관리한다. 모든 출력은 연구·운영 지원용이며 투자 조언이나 자동매매 시스템이 아니다. 매수/매도/보유 판단은 `predicted_return`(다음 날 기대수익률)을 기준으로 해야 하며, 뉴스와 공시는 표시·검토용 컨텍스트일 뿐 기대수익률, 순위, 추천, 신호를 바꾸면 안 된다.

---

## AppConfig 전체 구조

```python
@dataclass
class AppConfig:
    config_schema_version: int = 1
    universe: UniverseConfig
    feature: FeatureConfig
    external: ExternalFeatureConfig
    training: TrainingConfig
    signal: SignalConfig
    investment_criteria: InvestmentCriteriaConfig
    backtest: BacktestConfig
```

`config_schema_version`은 설정 스키마 추적용이다. 현재 지원 버전은 `1`이다.

---

## 설정 섹션별 상세

### UniverseConfig: 유니버스 설정

```python
@dataclass
class UniverseConfig:
    name: str = "KOSPI200"
    expected_size: int = 200
```

- `expected_size`는 양의 정수여야 한다.

### FeatureConfig: 피처 생성 설정

```python
@dataclass
class FeatureConfig:
    lookback_windows: list[int] = [1, 2, 3, 5, 10, 20, 60]
    moving_average_windows: list[int] = [5, 10, 20, 60, 120]
    volatility_windows: list[int] = [5, 20, 60]
    rsi_period: int = 14
    cci_period: int = 20
    stochastic_period: int = 14
```

- window 목록은 비어 있지 않은, 중복 없는, 오름차순 양의 정수 목록이어야 한다.
- 기간 값(`rsi_period`, `cci_period`, `stochastic_period`)은 양의 정수여야 한다.

### ExternalFeatureConfig: 외부 시장 피처 설정

```python
@dataclass
class ExternalFeatureConfig:
    enabled: bool = True
    market_symbols: list[str] = [
        "^KS11", "^KQ11", "^GSPC", "^IXIC", "NQ=F",
        "^SOX", "^VIX", "KRW=X", "^TNX",
    ]
```

- `enabled`는 boolean이어야 한다.
- `market_symbols`는 비어 있지 않은 문자열 목록이어야 한다.

### TrainingConfig: 학습/검증 설정

```python
@dataclass
class TrainingConfig:
    min_train_size: int = 252 * 3
    test_size: int = 252
    step_size: int = 126
    quantiles: list[float] = [0.1, 0.5, 0.9]
    random_state: int = 42
    model_n_jobs: int = -1
    model_head_n_jobs: int = 1
    walk_forward_n_jobs: int = -1
    use_gpu: bool = False
    purge_gap_days: int = 1
    embargo_days: int = 0
    final_model_lookback_days: int = 252 * 3
    walk_forward_lookback_days: int = 0
```

검증 규칙:

- `min_train_size`, `test_size`, `step_size`는 양의 정수여야 한다.
- `min_train_size > test_size`여야 walk-forward 학습 구간과 검증 구간이 분리된다.
- `step_size <= test_size`여야 검증 window가 비정상적으로 건너뛰지 않는다.
- `purge_gap_days`, `embargo_days`, `final_model_lookback_days`, `walk_forward_lookback_days`는 0 이상의 정수여야 한다.
- `quantiles`는 `(0, 1)` 범위의 중복 없는 오름차순 값 3개 이상이어야 한다.

### SignalConfig: 신호 점수 가중치

```python
@dataclass
class SignalConfig:
    return_weight: float = 0.45
    up_prob_weight: float = 0.35
    rel_strength_weight: float = 0.20
    uncertainty_penalty: float = 0.25
```

- 모든 값은 0 이상이어야 한다.
- `return_weight + up_prob_weight + rel_strength_weight > 0`이어야 한다.
- 학습 split에서 signal tuning을 수행하면 런타임에 값이 조정될 수 있다.
- 뉴스/공시/LLM 결과는 이 가중치나 `predicted_return`을 변경하면 안 된다.

### InvestmentCriteriaConfig: 투자 기준/표시 규칙

```python
@dataclass
class InvestmentCriteriaConfig:
    top_turnover_rank: int = 15
    high_conviction_net_buy_krw: float = 100_000_000_000.0
    rsi_buy_watch_low: float = 30.0
    rsi_buy_watch_high: float = 35.0
    rsi_overbought: float = 70.0
    nasdaq_tailwind_threshold: float = 0.01
    nasdaq_headwind_threshold: float = -0.01
    near_52w_distance_threshold: float = 0.03
    leader_top_n: int = 3
    leader_min_co_movers: int = 2
    leader_min_return: float = 0.0
```

검증 규칙:

- rank/count 값은 양의 정수여야 한다.
- RSI 기준은 0~100 범위이며 `rsi_buy_watch_low <= rsi_buy_watch_high <= rsi_overbought`여야 한다.
- `nasdaq_headwind_threshold < nasdaq_tailwind_threshold`여야 한다.
- `near_52w_distance_threshold`는 0~1 비율이다. `0.03`은 52주 고점 대비 3% 이내를 뜻한다.
- `near_52w_distance_threshold`는 `src/features/investment_signals.py`의 `near_52w_high_flag` 생성에 실제로 사용된다.

### BacktestConfig: 백테스트/포트폴리오 제약

```python
@dataclass
class BacktestConfig:
    top_k: int = 20
    portfolio_value: float = 1_000_000_000.0
    max_daily_participation: float = 0.10
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    dynamic_slippage_bps: float = 10.0
    conservative_slippage_multiplier: float = 1.5
    aggressive_slippage_multiplier: float = 0.75
    min_up_probability: float = 0.50
    min_signal_score: float = 0.0
    turnover_limit: float = 0.5
    min_value_traded: float = 3_000_000_000.0
    min_external_coverage_ratio: float = 0.0
    min_investor_coverage_ratio: float = 0.5
    max_positions_per_market_type: int = 12
```

검증 규칙:

- `top_k`, `max_positions_per_market_type`은 양의 정수여야 한다.
- `portfolio_value`, multiplier 값은 양수여야 한다.
- bps 비용(`fee_bps`, `slippage_bps`, `dynamic_slippage_bps`)은 0 이상이어야 한다.
- 참여율/확률/커버리지/회전율 비율은 0~1 범위여야 한다.

---

## 설정 로딩

```python
def load_app_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> AppConfig
```

우선순위:

1. CLI override로 만들어진 `overrides`
2. `config_path` JSON 파일
3. dataclass 기본값

파이프라인 CLI 옵션 이름은 `--config-json`이지만 내부적으로는 `load_app_config(config_path=...)`에 전달된다.

알 수 없는 설정 키가 들어오면 `ValueError`가 발생한다. 오타와 가까운 필드명이 있으면 예: `did you mean 'min_train_size'?` 형태로 제안한다.

### JSON 설정 파일 예시

```json
{
  "config_schema_version": 1,
  "backtest": {
    "top_k": 10,
    "min_value_traded": 5000000000,
    "min_up_probability": 0.55
  },
  "training": {
    "walk_forward_n_jobs": 4
  }
}
```

```bash
stock-predict --config-json configs/prod_conservative.json
```

---

## 환경변수

| 변수 | 용도 | 우선순위/기본값 |
|------|------|----------------|
| `OPENAI_API_KEY` | 이슈 요약, 뉴스 영향 LLM 인증 | CLI `--openai-api-key`가 우선 |
| `OPENAI_MODEL` | OpenAI 모델명 | CLI `--openai-model` > env > API key가 있으면 `gpt-5-mini` |
| `DART_API_KEY` | DART 공시 API | CLI `--dart-api-key`가 우선 |
| `DART_CORP_MAP_CSV` | DART 기업 코드 매핑 | CLI `--dart-corp-map-csv`가 우선 |
| `NAVER_CLIENT_ID` | Naver 뉴스 API client id | CLI `--naver-client-id`가 우선 |
| `NAVER_CLIENT_SECRET` | Naver 뉴스 API secret | CLI `--naver-client-secret`이 우선 |

비밀값(API key, ngrok token 등)은 git에 커밋하지 않는다. 로컬 `.env`, 셸 환경변수, CI secret, 또는 CLI 인자로만 주입한다.

---

## 설정 파일 위치

| 파일 | 용도 |
|------|------|
| `configs/prod_conservative.json` | 운영형 보수 설정 |
| `configs/research_balanced.json` | 연구형 균형 설정 |
| `configs/news_impact.example.json` | 뉴스 영향/OpenAI 설정 예시 |
| `configs/news_impact.gemma.example.json` | 뉴스 영향/로컬 LLM 설정 예시 |

---

## CLI override와 AppConfig 매핑

| CLI 옵션 | AppConfig 경로 |
|----------|----------------|
| `--min-value-traded` | `backtest.min_value_traded` |
| `--turnover-limit` | `backtest.turnover_limit` |
| `--min-up-probability` | `backtest.min_up_probability` |
| `--min-signal-score` | `backtest.min_signal_score` |
| `--min-external-coverage-ratio` | `backtest.min_external_coverage_ratio` |
| `--min-investor-coverage-ratio` | `backtest.min_investor_coverage_ratio` |
| `--portfolio-value` | `backtest.portfolio_value` |
| `--max-daily-participation` | `backtest.max_daily_participation` |
| `--max-positions-per-market-type` | `backtest.max_positions_per_market_type` |
| `--walk-forward-n-jobs` | `training.walk_forward_n_jobs` |
| `--model-n-jobs` | `training.model_n_jobs` |
| `--model-head-n-jobs` | `training.model_head_n_jobs` |

---

## 설정 직렬화와 리포트

```python
def app_config_to_dict(cfg: AppConfig) -> dict
```

파이프라인 리포트 JSON에는 전체 설정이 포함된다.

```json
{
  "config": {
    "config_schema_version": 1,
    "universe": {"name": "KOSPI200", "expected_size": 200},
    "backtest": {"top_k": 20, "min_value_traded": 3000000000.0}
  }
}
```

CSV 출력은 `result/` 아래에 두고 `utf-8-sig` 인코딩을 사용한다. 샘플/유니버스 입력은 `data/` 아래에 둔다.

---

## 운영 체크리스트

- 설정 변경 후 `pytest tests/test_operational_hardening.py -q`를 실행한다.
- 파이프라인 영향이 있으면 `pytest tests/test_pipeline_smoke.py -q`를 실행한다.
- 샘플 실행:

```bash
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

- 뉴스/공시/LLM 컨텍스트는 한국어 뉴스 우선 수집 원칙을 따른다.
- 뉴스/공시/LLM 컨텍스트는 표시·검토용으로만 사용하며 기대수익률, 랭킹, 추천, 신호를 변경하지 않는다.
