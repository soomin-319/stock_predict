# 10. 설정 및 환경변수

`src/config/settings.py`는 파이프라인 핵심 설정을 `AppConfig` dataclass로 관리한다. 모든 출력은 연구·운영
보조용이며, 매수/매도/보유 판단은 `predicted_return`만 사용한다. 뉴스/공시는 표시·검토용 컨텍스트일 뿐 기대수익률·순위·추천·신호를 바꾸지 않는다.

---

## AppConfig 구조

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

`config_schema_version`은 스키마 추적용이며 현재 지원 버전은 `1`이다.

---

## 설정 섹션

### UniverseConfig
`name="KOSPI200"`, `expected_size=200`. `expected_size`는 양의 정수.

### FeatureConfig
`lookback_windows=[1,2,3,5,10,20,60]`, `moving_average_windows=[5,10,20,60,120]`,
`volatility_windows=[5,20,60]`, `rsi_period=14`, `cci_period=20`, `stochastic_period=14`.
window 목록은 비어 있지 않은 중복 없는 오름차순 양의 정수, 기간 값은 양의 정수여야 한다.

### ExternalFeatureConfig
`enabled=True`, `market_symbols=["^KS11","^KQ11","^GSPC","^IXIC","NQ=F","^SOX","^VIX","KRW=X","^TNX"]`.

### TrainingConfig
`min_train_size=756`, `test_size=252`, `step_size=126`, `quantiles=[0.1,0.5,0.9]`, `random_state=42`,
`model_n_jobs=-1`, `model_head_n_jobs=1`, `walk_forward_n_jobs=-1`, `use_gpu=False`,
`early_stopping_rounds=0`, `reg_alpha=0.0`, `reg_lambda=0.0`, `min_child_samples=20`,
`purge_gap_days=1`, `embargo_days=0`, `final_model_lookback_days=756`, `walk_forward_lookback_days=0`.

검증 규칙: `min_train_size > test_size`, `step_size <= test_size`, 갭/룩백 값은 0 이상 정수,
`quantiles`는 `(0,1)` 범위의 중복 없는 오름차순 3개 이상.

### SignalConfig
`return_weight=0.65`, `up_prob_weight=0.35`, `uncertainty_penalty=0.25`. 모든 값 0 이상,
`return_weight + up_prob_weight > 0`. 학습 split에서 시그널 튜닝 시 런타임에 조정될 수 있다([05](05_validation.md)).

### InvestmentCriteriaConfig
`top_turnover_rank=15`, `high_conviction_net_buy_krw=1,000억`, `rsi_buy_watch_low/high=30/35`,
`rsi_overbought=70`, `nasdaq_tailwind/headwind_threshold=0.01/-0.01`, `near_52w_distance_threshold=0.03`,
`leader_top_n=3`, `leader_min_co_movers=2`, `leader_min_return=0.0`.

검증 규칙: rank/count는 양의 정수, RSI는 0~100이며 `low <= high <= overbought`,
`headwind < tailwind`, `near_52w_distance_threshold`는 0~1(0.03=고점 대비 3% 이내).

### BacktestConfig
`top_k=20`, `portfolio_value=10억`, `max_daily_participation=0.10`, `fee_bps=10`, `slippage_bps=5`,
`dynamic_slippage_bps=10`, `conservative/aggressive_slippage_multiplier=1.5/0.75`, `min_up_probability=0.50`,
`min_signal_score=0.0`, `turnover_limit=0.5`, `min_value_traded=30억`, `min_external_coverage_ratio=0.0`,
`min_investor_coverage_ratio=0.5`, `max_positions_per_market_type=12`.

검증 규칙: `top_k`/`max_positions_per_market_type`는 양의 정수, `portfolio_value`/multiplier는 양수,
bps 비용은 0 이상, 참여율/확률/커버리지/회전율은 0~1. (`min_signal_score`는 현재 백테스트 필터에 사용되지 않는다.)

---

## 설정 로딩

```python
def load_app_config(config_path=None, overrides=None) -> AppConfig
```

우선순위: CLI override(`overrides`) > `config_path` JSON > dataclass 기본값. 파이프라인 CLI는 `--config-json`을
`config_path`에 전달한다. 알 수 없는 키가 들어오면 `ValueError`가 발생하며 오타에 가까운 필드를
`did you mean '...'?`로 제안한다. 로드 후 `_validate_app_config`가 전 섹션을 검증한다.

### JSON 예시

```json
{
  "config_schema_version": 1,
  "backtest": {"top_k": 10, "min_value_traded": 5000000000, "min_up_probability": 0.55},
  "training": {"walk_forward_n_jobs": 4}
}
```

```bash
stock-predict --config-json configs/prod_conservative.json
```

---

## CLI override ↔ AppConfig 매핑

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

## 환경변수

| 변수 | 용도 | 우선순위/기본값 |
|------|------|----------------|
| `OPENAI_API_KEY` | 이슈 요약, 뉴스 임팩트 LLM | CLI `--openai-api-key` 우선 |
| `OPENAI_MODEL` | OpenAI 모델명 | CLI > env > (키 있으면) `gpt-5-mini` |
| `DART_API_KEY` | DART 공시 API | CLI `--dart-api-key` 우선 |
| `DART_CORP_MAP_CSV` | DART 기업 코드 매핑 | CLI `--dart-corp-map-csv` 우선 |
| `NAVER_CLIENT_ID` | Naver 뉴스 API client id | CLI `--naver-client-id` 우선 |
| `NAVER_CLIENT_SECRET` | Naver 뉴스 API secret | CLI `--naver-client-secret` 우선 |

비밀값은 git에 커밋하지 않는다. 로컬 `.env`, 셸 환경변수, CI secret, 또는 CLI 인자로만 주입한다.

---

## 설정 직렬화

`app_config_to_dict(cfg)`는 전체 설정을 dict로 반환하며 `pipeline_report.json`의 `config` 필드에 포함된다.
CSV 출력은 `result/` 아래에 `utf-8-sig`로 저장하고, 샘플/유니버스 입력은 `data/` 아래에 둔다.

## 설정 파일 위치

| 파일 | 용도 |
|------|------|
| `configs/prod_conservative.json` | 운영형 보수 설정 |
| `configs/research_balanced.json` | 연구형 균형 설정 |
| `configs/news_impact.example.json` | 뉴스 임팩트/OpenAI 설정 예시 |
| `configs/news_impact.gemma.example.json` | 뉴스 임팩트/로컬 LLM 설정 예시 |
