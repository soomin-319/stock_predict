# 10. 설정 및 환경변수

`src/config/settings.py`가 전체 파이프라인 설정을 관리한다.

## AppConfig 전체 구조

```python
# src/config/settings.py:97
@dataclass
class AppConfig:
    universe: UniverseConfig
    feature: FeatureConfig
    external: ExternalFeatureConfig
    training: TrainingConfig
    signal: SignalConfig
    investment_criteria: InvestmentCriteriaConfig
    backtest: BacktestConfig
```

---

## 설정 섹션별 상세

### UniverseConfig — 유니버스 설정

```python
@dataclass
class UniverseConfig:
    name: str = "KOSPI200"
    expected_size: int = 200
```

### FeatureConfig — 피처 설정

```python
@dataclass
class FeatureConfig:
    lookback_windows: list[int] = [1, 2, 3, 5, 10, 20, 60]    # 수익률 계산 윈도우
    moving_average_windows: list[int] = [5, 10, 20, 60, 120]   # 이동평균 윈도우
    volatility_windows: list[int] = [5, 20, 60]                 # 변동성 윈도우
    rsi_period: int = 14
    cci_period: int = 20
    stochastic_period: int = 14
```

### ExternalFeatureConfig — 외부 시장 설정

```python
@dataclass
class ExternalFeatureConfig:
    enabled: bool = True
    market_symbols: list[str] = [
        "^KS11",   # KOSPI
        "^KQ11",   # KOSDAQ
        "^GSPC",   # S&P500
        "^IXIC",   # NASDAQ
        "NQ=F",    # NASDAQ 선물
        "^SOX",    # 필라델피아 반도체
        "^VIX",    # VIX
        "KRW=X",   # 원/달러
        "^TNX",    # 미국 10년물 금리
    ]
```

### TrainingConfig — 학습 설정

```python
@dataclass
class TrainingConfig:
    min_train_size: int = 252 * 3          # 최소 학습 기간 (약 3년)
    test_size: int = 252                   # 검증 윈도우 (1년)
    step_size: int = 126                   # 슬라이딩 스텝 (반년)
    quantiles: list[float] = [0.1, 0.5, 0.9]
    random_state: int = 42
    model_n_jobs: int = -1                 # -1 = CPU 전체 사용
    model_head_n_jobs: int = 1
    walk_forward_n_jobs: int = -1
    use_gpu: bool = False
    purge_gap_days: int = 1               # 타겟 누수 방지 갭
    embargo_days: int = 0
    final_model_lookback_days: int = 252 * 3  # 최종 모델 최근 N일 (0=전체)
```

### SignalConfig — 시그널 가중치

```python
@dataclass
class SignalConfig:
    return_weight: float = 0.45       # 예상 수익률 가중치
    up_prob_weight: float = 0.35      # 상승 확률 가중치
    rel_strength_weight: float = 0.20 # 상대 강도 가중치
    uncertainty_penalty: float = 0.25 # 불확실성 패널티
```

(이 값들은 tune split에서 자동 튜닝됨)

### InvestmentCriteriaConfig — 투자 기준

```python
@dataclass
class InvestmentCriteriaConfig:
    top_turnover_rank: int = 15                    # 거래대금 상위 기준 순위
    high_conviction_net_buy_krw: float = 1e11      # 고확신 순매수 기준 (1,000억원)
    rsi_buy_watch_low: float = 30.0                # RSI 매수 감시 구간 하한
    rsi_buy_watch_high: float = 35.0               # RSI 매수 감시 구간 상한
    rsi_overbought: float = 70.0                   # RSI 과매수 기준
    nasdaq_tailwind_threshold: float = 0.01        # 나스닥 테일윈드 기준 (+1%)
    nasdaq_headwind_threshold: float = -0.01       # 나스닥 헤드윈드 기준 (-1%)
    near_52w_distance_threshold: float = 0.03      # 52주 고가 근접 기준 (3% 이내)
    leader_top_n: int = 3                          # 섹터 리더 상위 N개
    leader_min_co_movers: int = 2                  # 최소 동반 상승 종목 수
    leader_min_return: float = 0.0                 # 리더 최소 수익률
```

### BacktestConfig — 백테스트 설정

```python
@dataclass
class BacktestConfig:
    top_k: int = 20                                # Top-K 선택 종목 수
    portfolio_value: float = 1_000_000_000         # 포트폴리오 규모 (10억원)
    max_daily_participation: float = 0.10          # 일 거래대금 참여율 한도
    fee_bps: float = 10.0                          # 수수료 (10bps = 0.1%)
    slippage_bps: float = 5.0                      # 슬리피지 (5bps)
    dynamic_slippage_bps: float = 10.0             # 동적 슬리피지
    conservative_slippage_multiplier: float = 1.5  # 보수적 슬리피지 배수
    aggressive_slippage_multiplier: float = 0.75   # 공격적 슬리피지 배수
    min_up_probability: float = 0.50               # 최소 상승 확률
    min_signal_score: float = 0.0                  # 최소 시그널 점수
    turnover_limit: float = 0.5                    # 일 최대 회전율 (50%)
    min_value_traded: float = 3_000_000_000        # 최소 거래대금 (30억원)
    min_external_coverage_ratio: float = 0.0       # 최소 외부 커버리지 비율
    min_investor_coverage_ratio: float = 0.5       # 최소 투자자 커버리지 비율
    max_positions_per_market_type: int = 12        # 시장별 최대 보유 종목 수
```

---

## 설정 로딩

```python
# src/config/settings.py
def load_app_config(
    config_json: str | None = None,
    overrides: dict | None = None,
) -> AppConfig
```

우선순위: **CLI 오버라이드 > config_json 파일 > 기본값**

### JSON 설정 파일 예시

```json
// configs/prod_conservative.json
{
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

| 변수 | 용도 | 관련 기능 |
|------|------|-----------|
| `OPENAI_API_KEY` | OpenAI API 인증 | 이슈 요약, 뉴스 임팩트 LLM |
| `OPENAI_MODEL` | 사용할 OpenAI 모델 | 기본: `gpt-5-mini` |
| `DART_API_KEY` | DART 공시 API | 공시 컨텍스트 (deprecated, 환경변수 사용) |
| `DART_CORP_MAP_CSV` | DART 기업 코드 매핑 | 공시 컨텍스트 |
| `NAVER_CLIENT_ID` | Naver 뉴스 API | 뉴스 컨텍스트 |
| `NAVER_CLIENT_SECRET` | Naver 뉴스 API | 뉴스 컨텍스트 |

> API 키, 인증 정보, 실제 시장 데이터는 절대 git에 커밋하지 않는다.

---

## 설정 파일 위치

| 파일 | 용도 |
|------|------|
| `configs/prod_conservative.json` | 운영용 보수적 설정 |
| `configs/research_balanced.json` | 연구용 균형 설정 |
| `configs/news_impact.example.json` | 뉴스 임팩트 OpenAI 설정 템플릿 |
| `configs/news_impact.gemma.example.json` | 뉴스 임팩트 로컬 LLM 설정 템플릿 |

---

## CLI 오버라이드 옵션 → AppConfig 매핑

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

## 설정 직렬화

```python
# src/config/settings.py
def app_config_to_dict(cfg: AppConfig) -> dict
```

파이프라인 리포트 JSON에 전체 설정이 포함된다:

```json
{
    "config": {
        "universe": {"name": "KOSPI200", "expected_size": 200},
        "backtest": {"top_k": 20, "min_value_traded": 3000000000, ...},
        ...
    }
}
```

---

## 개선 및 수정 제안

> 우선순위: **P0(불일치) > P1(검증 강화) > P2(품질/문서)**.

### P0 — 문서의 `load_app_config` 시그니처 불일치

- **문제**: 문서는 `load_app_config(config_json=None, overrides=None)`로 적었지만 실제 인자명은 `config_path`다(`settings.py:161`). CLI 매핑 표를 따라 호출하면 `TypeError`가 난다.
- **제안**: 문서를 `load_app_config(config_path=None, overrides=None)`로 정정.

### P1 — 교차 필드(cross-field) 검증 부재

- **문제**: `_validate_app_config`는 개별 필드만 검사한다(`settings.py:136-158`). `min_train_size > test_size`, `step_size <= test_size` 같은 관계 제약이 없어, 잘못된 조합이면 walk-forward가 폴드를 0개 만들고 조용히 적응형 폴백으로 빠진다.
- **제안**: 관계 제약 검증 추가(위배 시 명확한 `ValueError`). 적응형 폴백이 동작했는지 리포트에 기록.

### P1 — 검증되지 않는 설정 섹션 다수

- **문제**: `SignalConfig`(가중치), `InvestmentCriteriaConfig`(임계값), `BacktestConfig`의 `fee_bps/slippage_bps/dynamic_slippage_bps/top_k 상한/max_positions_per_market_type` 등은 검증 대상에서 빠져 있다(`settings.py`). 음수 수수료·과대 슬리피지·`max_positions=0` 등이 무방비로 통과한다.
- **제안**: 비용 bps는 `allow_zero` 양수, 가중치/패널티는 합리 범위, `max_positions_per_market_type >= 1` 등으로 검증 확대. 가중치 합계 정규화 정책도 명시.

### P1 — 미사용/오해 소지 설정값 정리

- **문제**: `InvestmentCriteriaConfig.near_52w_distance_threshold(0.03)`는 피처 생성에서 사용되지 않고 코드가 0.95를 하드코딩한다(`03_features.md`/`06_signal_policy.md` 참고). 설정만 보고는 0.97로 오해한다. `SignalConfig` 기본값도 튜너가 `up_prob_weight=0.30`으로 덮어쓴다(`05_validation.md` 참고).
- **제안**: 미사용 설정은 실제로 연결하거나 제거. "런타임에 튜닝으로 덮어쓰는 필드"는 주석/문서에 명시.

### P2 — 설정 스키마 버전·오타 친화 오류

- **문제**: `_merge_dataclass_config`는 알 수 없는 키에 `ValueError`를 던진다(`settings.py:112`) — 엄격해서 좋지만 오타 시 "가장 가까운 키" 제안이 없다. 또한 스키마 버전 필드가 없어 설정 포맷 변경 추적이 어렵다.
- **제안**: 오류 메시지에 후보 키(difflib) 제안 추가, `config_schema_version` 도입 및 리포트 기록.

### P2 — 환경변수 문서 보강

- **문제**: 표에 `OPENAI_API_KEY/OPENAI_MODEL/DART_*/NAVER_*`가 있으나 ngrok·웹훅 관련 비밀(`09_chatbot.md`의 웹훅 시크릿 제안 포함)이나 `model`/`temperature` 기본값 출처가 분산돼 있다.
- **제안**: 모든 환경변수·기본값·우선순위(CLI > env > config_json > 기본값)를 한 표로 통합하고, 비밀은 `.env`/시크릿 매니저 사용을 권장으로 명시.
