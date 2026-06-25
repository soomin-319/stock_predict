# 왜 이 프로젝트는 다음날 상승/하락 방향을 사실상 못 맞추는가

> 분석 대상 실행: `run_id 20260625T011048Z_7059fbf5` (report의 git 해시는 미기록), 5종목(삼성전자·SK하이닉스·NAVER·카카오·LG화학), 다음 거래일 1일 예측.
> 모든 수치는 `result/latest/pipeline_report.json` 및 모델 아티팩트에서 직접 확인했습니다.

## TL;DR

- 검증된 방향 변별력은 **AUC 0.508, accuracy 0.518 = 사실상 동전던지기**입니다.
- 이는 **버그가 아니라 "1일 후 단일종목 방향 예측"이라는 과제의 근본 난이도**입니다. 모델이 다음날 수익률 분산의 **약 0.2%만** 설명합니다(corr 0.046 → R²≈0.0021).
- 화면의 "상승확률 90%대"는 *정확도*가 아니라 *과신 국면의 확신도*이며, 그 확신은 변별력으로 입증된 적이 없습니다.
- 가장 큰 구조적 페널티: **수급 피처가 코드 스텁 때문에 전부 0**으로 죽어 있음.

---

## 1. "방향을 맞춘다"의 정확한 정의

`src/validation/metrics.py:44-51`:

```python
def classification_metrics(y_true, y_prob, threshold=0.5):
    y_hat = (y_prob >= threshold).astype(int)      # 0.5 넘으면 "상승" 콜
    return {
        "accuracy": float(np.mean(y_true == y_hat)),   # 실제 방향과 일치한 비율
        "roc_auc": float(_roc_auc_binary(...)),
    }
```

- `y_true` = `target_up` (다음날 실제 종가가 오르면 1).
- **accuracy = "오를지/내릴지 부호 콜이 맞은 비율"**. 확률 *수치*를 맞혔는지가 아니라 순수 방향 적중률.
- `roc_auc` = 상승일과 하락일을 분리하는 변별력(기저비율 무관). 0.5 = 무작위.
- 확률 *수치*의 정확도(보정)는 별도로 Brier/ECE가 측정.

---

## 2. 측정된 실태 (walk-forward / OOF, 누수 제거된 정직한 일반화)

| 지표 | 값 | 의미 |
|---|---|---|
| 수익률 corr | **0.046** | 예측-실현 거의 무상관 (R²≈0.2%) |
| 방향 accuracy | **0.518** | 절반 + α |
| **ROC-AUC** | **0.508** | **변별력 ≈ 무작위** |
| OOF direction_accuracy | 0.499 / 0.510 | 회귀·랭크 헤드 부호 적중 ≈ 0.5 |
| baseline_zero accuracy | 0.475 | "항상 0 예측" |
| baseline_prev_return accuracy | 0.494 | "어제 수익률 답습" |
| validation **rank_ic** | **0.092** | 횡단면 랭킹은 *약한* 양(+) |

핵심 해석:
- 모델(0.518)이 베이스라인(0.475~0.494)을 **미세하게만** 상회 → 데이터에 착취할 단기 구조가 거의 없음.
- **AUC 0.508**이 가장 깔끔한 진단: 기저비율을 제거하면 상승/하락 변별력이 무작위와 거의 구분되지 않는다.
- 단, **rank_ic 0.092**는 "절대 방향"보다 "여러 종목 중 상대적으로 나은 것 고르기"는 *약하게* 가능함을 시사 → 이 시스템은 방향예측기보다 **랭커**에 가깝다.

---

## 3. 왜 이렇게 되는가 — 근본 원인

### 3-1. 신호 ≪ 잡음 (가장 큰 이유)
다음날 수익률 = (작은 예측가능 드리프트) + (큰 무작위 충격).
- 일간 변동성은 ~2~3%인데 모델이 잡는 드리프트는 R²≈0.2% 수준.
- 따라서 **부호는 사실상 그날 밤의 뉴스·수급 충격이 결정** → ~50%.
- 0 근처(±0.1%) 작은 수익률 날은 부호가 거의 랜덤인데, 이런 날이 다수라 달성 가능한 정확도 상한 자체를 깎는다.

### 3-2. 1일 지평선이 가장 어렵다
- 시장이 준효율적 → 예측가능 정보 대부분이 이미 가격에 반영.
- 기술지표(모멘텀/RSI/MACD)는 다음날 방향엔 약하고 레짐 의존적.
- 지평선이 길수록(주간/월간) 드리프트가 노이즈 위로 누적되어 예측이 쉬워지는데, 현재는 **1일 고정**.

### 3-3. 수급 피처 전멸 (구조적 자해)
`src/data/investor_context.py:51-61`의 `_fetch_flow`가 **구현 안 된 빈 스텁**:

```python
def _fetch_flow(symbols, start, end):
    coverage = {... "status": "not_configured", "source": "input_csv_only", ...}
    _ = (start, end)                          # 인자 버림
    return pd.DataFrame(columns=[...]), coverage   # 항상 빈 DF
```

→ 모델 입력에 포함된 `foreign_net_buy / institution_net_buy` 및 의존 수급 피처 16개(`foreign_buy_signal`, `institution_net_buy_z20`, `smart_money_strength`, `*_high_conviction_buy_flag` 등)가 전부 상수 0이고 feature importance도 0.
`investor_event_score`, `jongbae_score`는 최신 결과에서 0은 아니지만 현재 모델 입력이 아니라 표시/후처리 성격에 가깝다. 한국 시장에서 단기 예측력 있는 수급 신호원 한 묶음이 통째로 죽어 있는 셈.
부활 방법은 별도 문서 참조: `docs/INVESTOR_FLOW_FEATURE_REVIVAL_PLAN.md`.

### 3-4. 정칙화·캘리브레이션 미흡 (과신 유발)
- 모델 설정 `reg_alpha=0, reg_lambda=0`(`pipeline_report.json`의 model 블록) → 정규화 없음 → 강추세 입력에 과신.
- 결과적으로 분류기 raw 확률이 5종목 모두 0.9+로 쏠림(아래 4장).

---

## 4. "상승확률 90%대 vs 적중 50%" 역설

표시된 상승확률은 `cls_model.predict_proba(x)[:,1]` (`src/models/lgbm_heads.py:281`)이며, 캘리브레이션(IsotonicRegression)이 최신 예측에도 적용됩니다(`src/pipeline.py:728`).

이 실행에서 모델 .pkl을 직접 로드해 측정한 **보정 전(raw) vs 보정 후(calib)**:

| 종목 | RAW | CALIB |
|---|---|---|
| 카카오 | 96.3% | 97.5% |
| LG화학 | 96.0% | 97.3% |
| NAVER | 94.8% | 96.7% |
| SK하이닉스 | 94.0% | 95.9% |
| 삼성전자 | 90.8% | 94.1% |

- raw가 이미 90~96%로 극단적이고 **보정은 이를 낮추기는커녕 +1~3%p 더 올림**.
- 이유: 보정 표본의 92%(2,432/2,641)가 0.4~0.5에 몰려 있고 0.9~1.0 구간은 단 1표본. 현재 5종목은 **표본 ≈1개짜리 희소 극단 꼬리**로 들어가 보정 신뢰도가 낮다.
- eval **ECE 0.098, Brier 0.262** → 보정 불완전.

**결론:** 90%대는 "검증된 적중률"이 아니라 **유포리아 국면(KOSPI +127%, 반도체 폭등)에서 분류기가 '다 오른다'고 일괄 과신한 확신도**다. AUC 0.508이 말해주듯 그 확신엔 변별력 근거가 없다. → **액면 신뢰 금지.**

---

## 5. 개선 레버 (영향 큰 순)

1. **수급 피처 부활** — `_fetch_flow` 실제 구현(pykrx/KRX). 죽은 신호원 회복 + 과신 견제. → `docs/INVESTOR_FLOW_FEATURE_REVIVAL_PLAN.md`.
2. **뉴스/공시 계량 피처 활용** — 정책상 계산 입력 허용. 단, 발표시각 기준 누수 방지, 한국 뉴스 우선, deterministic scoring, 결측/429 fallback, OOF 검증을 함께 적용.
3. **횡단면 랭킹으로 목표 재정의** — 절대 방향(불가능에 가까움) 대신 "상대적 우위 종목 선택"(rank_ic 0.092의 약한 + 신호를 키움).
4. **지평선 확대** — 1일 → 주간/월간. 드리프트가 노이즈 위로 누적.
5. **정칙화 + 캘리브레이션 수축** — `reg_alpha/lambda` > 0, 희소 꼬리는 0.5 쪽으로 shrinkage(또는 신뢰구간 동반 표기).
6. **레짐 조건부 모델** — 추세/변동성 레짐별 분리 학습.

> 현실적 기대치: 다음날 단일종목 방향은 전문 퀀트도 50%를 *살짝* 넘는 수준이고, 대규모 분산·낮은 비용·레버리지로 수익화하지 종목별 고적중률로 하지 않는다. 위 레버는 **"50%를 못 넘는 상태 → 미세하지만 안정적인 +엣지 + 정직한 확신도 표기"** 를 목표로 한다. "고적중률"을 약속하는 방향이 아님.

---

## 부록: 수치 재현 방법

- 검증 지표: `result/latest/pipeline_report.json`의 `walk_forward`, `baselines`, `oof_diagnostics`, `tuned_signal`, `probability_calibration`.
- 보정 전 raw 확률(4장 표): 모델 아티팩트와 최신 피처행을 로드해 분류기 헤드를 직접 호출.

```python
import pandas as pd
from src.models.lgbm_heads import MultiHeadStockModel
m = MultiHeadStockModel.load("result/runs/<run_id>/model/model.pkl")
df = pd.read_csv("result/latest/csv/result_detail.csv")   # 최신 예측행 = 111개 피처 포함
raw = m.predict(df).up_probability                         # 보정 전(raw) 상승확률
calib = pd.to_numeric(df["up_probability"], errors="coerce")  # 보정 후(파이프라인이 표시)
```

- 방향 accuracy/AUC 정의: `src/validation/metrics.py:44-51`.
