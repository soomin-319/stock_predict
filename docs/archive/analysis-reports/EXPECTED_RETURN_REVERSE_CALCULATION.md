# 예상 수익률 계산 과정 역산 가이드

## 1. 핵심 결론

최종 출력 컬럼 `predicted_return`은 모델이 예측한 **다음 거래일 로그수익률**을 일반 수익률 퍼센트로 변환한 값이다.

```text
예상 수익률(%) = (exp(예상 로그수익률) - 1) × 100
```

역산하면 다음과 같다.

```text
예상 로그수익률 = ln(1 + 예상 수익률(%) / 100)
```

`signal_score`, 상승 확률, 뉴스, 공시를 조합하여 예상 수익률을 계산하지 않는다.  
예상 수익률은 회귀 모델이 직접 예측한 로그수익률에서만 변환된다.

---

## 2. 최종 예상 수익률부터 역산하기

최종 예상 수익률을 `R_pct`, 모델의 원본 예측값을 `L_pred`라고 정의한다.

```text
R_pct  = predicted_return
L_pred = predicted_log_return
```

두 값의 관계:

```text
R_pct = (exp(L_pred) - 1) × 100
```

따라서 최종 예상 수익률에서 모델 원본 예측값을 복원할 수 있다.

```text
L_pred = ln(1 + R_pct / 100)
```

### 예시: 예상 수익률이 `+2.5%`인 경우

```text
L_pred = ln(1 + 2.5 / 100)
       = ln(1.025)
       ≈ 0.0246926
```

검산:

```text
R_pct = (exp(0.0246926) - 1) × 100
      ≈ 2.5%
```

즉, 화면이나 결과 파일에 표시된 예상 수익률이 `+2.5%`라면 회귀 모델의 원본 출력은 약 `0.02469`이다.

### 예시: 예상 수익률이 `-3.0%`인 경우

```text
L_pred = ln(1 - 3.0 / 100)
       = ln(0.97)
       ≈ -0.0304592
```

음수 예상 수익률도 같은 방식으로 역산한다.

---

## 3. 예상 종가에서 역산하기

현재 종가를 `C_now`, 예상 종가를 `C_pred`라고 정의한다.

코드의 예상 종가 계산식:

```text
C_pred = C_now × exp(L_pred)
```

`exp(L_pred) = 1 + R_pct / 100`이므로 다음 식도 동일하다.

```text
C_pred = C_now × (1 + R_pct / 100)
```

### 예상 수익률 역산

현재 종가와 예상 종가를 알고 있다면:

```text
R_pct = (C_pred / C_now - 1) × 100
```

### 현재 종가 역산

예상 종가와 예상 수익률을 알고 있다면:

```text
C_now = C_pred / (1 + R_pct / 100)
```

### 숫자 예시

```text
현재 종가 C_now = 50,000원
예상 수익률 R_pct = +2.5%
```

예상 종가:

```text
C_pred = 50,000 × (1 + 2.5 / 100)
       = 51,250원
```

예상 종가에서 수익률 역산:

```text
R_pct = (51,250 / 50,000 - 1) × 100
      = 2.5%
```

---

## 4. 모델 원본 예측값은 무엇인가

이름 때문에 혼동하기 쉬운 부분이 있다.

`MultiHeadPrediction.predicted_return`에는 퍼센트 예상 수익률이 아니라 **예상 로그수익률**이 들어간다.

```python
predicted_return = self.reg_model.predict(x)
```

이 값은 예측 프레임을 만들 때 다음 두 컬럼으로 구분된다.

```python
out["predicted_log_return"] = pred.predicted_return
out["predicted_return"] = np.expm1(out["predicted_log_return"]) * 100.0
```

정리:

| 단계 | 값 | 단위 |
|---|---|---|
| 회귀 모델 원본 출력 | `MultiHeadPrediction.predicted_return` | 로그수익률 |
| 원본 출력 보존 컬럼 | `predicted_log_return` | 로그수익률 |
| 사용자용 최종 예상 수익률 | `predicted_return` | 퍼센트 `%` |

구현 위치: `src/models/lgbm_heads.py`, `src/inference/predict.py`

---

## 5. 모델은 무엇을 학습하는가

모델이 학습하는 정답은 `target_log_return`이다.

종목별 현재 종가를 `C_t`, 다음 거래일 종가를 `C_t+1`이라고 하면:

```text
target_log_return = ln(C_t+1 / C_t)
```

코드:

```python
out["target_log_return"] = grouped["Close"].transform(
    lambda x: np.log(x.shift(-1) / x)
)
```

이를 역산하면 실제 다음 거래일 종가는 다음과 같다.

```text
C_t+1 = C_t × exp(target_log_return)
```

학습 시점에는 과거 데이터의 다음 거래일 종가가 존재하므로 `target_log_return`을 계산할 수 있다.  
실제 예측 시점에는 다음 거래일 종가가 아직 없으므로 모델이 `target_log_return`을 추정한다.

```text
과거 피처 X_t
    ↓ 회귀 모델 학습
실제 target_log_return

최신 피처 X_latest
    ↓ 학습된 회귀 모델
예상 predicted_log_return
    ↓ expm1 변환
최종 predicted_return(%)
```

구현 위치: `src/features/price_features.py`

---

## 6. 전체 계산 과정을 역순으로 추적하기

최종 결과에서 시작하여 학습 정답까지 거슬러 올라가면 다음과 같다.

### 1단계: 최종 예상 수익률

```text
predicted_return = +2.5%
```

### 2단계: 예상 로그수익률 복원

```text
predicted_log_return
= ln(1 + predicted_return / 100)
= ln(1.025)
≈ 0.0246926
```

### 3단계: 회귀 모델 출력 확인

```text
reg_model.predict(최신 피처) ≈ 0.0246926
```

회귀 모델은 최신 피처를 입력받아 이 로그수익률을 직접 출력한다.

### 4단계: 모델이 학습한 정답 형태 확인

```text
target_log_return = ln(다음 거래일 종가 / 현재 종가)
```

과거의 여러 `피처 → 실제 다음 날 로그수익률` 관계를 학습한 결과가 최신 로그수익률 예측값이다.

### 5단계: 예상 종가 검산

현재 종가가 `50,000원`이라면:

```text
predicted_close
= 50,000 × exp(0.0246926)
= 51,250원
```

---

## 7. 피처에서 예상 수익률까지의 계산

모델 입력 피처에는 가격·거래량·기술 지표·시장 지표·투자자 수급 등이 포함된다.

대표 예:

- 수익률: `ret_*`, `daily_return`, `gap_return`
- 이동평균: `ma_*`, `close_to_ma_*`
- 거래량·변동성: `vol_*`, `vol_ratio_20`, `atr_14`
- 기술 지표: `rsi_14`, `macd`, `stoch_k`, `cci_20`
- 시장 지표: KOSPI, KOSDAQ, Nasdaq, VIX, 환율 관련 컬럼
- 투자자 수급: 외국인·기관 순매수 관련 컬럼

선택된 피처를 `X`라고 하면 개념적 계산은 다음과 같다.

```text
L_pred = 회귀모델(X)
R_pct  = (exp(L_pred) - 1) × 100
```

LightGBM이 설치된 환경에서는 `LGBMRegressor(objective="regression")`를 사용한다.  
LightGBM이 없으면 `GradientBoostingRegressor(loss="squared_error")`로 대체한다.

트리 앙상블 모델이므로 최종 예상 수익률 하나만 보고 각 피처의 원래 값을 유일하게 역산할 수는 없다. 서로 다른 피처 조합이 같은 예측값을 만들 수 있기 때문이다.

구현 위치: `src/features/feature_selection.py`, `src/models/lgbm_heads.py`

---

## 8. 최종 모델 학습과 예측 순서

기본 설정에서는 최종 모델이 최근 `252 × 3 = 756`개 거래일을 사용한다.

```text
1. 과거 데이터에서 피처와 target_log_return 생성
2. 학습 가능한 행 선택
3. 기본값 기준 최근 756개 거래일 선택
4. 회귀 모델을 target_log_return에 맞춰 학습
5. 종목별 최신 행 선택
6. 최신 피처로 predicted_log_return 예측
7. predicted_return(%)와 predicted_close 계산
```

핵심 코드 흐름:

```python
model.fit(train_df, feature_columns, cfg.training.quantiles)
latest = feat.sort_values("Date").groupby("Symbol", as_index=False).tail(1)
latest_pred = model.predict(latest)
pred_df = build_scored_prediction_frame(...)
```

구현 위치: `src/pipeline.py`

---

## 9. 예상 수익률 계산에 포함되지 않는 값

다음 값들은 예상 수익률을 계산하거나 수정하지 않는다.

| 값 | 역할 |
|---|---|
| `signal_score` | 종합 신호 등급과 보조 정렬용 점수 |
| `up_probability` | 다음 거래일 상승 방향 확률 |
| `quantile_low`, `quantile_mid`, `quantile_high` | 예상 로그수익률 분포 범위 |
| 뉴스·공시 | 표시·검토용 문맥 |
| `event_boost_score` | 신호 점수 보정값 |

특히 뉴스·공시 관련 컬럼은 `DISPLAY_ONLY_CONTEXT_COLUMNS`로 분리되어 모델 피처에서 제외된다.

```text
예상 수익률 = 회귀 모델의 예상 로그수익률을 퍼센트로 변환한 값
신호 점수   = 예상 수익률과 별도로 계산되는 보조 점수
```

---

## 10. 실무 역산 체크리스트

결과 파일에서 예상 수익률을 검증할 때:

1. `predicted_return` 값을 확인한다.
2. `ln(1 + predicted_return / 100)`으로 `predicted_log_return`을 복원한다.
3. 복원값이 결과의 `predicted_log_return`과 같은지 확인한다.
4. `Close × (1 + predicted_return / 100)`으로 `predicted_close`를 검산한다.
5. 예상 수익률이 `signal_score`, 뉴스, 공시로 변경되지 않았는지 확인한다.

검산 수식:

```text
predicted_log_return ≈ ln(1 + predicted_return / 100)

predicted_return
≈ (exp(predicted_log_return) - 1) × 100

predicted_close
≈ Close × exp(predicted_log_return)
≈ Close × (1 + predicted_return / 100)
```

부동소수점 계산과 출력 반올림 때문에 아주 작은 차이는 발생할 수 있다.

---

## 11. 해석 시 주의사항

- `predicted_return = 2.5`는 다음 거래일 수익률을 `+2.5%`로 예상한다는 뜻이다.
- 이는 확정 수익률이나 보장 수익률이 아니다.
- `up_probability`와 `predicted_return`은 서로 다른 모델 출력이다.
- 실제 다음 거래일 수익률은 이후 종가가 확정된 뒤 `target_log_return`과 비교한다.
- 최종 매수·매도·관망 판단 기준은 저장소 정책상 다음 거래일 예상 수익률 `predicted_return`이다.

