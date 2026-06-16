# 신호 가중치 역산 실무 가이드

## 1. 목적

이 문서는 최종 `signal_score`에서 출발해 점수를 구성한 이벤트 보정값, 모델 기반 기본 점수, 각 입력값의 기여도를 역순으로 확인하는 방법을 설명한다.

역산은 다음 질문에 답할 때 유용하다.

- 최종 점수가 왜 높거나 낮은가?
- 이벤트 조건이 점수를 얼마나 변경했는가?
- 예상수익률 순위, 상승확률, 상대강도, 불확실성 중 어떤 항목의 영향이 컸는가?
- 입력값 하나가 없을 때 나머지 값으로 복원할 수 있는가?

> 중요: `signal_score`는 보조 평가 점수다. 매수·매도·관망 결정과 백테스트의 우선순위는 다음 날 `predicted_return`을 기준으로 한다. 뉴스와 공시는 표시·검토용이며 점수와 의사결정을 변경하지 않는다.

---

## 2. 전체 계산식을 먼저 확인한다

최종 신호 점수는 기본 점수와 이벤트 보정값의 합이다.

```text
최종 신호 점수 = 기본 신호 점수 + 이벤트 보정값
```

변수로 표현하면 다음과 같다.

```text
S_final = S_base + E
```

기본 신호 점수 계산식:

```text
S_base =
    w_return × N
  + w_prob   × P
  + w_rel    × R
  - w_uncert × U
```

| 기호 | 실제 컬럼 또는 의미 |
|---|---|
| `S_final` | 이벤트 보정 후 최종 `signal_score` |
| `S_base` | 이벤트 보정 전 기본 신호 점수 |
| `E` | `event_boost_score` |
| `N` | `norm_return`, 예상 로그수익률의 종목 간 백분위 순위 |
| `P` | `up_probability`, 상승 확률 |
| `R` | `rel_strength`, 상대강도 |
| `U` | `uncertainty_score`, 불확실성 폭의 종목 간 백분위 순위 |
| `w_return` | `return_weight` |
| `w_prob` | `up_prob_weight` |
| `w_rel` | `rel_strength_weight` |
| `w_uncert` | `uncertainty_penalty` |

구현 위치:

- 기본 점수: `src/inference/predict.py`
- 가중치 튜닝: `src/validation/signal_tuning.py`
- 이벤트 보정: `src/domain/signal_policy.py`
- 튜닝 가중치 재적용: `src/pipeline.py`

---

## 3. 1단계: 최종 점수에서 이벤트 보정값을 제거한다

최종 점수와 이벤트 보정값을 알고 있다면 기본 점수는 바로 복원된다.

```text
S_base = S_final - E
```

예:

```text
최종 signal_score = 0.685
event_boost_score = 0.090

기본 signal_score = 0.685 - 0.090
                  = 0.595
```

이벤트 보정값은 여러 조건의 합이다. 따라서 `E`를 역산한 뒤 어떤 이벤트가 합산됐는지 원본 컬럼으로 확인해야 한다.

주요 이벤트 보정값:

| 조건 | 점수 변화 |
|---|---:|
| 거래대금 순위 3위 이내 | `+0.05` |
| 설정된 거래대금 상위 기준 충족 | `+0.04` |
| 외국인·기관 동시 순매수 | `+0.04` |
| 외국인·기관 각각 강한 순매수 | `+0.06` |
| 거래대금 상위이며 강한 동시 순매수 | `+0.08` |
| 주도주 확인 | `+0.05` |
| 52주 신고가 근처 또는 돌파 | `+0.03` |
| RSI 매수 관찰 구간 | `+0.02` |
| RSI 과매수 구간 | `-0.08` |
| 나스닥 선물 상승 | `+0.03` |
| 나스닥 선물 강한 상승 | 추가 `+0.06` |
| 나스닥 선물 강한 하락 | `-0.12` |

일부 조건은 동시에 성립한다. 예를 들어 나스닥 선물이 강하게 상승하면 일반 상승 보정 `+0.03`과 강한 상승 보정 `+0.06`이 함께 적용되어 총 `+0.09`가 된다.

---

## 4. 2단계: 기본 점수를 항목별 기여도로 분해한다

각 항목의 점수 기여도를 따로 계산한다.

```text
수익률 기여도   = w_return × N
상승확률 기여도 = w_prob × P
상대강도 기여도 = w_rel × R
불확실성 차감액 = w_uncert × U
```

검산식:

```text
S_base =
    수익률 기여도
  + 상승확률 기여도
  + 상대강도 기여도
  - 불확실성 차감액
```

예제 입력:

```text
w_return = 0.45
w_prob   = 0.30
w_rel    = 0.25
w_uncert = 0.25

N = 0.80
P = 0.70
R = 0.30
U = 0.20
```

기여도:

```text
수익률 기여도   = 0.45 × 0.80 = 0.360
상승확률 기여도 = 0.30 × 0.70 = 0.210
상대강도 기여도 = 0.25 × 0.30 = 0.075
불확실성 차감액 = 0.25 × 0.20 = 0.050

S_base = 0.360 + 0.210 + 0.075 - 0.050
       = 0.595
```

이벤트 보정값이 `0.090`이라면:

```text
S_final = 0.595 + 0.090
        = 0.685
```

---

## 5. 3단계: 상대강도와 수익률 순위의 관계를 이용한다

현재 구현에서 `norm_return`과 `rel_strength`는 같은 예상 로그수익률 백분위 순위로 계산된다.

```text
R = N - 0.5
```

따라서 두 값은 독립 변수가 아니다. 기본 점수식을 다음처럼 단순화할 수 있다.

```text
S_base =
    w_return × N
  + w_prob × P
  + w_rel × (N - 0.5)
  - w_uncert × U
```

정리:

```text
S_base =
    (w_return + w_rel) × N
  + w_prob × P
  - 0.5 × w_rel
  - w_uncert × U
```

이 식을 사용하면 `N` 또는 `R` 중 하나만 알아도 둘 다 복원할 수 있다.

```text
R = N - 0.5
N = R + 0.5
```

---

## 6. 4단계: 누락된 입력값 하나를 역산한다

가중치와 나머지 입력값을 알고 있을 때 누락된 입력값 하나를 복원할 수 있다.

### 6.1 예상수익률 순위 `N` 역산

`R = N - 0.5` 관계를 반영한 권장 역산식:

```text
N =
  (
      S_base
    - w_prob × P
    + 0.5 × w_rel
    + w_uncert × U
  )
  / (w_return + w_rel)
```

앞선 예제 역산:

```text
N =
  (0.595 - 0.30×0.70 + 0.5×0.25 + 0.25×0.20)
  / (0.45 + 0.25)

N = (0.595 - 0.210 + 0.125 + 0.050) / 0.70
  = 0.560 / 0.70
  = 0.80

R = 0.80 - 0.50
  = 0.30
```

### 6.2 상승확률 `P` 역산

```text
P =
  (
      S_base
    - w_return × N
    - w_rel × R
    + w_uncert × U
  )
  / w_prob
```

### 6.3 불확실성 점수 `U` 역산

```text
U =
  (
      w_return × N
    + w_prob × P
    + w_rel × R
    - S_base
  )
  / w_uncert
```

### 6.4 이벤트 보정값 `E` 역산

기본 입력값과 최종 점수를 알고 있다면:

```text
E = S_final - S_base
```

역산된 `E`는 이벤트 조건들의 합만 보여준다. 같은 합계를 만드는 이벤트 조합이 여러 개일 수 있으므로 `E`만으로 개별 이벤트를 확정할 수 없다.

---

## 7. 튜닝된 가중치를 역으로 이해하는 방법

초기 기본 가중치:

```text
w_return = 0.45
w_prob   = 0.35
w_rel    = 0.20
w_uncert = 0.25
```

파이프라인 실행 중에는 OOF 튜닝 데이터에서 다음 후보를 탐색한다.

```text
w_return ∈ {0.30, 0.45, 0.60}
w_prob   = 0.30
w_rel    = 1.00 - w_return - w_prob
w_uncert ∈ {0.15, 0.25, 0.35}
```

가능한 수익률·상대강도 조합:

| `w_return` | `w_prob` | `w_rel` | `w_return + w_rel` |
|---:|---:|---:|---:|
| 0.30 | 0.30 | 0.40 | 0.70 |
| 0.45 | 0.30 | 0.25 | 0.70 |
| 0.60 | 0.30 | 0.10 | 0.70 |

튜닝 후에는 항상 다음 관계가 성립한다.

```text
w_return + w_rel = 0.70
w_prob = 0.30
```

따라서 튜닝 후 기본 점수는 다음처럼 볼 수 있다.

```text
S_base =
    0.70 × N
  + 0.30 × P
  - 0.5 × w_rel
  - w_uncert × U
```

`w_return`이 커질수록 `w_rel`은 작아진다. 그러나 `N`과 `R`이 같은 백분위에서 파생되므로 예상수익률 관련 총 기울기 `0.70`은 유지된다. 실제 차이는 상대강도 항의 상수 조정값인 `-0.5 × w_rel`에서 발생한다.

튜닝은 각 후보 점수로 상위 10% 종목을 고른 뒤, 해당 종목들의 실제 다음 날 평균 로그수익률이 가장 높은 조합을 선택한다.

---

## 8. 결과 행 하나를 역산하는 실무 절차

결과 CSV 또는 DataFrame의 종목 한 행을 다음 순서로 확인한다.

### 절차 1: 최종값 확보

필수 확인값:

```text
signal_score
event_boost_score
norm_return
up_probability
rel_strength
uncertainty_score
```

사용된 가중치도 파이프라인 보고서 또는 실행 설정에서 확인한다.

### 절차 2: 이벤트 효과 제거

```text
S_base = signal_score - event_boost_score
```

### 절차 3: 각 항목 기여도 계산

```text
return_contribution      = w_return × norm_return
probability_contribution = w_prob × up_probability
relative_contribution    = w_rel × rel_strength
uncertainty_deduction    = w_uncert × uncertainty_score
```

### 절차 4: 합계 검산

```text
S_base
= return_contribution
 + probability_contribution
 + relative_contribution
 - uncertainty_deduction
```

부동소수점 처리와 결과 CSV 반올림 때문에 작은 차이가 발생할 수 있다.

### 절차 5: 이벤트 조건 확인

`event_boost_score`와 다음 원본 컬럼을 비교한다.

```text
turnover_rank_daily
foreign_net_buy
institution_net_buy
leader_confirmation_flag
near_52w_high_flag
breakout_52w_flag
rsi_14
nq_f_ret_1d
```

### 절차 6: 의사결정과 분리해 해석

높은 `signal_score`가 매수를 직접 의미하지 않는다.

```text
predicted_return > 2.0%    → 매수
predicted_return <= -2.0%  → 매도
그 외                     → 관망
```

백테스트 종목 선정도 `predicted_return`을 우선 정렬하고 `signal_score`는 보조 정렬값으로 사용한다.

---

## 9. 역산 가능 범위와 주의사항

### 정확히 역산 가능한 경우

- 최종 점수와 이벤트 보정값으로 기본 점수 복원
- 가중치와 기본 입력값으로 각 항목 기여도 복원
- 가중치와 나머지 입력값이 있을 때 누락 입력값 하나 복원
- `norm_return`으로 `rel_strength` 복원 또는 반대 방향 복원

### 단독으로는 역산 불가능한 경우

- 최종 점수 하나만으로 모든 입력값 복원
- 이벤트 보정 총합만으로 개별 이벤트 조합 확정
- 백분위 점수만으로 원래 `predicted_log_return` 값 정확히 복원
- 점수만으로 어떤 튜닝 후보가 선택됐는지 확정

백분위 점수는 종목 간 순위를 나타낸다. 원래 예상수익률 크기를 복원하려면 같은 평가 시점의 전체 종목별 `predicted_log_return` 분포가 필요하다.

---

## 10. 빠른 역산 체크리스트

```text
1. 최종 signal_score 확인
2. event_boost_score 차감
3. 실행에 사용된 튜닝 가중치 확인
4. 기본 입력값별 가중 기여도 계산
5. 기여도 합계와 기본 점수 비교
6. 이벤트 원본 컬럼으로 보정값 구성 확인
7. predicted_return 기반 추천 결과와 별도로 해석
```

