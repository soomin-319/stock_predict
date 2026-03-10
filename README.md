# Stock Predict

다음날 로그수익률/상승확률/불확실성/신호점수를 제공하는 파이프라인입니다.

## 이번 개선 적용 내용
- **OOF(워크포워드) 기반 평가/튜닝/백테스트**로 과최적화 리스크 완화
- **전문가 기술지표 추가**: RSI, MACD/Signal/Histogram, ATR(14), Stochastic(K/D), CCI(20), OBV
- **외부 시장 feature 추가**: KOSPI/KOSDAQ/S&P500/NASDAQ/SOX/VIX/USDKRW/미국10년금리
- **사용자 시각화 아티팩트 자동 생성**:
  - `equity_curve.png`
  - `drawdown_curve.png`
  - `signal_score_hist.png`
  - `actual_vs_predicted_return.png` (실제/예측 수익률 비교)
  - `actual_vs_predicted_price.png` (실제/예측 다음 종가 비교)
  - `up_probability_calibration.png` (상승확률 보정 품질)
  - `uncertainty_vs_error.png` (불확실성 폭 vs 절대 오차)
  - `symbol_summary_table_top20.png` / `symbol_summary_table.csv` (종목별 요약표, 한글 폰트 fallback 적용)
  - `symbol_level/*.png` (각 종목별 실제/예측 가격 및 수익률 비교 그래프)
- OOF 예측 저장: `reports/oof_predictions.csv`

## 실제 데이터 가져와서 실행
```bash
python src/pipeline.py --fetch-real
```

## 권장 실행 예시
```bash
python src/pipeline.py \
  --fetch-real \
  --input data/real_ohlcv.csv \
  --output C:\Users\카운\Desktop\predictions_direct.csv \
  --report-json reports/pipeline_report.json \
  --figure-dir reports/figures
```

## 주요 산출물
- 최종 추론: `--output` CSV
- 리포트: `--report-json` JSON
- OOF 예측: `reports/oof_predictions.csv`
- 그래프: `--figure-dir` 하위 PNG들

## 입력 컬럼
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- 선택: `Symbol`

## 참고
- 모든 산출물 저장 경로는 입력값과 무관하게 프로젝트의 `result/` 폴더로 고정됩니다. 폴더가 없으면 자동 생성됩니다.


## 설치 및 테스트 예시
```bash
python -m pip install -r requirements.txt
pytest -q
```

## 실행 예시 (외부 지표 비활성화, 빠른 스모크)
```bash
python src/pipeline.py   --input data/sample_ohlcv.csv   --disable-external   --output /tmp/predictions_smoke.csv   --report-json /tmp/pipeline_report_smoke.json   --figure-dir /tmp/figures_smoke
```

## 실행 예시 (외부 지표 활성화)
```bash
python src/pipeline.py   --input data/real_ohlcv.csv   --output /tmp/predictions_real.csv   --report-json /tmp/pipeline_report_real.json   --figure-dir /tmp/figures_real
```

## 상세 기능 문서
- 프로젝트 전체 기능 요약: `docs/PROJECT_FEATURES_OVERVIEW.md`


## Week1 구현 반영 사항
- 모델 백엔드 고도화: `MultiHeadStockModel`이 LightGBM 사용 가능 시 LightGBM을 우선 사용하고, 미설치 환경에서는 sklearn GBDT로 자동 fallback합니다.
- 신호 튜닝/백테스트 분리: OOF 데이터를 시간순으로 분할해(기본 70:30) 앞 구간에서 가중치 튜닝, 뒤 구간(holdout)에서 백테스트를 수행하도록 개선했습니다.
- 리포트 확장: `tuning_samples`, `backtest_samples`를 JSON에 기록해 튜닝/평가 샘플 규모를 확인할 수 있습니다.


## Week2 구현 반영 사항
- 백테스트 현실화: `min_up_probability`, `min_signal_score` 필터를 통과한 종목만 Top-K에 포함되도록 개선했습니다.
- 백테스트 리포트 확장: `avg_turnover`, `avg_selected_count`를 추가해 포트폴리오 교체율/선정 종목 수를 확인할 수 있습니다.
- 외부 지표 가시성 강화: 리포트 JSON에 `external_feature_coverage`를 기록해 요청/성공/실패/fallback 사용 현황을 확인할 수 있습니다.


## 기본 산출물 저장 위치
- 기본값으로 아래 경로에 저장됩니다.
  - 예측 CSV: `C:\Users\카운\Desktop\result\predictions_direct.csv`
  - 리포트 JSON: `C:\Users\카운\Desktop\result\pipeline_report.json`
  - 그래프: `C:\Users\카운\Desktop\result\figures`

## 350개 유니버스 자동 구성
- `--fetch-real` 실행 시 `--real-symbols`를 생략하면 KOSPI 시가총액 상위 200 + KOSDAQ 상위 150을 자동 구성해(총 350개) 데이터를 수집합니다.


## 출력 해석
- `predicted_return`은 사용자 가독성을 위해 **퍼센트 수익률(%)**로 출력됩니다.
- 원래 모델의 로그수익률 값은 `predicted_log_return` 컬럼으로 함께 제공합니다.
