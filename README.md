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
- Windows에서 `/tmp/...` 경로는 `%TEMP%/...`로 자동 매핑됩니다.
