# Stock Predict

다음 거래일 예측을 위한 멀티헤드(수익률/상승확률/분위수) 주가 예측 파이프라인입니다.

## 핵심 기능
- 워크포워드 기반 OOF 검증/튜닝/백테스트
- 회귀 + 분류 + 분위수 예측(불확실성 포함)
- 외부 시장 지표(지수/환율/금리) feature
- 투자자 컨텍스트 feature(외국인·기관 수급, 공시, 뉴스 감성) 선택 연동
- 리포트/그래프/OOF CSV 자동 생성
- 콘솔 Top10 출력(방향정확도 중심)

## 설치
```bash
python -m pip install -r requirements.txt
```

## 빠른 실행
### 1) 샘플 데이터 스모크
```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --output predictions_smoke.csv --report-json pipeline_report_smoke.json --figure-dir figures_smoke
```

### 2) 실제 데이터(fetch + 외부 시장 feature)
```bash
python src/pipeline.py --fetch-real --input data/real_ohlcv.csv
```

> 참고: 일부 Windows/pykrx 환경에서 KRX 시총 API 스키마가 달라 자동 유니버스 생성이 실패할 수 있습니다.  
> 이 경우 코드가 자동으로 `--input`의 `Symbol` 컬럼을 fallback 사용하며, 필요 시 `--real-symbols`로 직접 심볼을 지정하세요.

### 3) 투자자 컨텍스트 연동(fetch-investor-context)
```powershell
python src/pipeline.py `
  --fetch-real `
  --fetch-investor-context `
  --dart-api-key "YOUR_DART_API_KEY" `
  --dart-corp-map-csv data/dart_corp_map.csv `
  --input data/real_ohlcv.csv `
  --output predictions_with_context.csv `
  --report-json pipeline_report_with_context.json `
  --figure-dir figures_with_context
```

### (참고) bash/zsh에서 줄바꿈 실행
```bash
python src/pipeline.py \
  --fetch-real \
  --fetch-investor-context \
  --dart-api-key "YOUR_DART_API_KEY" \
  --dart-corp-map-csv data/dart_corp_map.csv \
  --input data/real_ohlcv.csv \
  --output predictions_with_context.csv \
  --report-json pipeline_report_with_context.json \
  --figure-dir figures_with_context
```

## CLI 옵션 요약
- `--input`: 입력 OHLCV CSV 경로
- `--output`: 예측 CSV 경로(파일명 기준으로 `result/` 하위 저장)
- `--universe-csv`: 유니버스 CSV(`Symbol` 컬럼 필요)
- `--report-json`: 파이프라인 리포트 JSON 경로
- `--figure-dir`: 그래프 저장 디렉토리
- `--fetch-real`: yfinance로 실제 OHLCV 수집 후 실행
- `--real-symbols`: `--fetch-real` 시 대상 심볼 직접 지정
- `--real-start`: 실제 데이터 수집 시작일
- `--add-symbols`: 기존 입력 CSV에 사용자 심볼 추가 수집
- `--disable-external`: 외부 시장 지표 feature 비활성화
- `--fetch-investor-context`: 투자자 컨텍스트(수급/공시/뉴스) 연동 활성화
- `--dart-api-key`: OpenDART API Key
- `--dart-corp-map-csv`: `Symbol,corp_code` 매핑 CSV

## 입력 컬럼
### 필수
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

### 선택
- `Symbol`

### 투자자 컨텍스트 선택 입력(수동 제공 시 자동 반영)
- 외국인 순매수: `foreign_net_buy` / `외국인순매수` / `ForeignNetBuy`
- 기관 순매수: `institution_net_buy` / `기관순매수` / `InstitutionNetBuy`
- 공시 점수: `disclosure_score` / `공시점수` / `DisclosureScore`
- 뉴스 감성: `news_sentiment` / `뉴스점수` / `NewsSentiment`

## 주요 산출물
모든 출력은 **프로젝트 `result/` 폴더**에 저장됩니다.

- 예측 CSV (`--output` 파일명 기준)
- OOF 예측 CSV: `result/oof_predictions.csv`
- 리포트 JSON (`--report-json` 파일명 기준)
- 그래프 PNG (`--figure-dir` 디렉토리명 기준)

### 예측 CSV 주요 컬럼
- `predicted_log_return`, `predicted_return`, `up_probability`
- `uncertainty_width`, `uncertainty_score`, `signal_score`, `signal_label`
- `history_direction_accuracy`, `risk_flag`, `position_size_hint`
- 백테스트 요약 컬럼: `backtest_days`, `backtest_cum_return`, `backtest_sharpe` 등

### 리포트 JSON 주요 키
- `walk_forward`, `baselines`, `tuned_signal`, `backtest`
- `probability_calibration`(ECE/Brier)
- `external_feature_coverage`
- `investor_context_coverage`
- `artifacts`

## 그래프(대표)
- `equity_curve.png`, `drawdown_curve.png`
- `signal_score_hist.png`
- `actual_vs_predicted_return.png`
- `actual_vs_predicted_price.png`
- `up_probability_calibration.png`
- `uncertainty_vs_error.png`
- `symbol_level/*.png`, `symbol_level/recent_month/*.png`

## 테스트
```bash
pytest -q
```

## 참고 문서
- `docs/PROJECT_FEATURES_OVERVIEW.md`
- `docs/EXTERNAL_DATA_INTEGRATION_GUIDE.md`
- `docs/EXPERT_ANALYSIS_ROADMAP.md`
- `docs/PRIORITIZED_INVESTOR_ACTION_PLAN.md`
