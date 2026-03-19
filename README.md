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
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json --figure-dir figures_smoke
```

### 2) 실제 데이터(fetch + 외부 시장 feature)
```bash
python src/pipeline.py --fetch-real --input data/real_ohlcv.csv
```

> 참고: 자동 KRX 유니버스 생성은 비활성화되었습니다.  
> `--fetch-real` 사용 시 `--real-symbols` 또는 `--universe-csv`를 우선 사용하며, 둘 다 없으면 `--input`의 `Symbol` 컬럼으로 진행합니다.  
> `Symbol` 컬럼도 없을 때만 **소규모 내장 데모 유니버스(12종목)** 로 동작합니다. 실전용 기본 유니버스가 아니라 빠른 실행 확인용 fallback입니다.

### 2-1) 기존 입력 CSV에 종목 추가 수집(`--add-symbols`)
이미 가지고 있는 `data/real_ohlcv.csv`에 특정 종목만 더 붙이고 싶으면 `--add-symbols`를 사용합니다.  
이 옵션은 **기존 CSV를 유지한 채**, 입력한 종목들의 OHLCV만 추가로 수집해서 `Date` + `Symbol` 기준으로 병합합니다.

```powershell
python src/pipeline.py `
  --input data/real_ohlcv.csv `
  --add-symbols 005930 000660.KS 035420 `
  --real-start 2024-01-01
```

- `005930`처럼 6자리 숫자만 넣으면 내부에서 `.KS` / `.KQ` 심볼 형태로 정규화하려고 시도합니다.
- 쉼표로도 입력할 수 있어서 `--add-symbols 005930,000660,035420` 형태도 가능합니다.
- 추가 수집만 하고 바로 파이프라인까지 돌리려면 `--add-symbols`와 다른 실행 옵션(`--fetch-investor-context`, `--report-json` 등)을 함께 주면 됩니다.

### 2-2) 종목 추가 + 투자자 컨텍스트까지 한 번에 실행
```powershell
python src/pipeline.py `
  --input data/real_ohlcv.csv `
  --add-symbols 005930 000660 `
  --real-start 2024-01-01 `
  --fetch-investor-context `
  --dart-api-key "YOUR_DART_API_KEY" `
  --dart-corp-map-csv data/dart_corp_map.csv `
  --report-json pipeline_report_added_symbols.json `
  --figure-dir figures_added_symbols
```

### (참고) bash/zsh에서 종목 추가 실행
```bash
python src/pipeline.py \
  --input data/real_ohlcv.csv \
  --add-symbols 005930 000660.KS 035420 \
  --real-start 2024-01-01
```

### 3) 투자자 컨텍스트 연동(fetch-investor-context)
```powershell
python src/pipeline.py `
  --fetch-real `
  --fetch-investor-context `
  --dart-api-key "YOUR_DART_API_KEY" `
  --dart-corp-map-csv data/dart_corp_map.csv `
  --input data/real_ohlcv.csv `
  --report-json pipeline_report_with_context.json `
  --figure-dir figures_with_context
```

> 참고: 뉴스 점수화는 기본적으로 규칙 기반 fallback을 유지하면서, `OPENAI_API_KEY`와 `OPENAI_MODEL` 환경변수가 있으면 AI 기반 제목 점수화를 우선 시도합니다.  
> 예: `NEWS_SCORING_MODE=ai` 또는 `NEWS_SCORING_MODE=auto`

### 3-1) 투자자 컨텍스트는 유지하고 뉴스만 끄기
```bash
python src/pipeline.py \
  --fetch-real \
  --fetch-investor-context \
  --disable-news-context \
  --dart-api-key "YOUR_DART_API_KEY" \
  --dart-corp-map-csv data/dart_corp_map.csv \
  --input data/real_ohlcv.csv \
  --report-json pipeline_report_without_news.json \
  --figure-dir figures_without_news
```

이 조합은 **수급/공시 컨텍스트는 유지**하면서 **뉴스 수집/뉴스 점수화만 비활성화**합니다.

### (참고) bash/zsh에서 줄바꿈 실행
```bash
python src/pipeline.py \
  --fetch-real \
  --fetch-investor-context \
  --dart-api-key "YOUR_DART_API_KEY" \
  --dart-corp-map-csv data/dart_corp_map.csv \
  --input data/real_ohlcv.csv \
  --report-json pipeline_report_with_context.json \
  --figure-dir figures_with_context
```

## CLI 옵션 요약
- `--input`: 입력 OHLCV CSV 경로
- `--output`: 레거시 옵션(실제 CSV는 항상 `result/result_detail.csv`, `result/result_simple.csv`로 저장)
- `--universe-csv`: 유니버스 CSV(`Symbol` 컬럼 필요)
- `--report-json`: 파이프라인 리포트 JSON 경로
- `--figure-dir`: 그래프 저장 디렉토리
- `--fetch-real`: yfinance로 실제 OHLCV 수집 후 실행
- `--real-symbols`: `--fetch-real` 시 대상 심볼 직접 지정
- `--real-start`: 실제 데이터 수집 시작일
- `--add-symbols`: 기존 입력 CSV에 사용자 심볼 추가 수집
- `--disable-external`: 외부 시장 지표 feature 비활성화
- `--fetch-investor-context`: 투자자 컨텍스트(수급/공시/뉴스) 연동 활성화
- `--disable-news-context`: 투자자 컨텍스트 중 뉴스 수집/점수화만 비활성화
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
- 개인 순매수: `individual_net_buy` / `개인순매수` / `PersonalNetBuy`
- 외국인 보유비중: `foreign_ownership_ratio` / `외국인보유비중` / `ForeignOwnershipRatio`
- 프로그램 매매: `program_trading_flow` / `프로그램순매수` / `ProgramTradingFlow`
- 공시 점수: `disclosure_score` / `공시점수` / `DisclosureScore`
- 뉴스 감성: `news_sentiment` / `뉴스점수` / `NewsSentiment`
- 뉴스 관련도/영향/건수: `news_relevance_score` / `뉴스관련도` / `NewsRelevanceScore`, `news_impact_score` / `뉴스영향점수` / `NewsImpactScore`, `news_article_count` / `뉴스건수` / `NewsArticleCount`

### 한국 시장 구조/이벤트 선택 입력(있을 때만 사용)
- 시장구분: `market_type` / `시장구분` / `MarketType` (`KOSPI`, `KOSDAQ`, `KONEX`)
- 거래소: `venue` / `거래소` / `Venue` (`KRX`, `NXT`)
- 세션: `session` / `세션` / `Session` (`정규장`, `프리마켓`, `애프터마켓`, `시간외`)
- 상장일/상장후일수: `listing_date` / `상장일` / `ListingDate`, `days_since_listing` / `상장후일수` / `DaysSinceListing`
- 시장경보: `warning_level` / `시장경보` / `투자경보단계` / `WarningLevel`
- 거래정지: `halt_flag` / `거래정지` / `HaltFlag`
- VI: `vi_flag` / `VI발동` / `VIFlag`, `vi_count` / `VI횟수` / `VICount`
- 단기과열: `short_term_overheat_flag` / `단기과열종목` / `ShortTermOverheatFlag`
- 공매도: `short_sell_flag` / `공매도가능` / `ShortSellFlag`, `short_sell_balance` / `공매도잔고` / `ShortSellBalance`, `short_sell_ratio` / `공매도비중` / `ShortSellRatio`, `short_sell_overheat_flag` / `공매도과열종목` / `ShortSellOverheatFlag`
- 가치/주주환원: `pbr` / `PBR`, `per` / `PER`, `roe` / `ROE`, `dividend_yield` / `배당수익률` / `DividendYield`, `buyback_flag` / `자사주취득` / `BuybackFlag`, `share_cancellation_flag` / `자사주소각` / `ShareCancellationFlag`, `value_up_disclosure_flag` / `밸류업공시` / `ValueUpDisclosureFlag`

## 주요 산출물
모든 출력은 **프로젝트 `result/` 폴더**에 저장됩니다.

- 상세 CSV: `result/result_detail.csv` (예측값 + 최신 feature 값 전체)
- 사용자용 요약 CSV: `result/result_simple.csv` (종목코드/이름/권고/예상 종가/예상 수익률/신뢰도/예측 이유)
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
