# Stock Predict (초기 실전안 구현)

다음날 로그수익률/상승확률/불확실성/신호점수를 제공하는 베이스라인 파이프라인입니다.

## 구현된 4단계
1. **데이터 실전화**: OHLCV 정제 + 유니버스 필터(KOSPI200+KOSDAQ150 기준)
2. **기준선 비교**: walk-forward + naive baseline(0수익률/전일수익률)
3. **신호 튜닝**: signal weight grid-search(top decile 수익률 최적화)
4. **백테스트**: 비용(수수료+슬리피지) 반영 long-only top-k

## 실제 데이터 가져와서 바로 실행
```bash
python src/pipeline.py --fetch-real
```
- 기본 실데이터 티커: `005930.KS`, `000660.KS`, `035420.KS`, `051910.KS`, `207940.KS`
- 기본 저장 경로: `data/real_ohlcv.csv`
- 기본 예측 출력 경로: `C:\Users\카운\Desktop\predictions_direct.csv`
설계 문서의 핵심 권장사항(다음날 로그수익률 + 방향분류 + 불확실성 구간 + 신호 점수)을 반영한 **초기 베이스라인 코드**입니다.

## 포함 기능
- 다음날 로그수익률 타깃 생성
- 상승/하락 분류 타깃 생성
- 기술적 파생 특징 생성(수익률, 이동평균, 변동성, RSI, 거래량 비율)
- 멀티헤드 모델
  - 회귀(head 1)
  - 분류(head 2)
  - 분위수 회귀 기반 불확실성(head 3)
- Walk-forward 검증
- 최종 출력: `predicted_return`, `predicted_close`, `up_probability`, `uncertainty_band`, `signal_score`

## 설치 후 실행(권장)
```bash
python -m pip install -e .
stock-predict --input data/real_ohlcv.csv --output C:\Users\카운\Desktop\predictions_direct.csv
```

## 직접 실행
```bash
python src/pipeline.py --input data/real_ohlcv.csv --output C:\Users\카운\Desktop\predictions_direct.csv
```

## 실제 운용 권장 실행 예시
```bash
python src/pipeline.py \
  --fetch-real \
  --input data/real_ohlcv.csv \
  --universe-csv data/your_universe.csv \
  --output C:\Users\카운\Desktop\predictions_direct.csv \
  --input data/your_ohlcv.csv \
  --universe-csv data/your_universe.csv \
  --output output/predictions.csv \
  --report-json reports/pipeline_report.json
```

- `--universe-csv`: `Symbol` 컬럼 필수
  - 실전에서는 KOSPI200+KOSDAQ150 구성 종목을 넣어 고정 유니버스로 운용 권장
- `--report-json`: 단계 2~4 결과(검증/기준선/튜닝/백테스트) 저장

## 콘솔 출력
실행 시 아래가 콘솔에 함께 출력됩니다.
- `Pipeline summary` (검증/기준선/튜닝/백테스트 요약)
- `Top predictions` (상위 신호 종목 테이블)

## 입력 CSV 필수 컬럼
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- 선택: `Symbol`

## 출력
### predictions.csv
- `predicted_return`, `predicted_close`, `up_probability`
- `uncertainty_width`, `uncertainty_band`
- `market_regime`, `signal_score`, `signal_label`

### pipeline_report.json
- `walk_forward`
- `baselines`
- `tuned_signal`
- `backtest`

## Windows 출력 경로
- `/tmp/...` 경로는 Windows에서 `%TEMP%/...`로 자동 매핑됩니다.
