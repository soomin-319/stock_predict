# Stock Predict (초기 실전안 구현)

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
stock-predict --input data/sample_ohlcv.csv --output predictions.csv
```

## 설치 없이 실행(저장소 루트에서만)
```bash
python -m src.pipeline --input data/sample_ohlcv.csv --output predictions.csv
```

## IDE/직접 파일 실행 (Windows 포함)
```bash
python src/pipeline.py --input data/sample_ohlcv.csv --output predictions.csv
```

위 실행도 지원하도록 `src/pipeline.py`에 경로 부트스트랩을 넣었습니다.

입력 CSV 필수 컬럼:
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- 선택: `Symbol`

## 출력 컬럼
- `predicted_return`
- `predicted_close`
- `up_probability`
- `uncertainty_width`, `uncertainty_band`
- `market_regime`
- `signal_score`, `signal_label`

## 참고
현재 모델 구현은 환경 의존성을 줄이기 위해 sklearn GBDT 기반으로 작성했습니다.
추후 LightGBM/XGBoost/GRU로 동일 인터페이스를 유지하면서 확장할 수 있습니다.


> `ModuleNotFoundError: No module named 'src'`가 발생하면 저장소 루트가 아닌 경로에서 실행한 경우가 많습니다.
> 이때는 `python -m pip install -e .` 후 `stock-predict ...`로 실행하세요.


## 출력 경로 주의사항 (Windows)
- Linux 스타일 경로(`/tmp/...`)를 Windows에서 넘기면 기본적으로 `\tmp`로 해석되어 실패할 수 있습니다.
- 현재 파이프라인은 Windows에서 `/tmp/...` 입력 시 자동으로 시스템 임시 폴더로 매핑합니다.
  - 예: `/tmp/predictions.csv` → `%TEMP%\predictions.csv`
- 일반적으로는 `predictions.csv` 또는 `output/predictions.csv` 같은 상대 경로를 권장합니다.
