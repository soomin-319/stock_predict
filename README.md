# Stock Predict

다음 거래일 예측을 위한 멀티헤드(수익률/상승확률/분위수) 주가 예측 파이프라인입니다.

## 핵심 기능
- 워크포워드 기반 OOF 검증/튜닝/백테스트
- 회귀 + 분류 + 분위수 예측(불확실성 포함)
- 외부 시장 지표(지수/환율/금리) feature
- 투자자 수급 컨텍스트 feature(외국인·기관 수급) 선택 연동
- 설정 JSON 기반 실행 모드 분리(research / conservative production 예시 포함)
- 리포트/그래프/OOF CSV 자동 생성
- 콘솔 Top10 출력(방향정확도 중심)
- 포트폴리오 액션/거래 게이트/리스크 플래그 기반 PM 요약

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
> `--fetch-real` 사용 시 `--real-symbols` 또는 `--universe-csv`를 우선 사용합니다.  
> 둘 다 없으면 저장소에 포함된 기본 유니버스 CSV(`data/default_universe_kospi50_kosdaq50.csv`)의 100개 심볼(KOSPI 50 + KOSDAQ 50 스타일)로 자동 수집을 진행합니다.  
> 코랩 `run_colab_pipeline()`의 첫 실행도 같은 원칙을 따르며, `data/sample_ohlcv.csv` 같은 데모 입력이 감지되면 먼저 같은 기본 100종목을 `data/real_ohlcv.csv`로 수집한 뒤 예측을 진행합니다.

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

### 3) 투자자 수급 컨텍스트 연동(fetch-investor-context)
```powershell
python src/pipeline.py `
  --fetch-real `
  --fetch-investor-context `
  --input data/real_ohlcv.csv `
  --report-json pipeline_report_with_context.json `
  --figure-dir figures_with_context
```

### (참고) bash/zsh에서 줄바꿈 실행
```bash
python src/pipeline.py \
  --fetch-real \
  --fetch-investor-context \
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
- `--fetch-investor-context`: 투자자 수급 컨텍스트(외국인/기관 순매수) 연동 활성화
- `--disable-investor-flow`: 투자자 수급(pykrx) 비활성화
- `--disable-disclosure-context`: 공시 컨텍스트 비활성화
- `--disable-news-context`: 뉴스 컨텍스트 비활성화
- `--news-scoring-mode`: `auto` / `rule` / `ai`
- `--openai-api-key`, `--openai-model`: AI 뉴스 점수화 옵션
- `--config-json`: `configs/*.json` 형식의 AppConfig 오버라이드
- `--min-value-traded`: 백테스트/리포트용 최소 거래대금 필터
- `--turnover-limit`: 백테스트 turnover limit override
- `--min-up-probability`, `--min-signal-score`: 백테스트 필터 override
- `--min-external-coverage-ratio`: 외부 지표 커버리지 최소 비율 override
- `--min-investor-coverage-ratio`: 투자자 컨텍스트 커버리지 최소 비율 override
- `--portfolio-value`: 백테스트 포트폴리오 총 운용금액(유동성/참여율 체크)
- `--max-daily-participation`: 종목별 하루 거래대금 대비 최대 참여율
- `--max-positions-per-market-type`: `market_type` 버킷별 최대 편입 종목 수
- `--dart-api-key`: 레거시 옵션(현재 사용하지 않음)
- `--dart-corp-map-csv`: 레거시 옵션(현재 사용하지 않음)

기본 출력 파일명은 `result_detail.csv`, `pipeline_report.json`, `figures/`이며 실제 저장 위치는 항상 프로젝트의 `result/` 아래로 정규화됩니다.

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
- 사용자용 요약 CSV: `result/result_simple.csv` (종목코드/이름/권고/예상 종가/예상 수익률/상승확률/신뢰도/예측 이유)
- CSV는 한글 깨짐을 줄이기 위해 `UTF-8 BOM(utf-8-sig)`으로 저장합니다. Windows Excel/Colab에서 바로 열어도 한글이 최대한 유지되도록 맞춰두었습니다.
- 리포트 JSON (`--report-json` 파일명 기준)
- 그래프 PNG (`--figure-dir` 디렉토리명 기준)

### 예측 CSV 주요 컬럼
- `predicted_log_return`, `predicted_return`, `up_probability`
- `uncertainty_width`, `uncertainty_score`, `signal_score`, `signal_label`
- `predicted_return_5d`, `predicted_return_20d`
- `up_probability_5d`, `up_probability_20d`
- `confidence_score`, `confidence_label`
- `history_direction_accuracy`, `risk_flag`, `position_size_hint`
- `portfolio_action`, `trading_gate`
- 백테스트 요약 컬럼: `backtest_days`, `backtest_cum_return`, `backtest_sharpe` 등

### 리포트 JSON 주요 키
- `walk_forward`, `baselines`, `tuned_signal`, `backtest`
- `probability_calibration`(ECE/Brier)
- `external_feature_coverage`
- `investor_context_coverage`
- `coverage_gate`
- `pm_summary`
- `artifacts.pm_report_json`
- `config`
- `artifacts`

## 설정 프로필 예시
```bash
python src/pipeline.py \
  --input data/real_ohlcv.csv \
  --config-json configs/prod_conservative.json \
  --fetch-investor-context \
  --news-scoring-mode rule
```

- `configs/research_balanced.json`: 연구/검증용 기본형
- `configs/prod_conservative.json`: 더 보수적인 거래대금·회전율·비용 가정

## 그래프(대표)
- `equity_curve.png`, `drawdown_curve.png`
- `signal_score_hist.png`
- `actual_vs_predicted_return.png`
- `actual_vs_predicted_price.png`
- `up_probability_calibration.png`
- `uncertainty_vs_error.png`
- `symbol_level/*.png`, `symbol_level/recent_month/*.png`


## 카카오톡 챗봇 + 코랩 연동
`src/chatbot/kakao_colab_bot.py`는 **카카오톡 챗봇 웹훅**에서 바로 사용할 수 있는 Flask 엔드포인트와, 종목코드별 예측 캐시/비동기 실행 로직을 제공합니다.

### 동작 방식
1. 사용자가 카카오톡 챗봇에 `005930` 같은 종목코드를 보냅니다.
   - 종목명(예: `삼성전자`)을 보내도 exact match면 바로 예측하고, 비슷한 이름이면 후보를 먼저 제안합니다.
2. 웹훅은 카카오 payload의 `userRequest.user.id`를 읽어 사용자별 마지막 조회 종목을 기억합니다.
3. `result/result_simple.csv`에 해당 종목의 예측 결과가 있으면, 챗봇이 아래 항목을 바로 응답합니다.
   - 종목명
   - 권고
   - 내일 예측 수익률
   - 내일 예측 종가
   - 신뢰도
   - 사유
4. 아직 예측 결과가 없으면, 챗봇은 `005930 예측을 시작합니다` 메시지와 함께 `결과 확인`, `최신화`, `도움말` quick reply를 보여줍니다.
5. 사용자가 이어서 `결과`를 보내면 방금 요청한 종목의 진행상태/완료결과를 확인하고, `최신화`를 보내면 같은 종목으로 재예측을 시작합니다.
6. 서버는 백그라운드에서 아래 형태의 파이프라인 명령을 실행합니다.

```bash
python src/pipeline.py \
  --input data/real_ohlcv.csv \
  --add-symbols 005930 \
  --fetch-investor-context \
  --report-json pipeline_report_with_context.json \
  --figure-dir figures_with_context
```

> 기존 예시 명령은 `--fetch-real` 중심이었는데, 챗봇 입력 기반 단일 종목 추가 시에는 기존 CSV를 유지하면서 대상 종목만 붙일 수 있도록 `--add-symbols` 기반으로 구현했습니다.


### 실제 카카오톡 대화 예시
- 사용자: `005930`
- 챗봇: `005930 예측을 시작합니다...`
- 사용자: `삼성전`
- 챗봇: `비슷한 종목입니다: 삼성전자(005930), 삼성전자우(005935) ...`
- 사용자: `결과`
- 챗봇: 진행 중이면 진행 상태를, 완료 후에는 `종목명/권고/내일 예측 수익률/내일 예측 종가/신뢰도/사유`를 응답
- 사용자: `최신화`
- 챗봇: 같은 종목으로 최신 예측 재실행

### pyngrok으로 코랩 공개 HTTPS URL 열기
코랩에서는 `launch_colab_kakao_bot(...)`를 사용하면 Flask 서버 스레드 실행 + pyngrok 공개 HTTPS URL 생성까지 한 번에 처리할 수 있습니다. 이때 기본값으로 서버 시작 전에 기본 심볼 유니버스 예측을 한 번 미리 돌려 `result/result_simple.csv` 캐시를 채워두므로, 이후 사용자 요청에 더 빠르게 응답할 수 있습니다.

```python
import os
from pyngrok import ngrok
from src.chatbot.kakao_colab_bot import (
    PipelineRuntimeConfig,
    PyngrokTunnelConfig,
    launch_colab_kakao_bot,
)

# 필수 키
os.environ["DART_API_KEY"] = "YOUR_DART_API_KEY"
os.environ["NGROK_AUTHTOKEN"] = "YOUR_NGROK_AUTHTOKEN"

# 선택: 공시/뉴스 LLM 해석 기능(OpenAI)
# - 설정하면 공시/뉴스 요약 품질 개선에 사용됩니다.
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

runtime_config = PipelineRuntimeConfig(
    dart_api_key=os.environ["DART_API_KEY"],
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_model="gpt-4o-mini",
    input_csv="data/real_ohlcv.csv",
    report_json="pipeline_report_with_context.json",
    figure_dir="figures_with_context",
)

tunnel = launch_colab_kakao_bot(
    runtime_config=runtime_config,
    tunnel_config=PyngrokTunnelConfig(
        port=8000,
        auth_token=os.environ.get("NGROK_AUTHTOKEN"),
    ),
)

print(tunnel["public_url"])
print(tunnel["webhook_url"])
print(ngrok.get_tunnels())
```

카카오 오픈빌더 스킬 서버 URL에는 출력된 `webhook_url` 값을 그대로 입력하면 됩니다.
예측 작업이 시작되면 코랩 콘솔에는 `[KAKAO BOT ...]` 접두어로 시작 메시지와 최종 예측 결과만 출력되고, 상세 진행 로그는 `result/chatbot_logs/` 파일에 저장됩니다.

### 코랩에서 웹훅 서버 실행
```python
import os
import time
import threading

from flask import request
from openai import OpenAI
from pyngrok import ngrok
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.chatbot.kakao_colab_bot import (
    PipelineRuntimeConfig,
    PyngrokTunnelConfig,
    create_app,
    start_pyngrok_tunnel,
)

# =========================
# 1) 환경변수 설정
# =========================
os.environ["DART_API_KEY"] = "YOUR_DART_API_KEY"
os.environ["NGROK_AUTHTOKEN"] = "YOUR_NGROK_AUTHTOKEN"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # 공시/뉴스 LLM 요약 사용 시(선택)
os.environ["NAVER_CLIENT_ID"] = "YOUR_NAVER_CLIENT_ID"  # 네이버 뉴스 API(선택)
os.environ["NAVER_CLIENT_SECRET"] = "YOUR_NAVER_CLIENT_SECRET"  # 네이버 뉴스 API(선택)

PORT = 8000
OPENAI_MODEL = "gpt-4o-mini"
TEST_QUERY = "삼성전자"


def check_openai_connection(api_key: str | None, model: str) -> bool:
    if not api_key:
        print("[OPENAI CHECK] OPENAI_API_KEY가 비어 있어 연결 확인을 건너뜁니다.")
        return False
    client = OpenAI(api_key=api_key)
    try:
        resp = client.responses.create(
            model=model,
            input="ping",
            max_output_tokens=16,
        )
        print("[OPENAI CHECK] 연결 성공")
        print(f"[OPENAI CHECK] model={model}")
        print(f"[OPENAI CHECK] response_id={resp.id}")
        return True
    except Exception as exc:
        print("[OPENAI CHECK] 연결 실패")
        print(f"[OPENAI CHECK] error={type(exc).__name__}: {exc}")
        return False


def check_naver_news_connection(client_id: str | None, client_secret: str | None, query: str) -> bool:
    if not client_id or not client_secret:
        print("[NAVER CHECK] NAVER_CLIENT_ID/NAVER_CLIENT_SECRET가 비어 있어 연결 확인을 건너뜁니다.")
        return False
    params = urlencode({"query": query, "display": 5, "start": 1, "sort": "date"})
    req = Request(
        f"https://openapi.naver.com/v1/search/news.json?{params}",
        headers={
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        },
    )
    try:
        with urlopen(req, timeout=15) as resp:
            import json

            payload = json.loads(resp.read().decode("utf-8"))
        items = payload.get("items", []) if isinstance(payload, dict) else []
        print(f"[NAVER CHECK] query={query} | items={len(items)}")
        if items:
            print(f"[NAVER CHECK] sample_title={items[0].get('title', '')[:100]}")
        return True
    except Exception as exc:
        print("[NAVER CHECK] 연결 실패")
        print(f"[NAVER CHECK] error={type(exc).__name__}: {exc}")
        return False


check_openai_connection(os.environ.get("OPENAI_API_KEY"), OPENAI_MODEL)
check_naver_news_connection(
    os.environ.get("NAVER_CLIENT_ID"),
    os.environ.get("NAVER_CLIENT_SECRET"),
    TEST_QUERY,
)

runtime_config = PipelineRuntimeConfig(
    input_csv="data/real_ohlcv.csv",
    report_json="pipeline_report_with_context.json",
    figure_dir="figures_with_context",
    dart_api_key=os.environ["DART_API_KEY"],
    dart_corp_map_csv="data/dart_corp_map.csv",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_model=OPENAI_MODEL,
    naver_client_id=os.environ.get("NAVER_CLIENT_ID"),
    naver_client_secret=os.environ.get("NAVER_CLIENT_SECRET"),
)

# =========================
# 2) Flask 앱 생성
# =========================
app = create_app(runtime_config=runtime_config)

# 카카오에서 들어오는 메시지를 코랩 출력에 같이 보여주기 위한 로그
@app.before_request
def log_kakao_message():
    if request.path == "/kakao/webhook" and request.method == "POST":
        payload = request.get_json(silent=True) or {}
        user_request = payload.get("userRequest") or {}
        utterance = (user_request.get("utterance") or "").strip()
        user = user_request.get("user") or {}
        user_id = user.get("id") or user.get("userKey") or "anonymous"
        print(f"[KAKAO MESSAGE] user_id={user_id} | utterance={utterance}")


# =========================
# 3) Flask 서버 실행
# =========================
def run_server():
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(2)  # 서버 뜰 시간 잠깐 대기

# =========================
# 4) pyngrok 공개 HTTPS URL 생성
# =========================
public_url = start_pyngrok_tunnel(
    PyngrokTunnelConfig(
        port=PORT,
        auth_token=os.environ.get("NGROK_AUTHTOKEN"),
    )
)

webhook_url = f"{public_url}/kakao/webhook"
health_url = f"{public_url}/health"

print("=" * 80)
print("Kakao chatbot server is running.")
print("Public URL :", public_url)
print("Webhook URL:", webhook_url)
print("Health URL :", health_url)
print("Active tunnels:", ngrok.get_tunnels())
print("=" * 80)
print("이제 카카오 오픈빌더 스킬 서버 URL에 위 Webhook URL을 넣으세요.")
print("이 셀을 실행한 상태로 두면, 카카오톡에서 들어오는 메시지를 계속 처리합니다.")
print("중지하려면 코랩에서 '런타임 > 실행 중단' 또는 셀 인터럽트를 누르세요.")
print("=" * 80)

# =========================
# 5) 셀을 계속 살아 있게 유지
# =========================
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Stopping ngrok tunnel...")
    try:
        ngrok.disconnect(public_url)
    except Exception:
        pass
    try:
        ngrok.kill()
    except Exception:
        pass
    print("Stopped.")
```

카카오 오픈빌더 스킬 서버의 URL은 `POST /kakao/webhook` 엔드포인트에 연결하면 됩니다. 헬스체크는 `GET /health`입니다.

### 직접 실행
```bash
python -m src.chatbot.kakao_colab_bot \
  --dart-api-key "YOUR_DART_API_KEY" \
  --openai-api-key "YOUR_OPENAI_API_KEY" \
  --openai-model "gpt-4o-mini" \
  --naver-client-id "YOUR_NAVER_CLIENT_ID" \
  --naver-client-secret "YOUR_NAVER_CLIENT_SECRET" \
  --input data/real_ohlcv.csv \
  --report-json pipeline_report_with_context.json \
  --figure-dir figures_with_context
```

백그라운드 실행 상태는 `result/chatbot_jobs.json`, 파이프라인 로그는 `result/chatbot_logs/` 아래에 저장됩니다.

## 테스트
```bash
pytest -q
```

## 참고 문서
- `docs/PROJECT_FEATURES_OVERVIEW.md`
- `docs/EXTERNAL_DATA_INTEGRATION_GUIDE.md`
- `docs/EXPERT_ANALYSIS_ROADMAP.md`
- `docs/PRIORITIZED_INVESTOR_ACTION_PLAN.md`
