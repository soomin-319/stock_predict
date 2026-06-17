# 09. 카카오 챗봇 및 Colab 통합

`src/chatbot/`은 KakaoTalk 챗봇과 Google Colab 연동 기능을 제공한다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `kakao_colab_bot.py` | Flask 웹훅 서버, 예측 잡 관리 |
| `intent.py` | 발화 의도 분류 |
| `responses.py` | 카카오 응답 형식 생성 |

---

## 배포 아키텍처

```
GitHub (코드 저장소)
    │
    │ clone / git pull
    ↓
Google Colab (런타임)
    │  ┌─────────────────────────────────┐
    │  │  stock-predict-kakao            │
    │  │  → Flask 서버 시작               │
    │  │  → ngrok 터널 노출               │
    │  └─────────────────────────────────┘
    │        ↑↓ HTTP webhook
    ↓
KakaoTalk 챗봇 (사용자 인터페이스)
```

Colab 통합 노트북: `colab/stock_predict_colab.py`

---

## 카카오 챗봇 서버 (`kakao_colab_bot.py`)

### 시작

```bash
stock-predict-kakao
# 또는
python src/chatbot/kakao_colab_bot.py
```

Flask 서버가 기본 포트 8000에서 시작되며 `/kakao/webhook` 엔드포인트(및 `/health`)를 노출한다.

### 핵심 클래스

```python
# src/chatbot/kakao_colab_bot.py:51
@dataclass(slots=True)
class PredictionJobState:
    symbol: str
    display_code: str
    command: list[str]
    log_path: str
    submitted_at: str
    status: str = "running"          # running / completed / failed
    pid: int | None = None
    exit_code: int | None = None
    completed_at: str | None = None
```

백그라운드 파이프라인 실행 잡의 상태를 추적한다.

### 실시간 추천 서비스

```python
# src/recommendation/realtime_close_betting.py
class RealTimeCloseBettingRecommendationService
```

장 마감 직전 종가 기준 베팅 추천을 계산한다.

---

## 기준데이터 서빙 모델 (published 베이스라인 + 세션 오버레이)

봇은 기본적으로 GitHub에 게시된 **기준데이터(baseline)** 를 서빙하고, 사용자가 요청한 종목만 세션에서 예측해 그 위에 덮어 보여준다. 첫 요청 시 기본 200종목 전체를 자동 예측(부트스트랩)하지 않는다.

```python
# src/chatbot/kakao_colab_bot.py
def _load_cached_result_simple(self) -> pd.DataFrame:
    # 1) published/latest/csv/result_simple.csv 베이스라인 로드
    # 2) 세션 result/ 결과 로드(_load_session_result_simple)
    # 3) 종목코드 기준으로 세션 행이 베이스라인을 덮어쓰는 합집합 반환
```

| 동작 | 설명 |
|------|------|
| 베이스라인 | `published/latest/csv/result_simple.csv`(GitHub 게시본)를 읽어 즉시 응답 |
| 세션 오버레이 | 사용자가 요청/"최신화"한 종목만 세션 `result/`에 예측 후, **종목코드 기준**으로 베이스라인 위에 덮음 |
| 부트스트랩 OFF | `_is_bootstrap_required()`는 항상 `False` — 캐시가 없어도 200종목 전체를 자동 실행하지 않음 |
| 세션 한정 | 세션 예측 결과는 GitHub `published/`에 push하지 않는다(로컬/런타임 한정) |

`PipelineRuntimeConfig` 관련 기본값(`src/chatbot/kakao_colab_bot.py`):

```python
input_csv: str = "result/session/session_ohlcv.csv"   # publish 입력(data/real_ohlcv.csv)과 분리
bootstrap_default_symbols: bool = False
bootstrap_on_launch: bool = False
prewarm_default_predictions: bool = False
published_dir: str = "published/latest"               # 서빙할 기준데이터 경로
```

> 기준데이터는 로컬에서 `stock-predict-publish`로 생성·게시한다(자세한 내용은 [01_pipeline.md](01_pipeline.md)의 "기준데이터 게시", `docs/OPERATIONS.md` 참고).

---

## 발화 의도 분류 (`intent.py`)

```python
# src/chatbot/intent.py
def normalize_utterance(utterance: str) -> str
def is_help_utterance(utterance: str) -> bool
def is_status_utterance(utterance: str) -> bool
```

| 키워드 패턴 | 분류 |
|-------------|------|
| `도움말`, `help`, `사용법`, `시작`, `안내` | 도움말 |
| `결과`, `상태`, `진행상황`, `조회`, `확인` | 상태 조회 |
| `최신화`, `새로고침`, `재실행`, `다시예측` | 갱신 요청 |
| `추천` | 추천 종목 요청 |
| `\d{6}(\.KS|\.KQ)?` | 종목 코드 조회 |
| 한글 종목명 | 종목명으로 검색 |

---

## 카카오 응답 형식 (`responses.py`)

```python
# src/chatbot/responses.py
def simple_text_response(text: str) -> dict
def quick_reply(label: str, message_text: str) -> dict
def attach_quick_replies(response: dict, quick_replies: list[tuple[str, str]] | None) -> dict
```

카카오 스킬 API 형식 (v2)으로 응답을 생성한다:

```json
{
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": "삼성전자 (005930)\n권고: 매수\n예상 수익률: +2.34%"
                }
            }
        ],
        "quickReplies": [
            {"label": "도움말", "action": "message", "messageText": "도움말"}
        ]
    }
}
```

---

## 주요 챗봇 기능

### 1. 종목 예측 조회

사용자가 종목 코드(6자리) 또는 한글 종목명을 입력하면:

1. `published/latest/` 기준데이터(+세션 오버레이)에 해당 종목이 있으면 즉시 응답
2. 사용자가 그 종목을 명시적으로 요청/"최신화"하면 **해당 종목만** 세션에서 백그라운드 예측 → 완료 후 베이스라인 위에 오버레이
3. 기준데이터에도 없고 세션 예측도 아직이면 "예측 생성 중..." 응답 (기본 200종목 전체 부트스트랩은 하지 않음)

### 2. 예측 갱신 요청

"최신화", "새로고침" 등의 키워드 → 현재 종목에 대한 파이프라인 재실행.

### 3. 추천 종목 조회

"추천" 키워드 → `result_simple.csv`에서 `recommendation == "매수"` 종목 목록 반환.

### 4. 실시간 종가 추천

```python
# src/recommendation/realtime_close_betting.py
class RealTimeCloseBettingRecommendationService:
    def get_recommendation(self, symbol: str) -> str
```

장 마감 직전 종가 기준 베팅 추천을 실시간으로 계산.

---

## 보안 처리

```python
# src/utils/secrets.py
def redact_argv(argv: list) -> list
def redact_value(value: str) -> str
```

로그 및 에러 메시지에서 API 키(`OPENAI_API_KEY`, `DART_API_KEY` 등)가 자동으로 마스킹된다.

---

## 종목 검색

```python
# src/data/krx_universe.py
def find_symbol_candidates_by_name(query: str, limit: int | None = None) -> list[dict[str, str | float]]
```

한글 종목명으로 심볼을 검색할 때 사용. 각 후보는 `symbol`, `ticker`, `name`, `market`, `score` 키를 가진 dict다.
"삼성전자" → `[{"symbol": "005930.KS", "ticker": "005930", "name": "삼성전자", "market": "KOSPI", "score": 1.0}]`.

---

## Colab 통합 (`colab/stock_predict_colab.py`)

Colab 기본 흐름은 GitHub에 게시된 기준데이터를 **파이프라인 실행 없이** 서빙하는 것이다:

```python
# 1) 최신 코드/기준데이터 받기
!git pull
# 2) GitHub 기준데이터 표시 (파이프라인 미실행)
from colab.stock_predict_colab import load_published_predictions
load_published_predictions()           # 최신; 특정일은 load_published_predictions("2026-06-17")
# 3) 봇 실행 (자동 부트스트랩 OFF, published 베이스라인 서빙)
from src.chatbot.kakao_colab_bot import launch_colab_kakao_bot, PyngrokTunnelConfig
launch_colab_kakao_bot(tunnel_config=PyngrokTunnelConfig(auth_token="..."), prewarm_cache=False)
```

```python
# colab/stock_predict_colab.py
def load_published_predictions(date: str | None = None, rows: int = 5) -> dict[str, str]
```

`published/latest/`(또는 `published/history/<날짜>/`)의 `result_simple.csv` 경로·`trading_date`를 반환하고 미리보기를 출력하며 **파이프라인을 실행하지 않는다**. 데이터가 없으면 빈 경로와 가용 인덱스(`published/index.json`)를 안내한다. 기본 200종목 재예측은 사용자가 명시적으로 `run_colab_pipeline(...)`을 호출할 때만 수행한다.

`launch_colab_kakao_bot(prewarm_cache=...)`의 `prewarm_cache` 기본값은 `None`이며, 미지정 시 `prewarm_default_predictions` 값을 따른다. 실제 prewarm은 그 값과 `bootstrap_on_launch`가 모두 켜진 경우에만 동작한다(둘 다 기본 `False`).

Colab 런타임은 재시작 시 상태가 초기화되지만, 기준데이터는 GitHub `published/`에 있으므로 `git pull`로 복원된다. 세션 예측 결과 영속성이 필요하면 Google Drive 마운트를 사용한다.

---

## 응답 예시

```
삼성전자 (005930.KS)
━━━━━━━━━━━━━━━━━━
권고: 매수 ✓
예상 수익률: +2.34%
상승확률: 62.1%
신뢰도: 높음
내일 예상 종가: 78,500원
━━━━━━━━━━━━━━━━━━
[기준일: 2025-06-13]
```

---

## 개선 및 수정 진행 현황

> 우선순위: **P0(보안) > P1(견고성/UX) > P2(문서)**. 기준일: 2026-06-17.

### 미해결 — P0 `/kakao/webhook` 엔드포인트 인증 부재

- **문제**: Flask `/kakao/webhook`은 `request.get_json(silent=True)`만 받고(`kakao_colab_bot.py:1912-1924`) **요청 검증이 없다**. 코드의 `auth_token`은 **ngrok 토큰**(`PyngrokTunnelConfig`)이지 웹훅 인증이 아니다. ngrok URL이 노출되면 누구나 예측 잡(서브프로세스)을 제출해 **자원 고갈(DoS)**·로그/캐시 오염을 유발할 수 있다.
- **제안**: 공유 시크릿 헤더 검증(예: `X-Webhook-Secret` 상수시간 비교) 또는 카카오 요청 서명/허용 IP 검증을 추가. 시크릿은 환경변수로 주입.

### 미해결 — P0 잡 제출 레이트리밋/동시성 상한

- **문제**: 발화마다 백그라운드 파이프라인 잡을 띄울 수 있어(`PredictionJobState`/`process_runner`), 반복 호출 시 동시 서브프로세스가 무제한 증가한다(특히 무료 Colab 1코어). 현재 코드에 동시성 세마포어·쿨다운·in-flight 중복 차단이 없다.
- **제안**: 사용자/전역 동시 잡 수 상한, 동일 종목 in-flight 중복 제출 차단, 최소 재실행 간격(쿨다운) 적용.

> 참고(양호): 시크릿은 argv가 아닌 **subprocess 환경변수**로 전달되고(`build_subprocess_env`), 로그·에러는 `redact_*`로 마스킹되며, `Popen`은 리스트 인자(shell=False)라 셸 인젝션 위험은 낮다. 이 패턴은 유지 권장.

### 미해결 — P1 의도 분류가 정확일치(exact match)뿐

- **문제**: `intent.py`는 `normalize_utterance(...).lower() in SET` 정확일치만 한다(`intent.py:11-16`). "도움말좀", "결과 좀 보여줘"처럼 조사·공백이 붙으면 매칭 실패한다. 표가 약속한 `최신화/추천/종목코드/종목명` 분기는 `intent.py`가 아니라 `kakao_colab_bot.py`에 흩어져 있어 일관성이 낮다.
- **제안**: 부분일치/정규화(공백·문장부호 제거)·동의어 사전, 종목코드 정규식과 한글명 검색을 하나의 의도 라우터로 통합하고 단위 테스트로 고정.

### 미해결 — P1 종목 추출 입력 검증

- **문제**: 첫 토큰에 숫자가 있으면 종목 코드로 사용한다(`kakao_colab_bot.py`). 비정상 토큰이 `--add-symbols` 인자로 흘러갈 수 있다(셸 인젝션은 아니나 무의미한 잡 유발).
- **제안**: `^\d{6}(\.(KS|KQ))?$` 형식 검증 후에만 잡 제출, 한글명은 `find_symbol_candidates_by_name`로 해석 실패 시 후보 제시.

### 해결됨 — P1 `attach_quick_replies` 시그니처 문서 정정

- 본문 시그니처를 실제 `attach_quick_replies(response, quick_replies: list[tuple[str, str]] | None)`로 정정했다(`responses.py:14`).

### 미해결 — P1 카카오 응답 길이·형식 제약

- **문제**: `responses.py`는 `simpleText`만 지원하고(`simple_text_response`) 카카오의 텍스트 길이 상한(약 1000자 단위)을 강제하지 않는다. 추천 종목 다수 출력 시 잘릴 수 있다.
- **제안**: 길이 초과 시 분할/요약, `listCard`·`basicCard` 등 리치 포맷 지원.

### 미해결 — P2 Colab 상태 비영속성/복구

- **문제**: Colab 재시작 시 in-flight 잡과 로그가 사라진다. 잡 상태가 메모리(`PredictionJobState`)에만 있으면 복구 불가.
- **제안**: 잡 상태를 Drive 마운트 경로의 JSON으로 영속화하고, 부팅 시 `running` 잡을 정리/재동기화.
