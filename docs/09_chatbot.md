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

Flask 서버가 기본 포트 5000에서 시작되며 `/webhook` 엔드포인트를 노출한다.

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

캐시된 `result/latest/csv/result_simple.csv`에서 즉시 응답하거나, 캐시가 없으면 백그라운드 파이프라인 잡을 제출한다.

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
def attach_quick_replies(response: dict, replies: list[dict]) -> dict
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

1. `result_simple.csv` 캐시 확인
2. 캐시 있으면 즉시 응답
3. 캐시 없으면 백그라운드 파이프라인 잡 제출 → "예측 생성 중..." 응답

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
def find_symbol_candidates_by_name(name: str) -> list[tuple[str, str]]
```

한글 종목명으로 심볼을 검색할 때 사용. "삼성전자" → `[("005930.KS", "삼성전자")]`.

---

## Colab 통합 (`colab/stock_predict_colab.py`)

Google Colab에서의 전체 실행 흐름:

```python
# 1. 의존성 설치
# 2. GitHub에서 코드 clone
# 3. 환경변수 설정 (API 키 등)
# 4. 파이프라인 실행 또는 챗봇 서버 시작
# 5. ngrok으로 외부 노출
```

Colab 런타임은 재시작 시 상태가 초기화되므로, 결과 파일을 Google Drive에 마운트하여 영속성을 유지한다.

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

## 개선 및 수정 제안

> 우선순위: **P0(보안) > P1(견고성/UX) > P2(문서)**.

### P0 — `/webhook` 엔드포인트 인증 부재

- **문제**: Flask `/webhook`은 `request.get_json()`만 받고(`kakao_colab_bot.py:1878`) **요청 검증이 없다**. 코드의 `auth_token`은 **ngrok 토큰**(`:1985`)이지 웹훅 인증이 아니다. ngrok URL이 노출되면 누구나 예측 잡(서브프로세스)을 제출해 **자원 고갈(DoS)**·로그/캐시 오염을 유발할 수 있다.
- **제안**: 공유 시크릿 헤더 검증(예: `X-Webhook-Secret` 상수시간 비교) 또는 카카오 요청 서명/허용 IP 검증을 추가. 시크릿은 환경변수로 주입.

### P0 — 잡 제출 레이트리밋/동시성 상한

- **문제**: 발화마다 백그라운드 파이프라인 잡을 띄울 수 있어(`PredictionJobState`/`process_runner`), 반복 호출 시 동시 서브프로세스가 무제한 증가한다(특히 무료 Colab 1코어).
- **제안**: 사용자/전역 동시 잡 수 상한, 동일 종목 in-flight 중복 제출 차단, 최소 재실행 간격(쿨다운) 적용.

> 참고(양호): 시크릿은 argv가 아닌 **subprocess 환경변수**로 전달되고(`build_subprocess_env`), 로그·에러는 `redact_*`로 마스킹되며, `Popen`은 리스트 인자(shell=False)라 셸 인젝션 위험은 낮다. 이 패턴은 유지 권장.

### P1 — 의도 분류가 정확일치(exact match)뿐

- **문제**: `intent.py`는 `normalize_utterance(...).lower() in SET` 정확일치만 한다(`intent.py:11-16`). "도움말좀", "결과 좀 보여줘"처럼 조사·공백이 붙으면 매칭 실패한다. 문서 표가 약속한 `최신화/추천/종목코드/종목명` 분기는 `intent.py`가 아니라 `kakao_colab_bot.py`에 흩어져 있어 일관성이 낮다.
- **제안**: 부분일치/정규화(공백·문장부호 제거)·동의어 사전, 종목코드 정규식과 한글명 검색을 하나의 의도 라우터로 통합하고 단위 테스트로 고정.

### P1 — 종목 추출 입력 검증

- **문제**: 첫 토큰에 숫자가 있으면 종목 코드로 사용한다(`kakao_colab_bot.py:620-622`). 비정상 토큰이 `--add-symbols` 인자로 흘러갈 수 있다(셸 인젝션은 아니나 무의미한 잡 유발).
- **제안**: `^\d{6}(\.(KS|KQ))?$` 형식 검증 후에만 잡 제출, 한글명은 `find_symbol_candidates_by_name`로 해석 실패 시 후보 제시.

### P1 — 카카오 응답 길이·형식 제약

- **문제**: `responses.py`는 `simpleText`만 지원하고 카카오의 텍스트 길이 상한(약 1000자 단위)을 강제하지 않는다. 추천 종목 다수 출력 시 잘릴 수 있다.
- **제안**: 길이 초과 시 분할/요약, `listCard`·`basicCard` 등 리치 포맷 지원. 문서의 `attach_quick_replies(replies: list[dict])` 시그니처는 실제 `list[tuple[str, str]]`이므로 정정.

### P2 — Colab 상태 비영속성/복구

- **문제**: Colab 재시작 시 in-flight 잡과 로그가 사라진다(문서 언급). 잡 상태가 메모리(`PredictionJobState`)에만 있으면 복구 불가.
- **제안**: 잡 상태를 Drive 마운트 경로의 JSON으로 영속화하고, 부팅 시 `running` 잡을 정리/재동기화.
