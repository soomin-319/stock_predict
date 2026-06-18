# 09. 카카오 챗봇 및 Colab 통합

`src/chatbot/`은 KakaoTalk 챗봇과 Google Colab 연동 기능을 제공한다. 이 기능은 연구/운영 보조용이며, 투자 조언이나 자동매매 시스템이 아니다. 매수/매도/보유 판단은 `predicted_return` 기반 결과만 사용한다. 뉴스와 공시는 표시용 참고 정보이며 예측값, 순위, 추천, 신호를 바꾸면 안 된다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `kakao_colab_bot.py` | Flask 웹훅 서버, 예측 잡 관리, Colab/ngrok 실행 보조 |
| `intent.py` | 발화 정규화 및 도움말/상태 의도 분류 |
| `responses.py` | 카카오 응답 형식 생성, 텍스트 길이 제한, quick reply 생성 |

---

## 배포 아키텍처

```
GitHub (코드 저장소)
    │ clone / git pull
    ↓
Google Colab (런타임)
    │  stock-predict-kakao
    │  → Flask 서버 시작
    │  → ngrok 터널 노출
    ↓
KakaoTalk 챗봇 Webhook (/kakao/webhook)
```

Colab 실행 엔트리포인트: `colab/stock_predict_colab.py`

---

## 챗봇 서버 실행

```bash
stock-predict-kakao
# 또는
python src/chatbot/kakao_colab_bot.py
```

기본 포트는 `8000`이며, Flask 앱은 `/health`와 `/kakao/webhook` 엔드포인트를 제공한다.

### 주요 실행 옵션/환경변수

| 설정 | CLI | 환경변수 | 기본값 | 설명 |
|------|-----|----------|--------|------|
| 입력 CSV | `--input` | - | `data/real_ohlcv.csv` | 파이프라인 입력 데이터 |
| 리포트 JSON | `--report-json` | - | `pipeline_report_with_context.json` | 실행 리포트 파일명 |
| 웹훅 공유 시크릿 | `--kakao-webhook-secret` | `KAKAO_WEBHOOK_SECRET` | 미설정 | 설정 시 `X-Webhook-Secret` 헤더 필수 |
| 동시 예측 잡 수 | `--max-concurrent-prediction-jobs` | `MAX_CONCURRENT_PREDICTION_JOBS` | `2` | 백그라운드 파이프라인 동시 실행 상한 |
| 재실행 쿨다운 | `--refresh-cooldown-seconds` | `PREDICTION_REFRESH_COOLDOWN_SECONDS` | `60` | 같은 종목 완료 직후 재실행 최소 간격 |
| ngrok 인증 | `--ngrok-auth-token` | - | 미설정 | pyngrok 터널 인증 토큰 |

---

## 보안: 웹훅 인증

`KAKAO_WEBHOOK_SECRET` 또는 `--kakao-webhook-secret`을 설정하면 `/kakao/webhook`은 요청 헤더 `X-Webhook-Secret` 값을 상수 시간 비교로 검증한다.

```http
POST /kakao/webhook
X-Webhook-Secret: <shared-secret>
Content-Type: application/json
```

- 시크릿 미설정: 기존 Colab/로컬 개발처럼 인증 없이 동작한다.
- 시크릿 설정 + 헤더 누락/불일치: `401 Unauthorized`를 반환한다.
- ngrok 인증 토큰은 터널 인증용이지 애플리케이션 웹훅 인증이 아니므로 별도로 설정해야 한다.

---

## 예측 잡 관리

```python
@dataclass(slots=True)
class PredictionJobState:
    symbol: str
    display_code: str
    command: list[str]
    log_path: str
    submitted_at: str
    status: str = "running"  # running / completed / failed
    pid: int | None = None
    exit_code: int | None = None
    completed_at: str | None = None
```

잡 상태는 `result/runtime/chatbot_jobs.json`에 저장된다. 로그는 `result/runtime/logs/`에 기록된다.

안전장치:

1. 같은 종목의 `running` 잡이 있으면 새 subprocess를 만들지 않고 기존 잡을 재사용한다.
2. 전체 동시 실행 수가 `max_concurrent_prediction_jobs` 이상이면 새 잡을 거부한다.
3. 직전 완료 시각이 `refresh_cooldown_seconds`보다 짧으면 재실행을 거부한다.
4. API 키는 subprocess argv가 아니라 환경변수로 전달하고, 로그/상태 저장 전 `redact_*`로 마스킹한다.

---

## 발화 의도 분류

`src/chatbot/intent.py`는 입력을 소문자화하고 공백/문장부호를 정리한 뒤 키워드 포함 여부를 확인한다.

| 예시 발화 | 분류 |
|-----------|------|
| `도움말`, `도움말 좀 알려줘!`, `help`, `사용법`, `시작`, `안내` | 도움말 |
| `결과`, `결과 확인 부탁`, `상태`, `진행상황`, `조회`, `status` | 상태 조회 |
| `최신화`, `새로고침`, `재실행`, `다시 예측` | 예측 갱신 요청 |
| `추천` | 추천 종목 요청 |
| `005930`, `005930.KS`, `005930 보여줘` | 종목코드 조회 |
| `삼성전자` | 종목명 검색 |

종목코드는 `\d{6}` 또는 `\d{6}.KS/.KQ` 형식만 허용한다. `abc123`, `12345` 같은 잡음 섞인 숫자 토큰은 종목코드로 실행하지 않는다.

---

## 카카오 응답 형식

`src/chatbot/responses.py`는 Kakao i Open Builder v2 형식의 `simpleText` 응답을 만든다.

```json
{
  "version": "2.0",
  "template": {
    "outputs": [
      {"simpleText": {"text": "삼성전자 (005930.KS)\n권고: 매수\n예상 수익률: +2.34%"}}
    ],
    "quickReplies": [
      {"label": "도움말", "action": "message", "messageText": "도움말"}
    ]
  }
}
```

- `simple_text_response()`는 기본 900자까지 출력하고 초과분은 `...(생략)`으로 줄인다.
- `_build_response()`는 quick reply를 최대 10개까지 붙인다.
- `attach_quick_replies()`의 입력은 `list[tuple[str, str]]` 형식이다.

---

## 주요 챗봇 기능

### 1. 종목 예측 조회

사용자가 6자리 종목코드나 한국 종목명을 입력하면:

1. `result/result_simple.csv` 캐시 확인
2. 캐시가 있으면 즉시 응답
3. 캐시가 없으면 백그라운드 파이프라인 실행 후 진행 안내 응답

### 2. 예측 갱신

`최신화`, `새로고침`, `다시 예측` 등은 마지막 조회 종목의 파이프라인 재실행을 요청한다. 중복 실행, 동시 실행 초과, 쿨다운 위반은 거부된다.

### 3. 추천 종목 조회

`추천` 발화는 `result_simple.csv`에서 추천 후보를 조회한다. 추천/권고 판단은 `predicted_return` 및 기존 추천 정책만 사용한다.

### 4. 실시간 종가 추천

```python
class RealTimeCloseBettingRecommendationService:
    def get_recommendation(self, symbol: str) -> str
```

장 마감 직전 종가 기준 추천 메시지를 계산한다. 외부 뉴스/공시 맥락은 메시지 표시용이며 추천 신호를 바꾸지 않는다.

---

## 보안 처리

```python
from src.utils.secrets import redact_argv, redact_value
```

로그와 상태 파일에 남을 수 있는 `OPENAI_API_KEY`, `DART_API_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` 값은 저장 전 마스킹한다. 새 비밀값은 CLI 인자보다 환경변수 사용을 우선한다.

---

## Colab 운영 팁

Colab 런타임은 재시작 시 메모리 상태가 사라진다. 현재 잡/세션 상태는 `result/runtime/*.json`에 저장되지만, Colab 세션 자체가 사라지면 로컬 디스크도 사라질 수 있다. 장기 운영 시 Google Drive 마운트 경로에 `result/`를 보존하는 구성이 필요하다.

---

## 남은 개선 과제

- P2: Colab Drive 기반 상태/로그 영속화와 부팅 시 `running` 잡 정리.
- P2: 추천 목록이 길 때 `listCard`/`basicCard` 같은 리치 포맷 추가 검토.
- P2: 운영 환경에서 허용 IP 또는 카카오 요청 서명 검증을 추가할 수 있는지 검토.
