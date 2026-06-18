# 09. 카카오 챗봇 및 Colab 통합

`src/chatbot/`은 KakaoTalk 챗봇과 Google Colab 연동 기능을 제공한다. 연구/운영 보조용이며 투자 자문이나
자동매매 시스템이 아니다. 매수/매도/보유 판단은 `predicted_return` 기반 결과만 사용하고, 뉴스/공시는
표시용 참고 정보로 예측값·순위·추천·신호를 바꾸지 않는다.

## 모듈 구성

| 모듈 | 역할 |
|------|------|
| `kakao_colab_bot.py` | Flask 웹훅 서버, 예측 잡 관리, Colab/ngrok 실행 보조 |
| `intent.py` | 발화 정규화 및 도움말/상태 의도 분류 |
| `responses.py` | 카카오 응답 형식 생성, 텍스트 길이 제한, quick reply/listCard 생성 |

---

## 배포 아키텍처

```
GitHub (코드 저장소)
    │ clone / git pull
    ↓
Google Colab (런타임)
    │  stock-predict-kakao → Flask 서버 시작 → ngrok 터널 노출
    ↓
KakaoTalk 챗봇 Webhook (/kakao/webhook)
```

---

## 챗봇 서버 실행

```bash
stock-predict-kakao
# 또는
python src/chatbot/kakao_colab_bot.py
```

기본 포트는 `8000`이며 Flask 앱은 `/health`와 `/kakao/webhook` 엔드포인트를 제공한다.

### 주요 실행 옵션/환경변수

| 설정 | CLI | 환경변수 | 기본값 | 설명 |
|------|-----|----------|--------|------|
| 입력 CSV | `--input` | - | `data/real_ohlcv.csv` | 파이프라인 입력 데이터 |
| 리포트 JSON | `--report-json` | - | `pipeline_report_with_context.json` | 실행 리포트 파일명 |
| 런타임 디렉터리 | `--runtime-dir` | `CHATBOT_RUNTIME_DIR` | `result/runtime` | 잡/세션 상태, 로그, prewarm 메타 저장 위치 |
| 웹훅 공유 시크릿 | `--kakao-webhook-secret` | `KAKAO_WEBHOOK_SECRET` | 미설정 | 설정 시 `X-Webhook-Secret` 헤더 필수 |
| 동시 예측 잡 수 | `--max-concurrent-prediction-jobs` | `MAX_CONCURRENT_PREDICTION_JOBS` | `2` | 백그라운드 파이프라인 동시 실행 상한 |
| 재실행 쿨다운 | `--refresh-cooldown-seconds` | `PREDICTION_REFRESH_COOLDOWN_SECONDS` | `60` | 같은 종목 완료 직후 재실행 최소 간격(초) |
| ngrok 인증 | `--ngrok-auth-token` | - | 미설정 | pyngrok 터널 인증 토큰 |
| 웹훅 CIDR 허용목록 | `--allowed-webhook-cidrs` | `KAKAO_ALLOWED_WEBHOOK_CIDRS` | 미설정 | 쉼표 구분 IP/CIDR 허용목록. 미설정 시 기존 전체 허용 동작 유지 |

---

## 보안: 웹훅 인증

`KAKAO_WEBHOOK_SECRET` 또는 `--kakao-webhook-secret`을 설정하면 `/kakao/webhook`은 요청 헤더
`X-Webhook-Secret` 값을 상수 시간 비교로 검증한다.

- 시크릿 미설정: 기존 Colab/로컬 개발처럼 인증 없이 동작한다.
- 시크릿 설정 + 헤더 누락/불일치: `401 Unauthorized`를 반환한다.
- ngrok 인증 토큰은 터널 인증용이지 애플리케이션 웹훅 인증이 아니므로 별도로 설정해야 한다.
- `KAKAO_ALLOWED_WEBHOOK_CIDRS`/`--allowed-webhook-cidrs`는 `request.remote_addr`로 출처 IP/CIDR을 제한한다.
  예: `203.0.113.10/32,198.51.100.0/24`. `X-Forwarded-For`는 신뢰하지 않는다(신뢰 프록시 지원은 별도 설계 필요).

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
    status: str = "running"   # running / completed / failed
    pid: int | None = None
    exit_code: int | None = None
    completed_at: str | None = None
```

잡 상태는 기본적으로 `result/runtime/chatbot_jobs.json`에, 세션은 `chatbot_sessions.json`에, prewarm 메타는
`prewarm_cache_meta.json`에 저장되고, 잡 로그는 `result/runtime/logs/`에 쌓인다. `--runtime-dir` 또는
`CHATBOT_RUNTIME_DIR`로 이 디렉터리를 바꿀 수 있다(레거시 `result/chatbot_jobs.json`은 부팅 시 마이그레이션).
Colab에서는 Drive 경로를 쓰면 런타임 재시작 후에도 상태가 유지된다(예:
`CHATBOT_RUNTIME_DIR=/content/drive/MyDrive/stock_predict/runtime`).

안전장치:

1. 같은 종목의 `running` 잡이 있으면 새 subprocess를 만들지 않고 기존 잡을 재사용한다.
2. 전체 동시 실행 수가 `max_concurrent_prediction_jobs` 이상이면 새 잡을 거부한다.
3. 직전 완료 시각이 `refresh_cooldown_seconds`보다 짧으면 재실행을 거부한다.
4. API 키는 subprocess argv가 아니라 환경변수로 전달하고, 로그/상태 저장 전 `redact_*`로 마스킹한다.
5. 부팅 시 실제 활성 프로세스가 없는 `running` 잡은 좀비 상태로 보고 `failed`, `exit_code=-2`,
   `note="stale_after_restart"`(요청 처리 중 발견 시 `note="stale_running_state"`)로 정리한다.

---

## 발화 의도 분류 (`intent.py`)

입력을 소문자화하고 공백/문장부호를 정리한 뒤 키워드 포함 여부를 확인한다.

| 예시 발화 | 분류 |
|-----------|------|
| `도움말`, `help`, `사용법`, `시작`, `안내` | 도움말 |
| `결과`, `상태`, `진행상황`, `조회`, `확인`, `status` | 상태 조회 |
| `최신화`, `새로고침`, `재실행`, `다시 예측` | 예측 갱신 요청 |
| `추천` | 추천 종목 요청 |
| `005930`, `005930.KS`, `005930 보여줘` | 종목코드 조회 |
| `삼성전자` | 종목명 검색 |

종목코드는 `\d{6}` 또는 `\d{6}.KS/.KQ` 형식만 허용한다. `abc123`, `12345` 같은 잡음 토큰은 종목코드로 실행하지 않는다.

---

## 카카오 응답 형식 (`responses.py`)

Kakao i Open Builder v2 형식의 `simpleText`/`listCard` 응답을 만든다.

```json
{
  "version": "2.0",
  "template": {
    "outputs": [{"simpleText": {"text": "삼성전자 (005930.KS)\n권고: 매수\n예상 수익률: +2.34%"}}],
    "quickReplies": [{"label": "도움말", "action": "message", "messageText": "도움말"}]
  }
}
```

- `simple_text_response()`는 기본 900자(`DEFAULT_SIMPLE_TEXT_MAX_LENGTH`)까지 출력하고 초과분은 `...(생략)`으로 줄인다.
- `attach_quick_replies()`는 `list[tuple[str, str]]`을 받아 quick reply를 붙인다.
- `list_card_response()`는 추천 목록을 listCard로 만들며 quick reply를 최대 10개까지 첨부한다.

---

## 주요 챗봇 기능

1. **종목 예측 조회**: 6자리 코드나 한국 종목명을 입력하면 `result/result_simple.csv` 캐시를 확인해 즉시 응답하고,
   캐시가 없으면 백그라운드 파이프라인을 실행한 뒤 진행 안내를 보낸다.
2. **예측 갱신**: `최신화`/`새로고침`/`다시 예측`은 마지막 조회 종목의 재실행을 요청한다. 중복/동시 초과/쿨다운 위반은 거부된다.
3. **추천 종목 조회**: `추천`은 `result_simple.csv`에서 `predicted_return` 기준으로 상위 종목을 골라 Kakao
   `listCard`(종목명/추천/예상수익률)로 보여준다. 외부 뉴스/공시 맥락은 메시지 표시용일 뿐 추천 신호를 바꾸지 않으며,
   결과가 없으면 `simpleText`로 폴백한다.
4. **실시간 종가 추천**: `RealTimeCloseBettingRecommendationService.get_recommendation(symbol)`로 장 마감 직전
   종가 기준 추천을 계산한다([06](06_signal_policy.md)).

---

## 보안 처리

로그/상태 파일에 남을 수 있는 `OPENAI_API_KEY`, `DART_API_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`
값은 `src.utils.secrets`의 `redact_argv`/`redact_text`로 저장 전 마스킹한다. 새 비밀값은 CLI 인자보다 환경변수 사용을 우선한다.

---

## Colab 운영 팁

Colab 런타임은 종료 시 로컬 파일이 사라진다. 잡/세션 상태는 `result/runtime/*.json`에 저장되므로,
재시작 후에도 상태를 유지하려면 `CHATBOT_RUNTIME_DIR`/`--runtime-dir`을 Google Drive 경로로 지정한다
(예: `/content/drive/MyDrive/stock_predict/runtime`). 재시작 직후 실제 프로세스가 없는 `running` 잡은
부팅 정리 로직이 좀비로 처리한다.

---

## 개선 및 수정 제안

> 우선순위: **P2**. 기존 P0/P1(웹훅 인증·CIDR 허용목록, 잡 동시성/쿨다운 상한, Drive 친화 런타임 영속화,
> 좀비 잡 정리)은 모두 구현되었다.

### P2 — 공식 Kakao 요청 서명 검증

- **제안**: 현재는 공유 시크릿 헤더 + CIDR 허용목록으로 보호한다. 공식 Kakao 요청 서명(request signature) 스펙을
  확인한 뒤, 가능하면 서명 검증을 추가해 출처 위조를 더 강하게 방지한다.
