# 코드베이스 개선 및 수정 제안서 (2026-06-22)

> 기준 문서: `docs/CODEBASE_ANALYSIS_FULL_2026-06-22.md`  
> 대상 저장소: `stock_predict`  
> 목적: 현재 분석 결과를 바탕으로 정책 무결성, 운영 안정성, 유지보수성, 테스트 신뢰도를 높이기 위한 개선 및 수정 후보를 우선순위별로 정리한다.  
> 주의: 본 프로젝트 산출물은 리서치/운영 지원용이며 투자자문 또는 자동매매 지시가 아니다.

---

## 1. 핵심 요약

현재 코드베이스는 단순 예측 스크립트가 아니라 데이터 수집, 피처 생성, walk-forward 검증, 백테스트, 산출물 관리, 챗봇 응답, 뉴스 임팩트 표시 컨텍스트까지 포함한 통합 파이프라인이다.

가장 중요한 개선 방향은 다음 네 가지다.

1. **정책 무결성 강화**
   - 매수/매도/관망은 계속 `predicted_return`만 사용한다.
   - 뉴스, 공시, 뉴스 임팩트, 이슈 요약은 display-only로 유지한다.
2. **대형 모듈 분해**
   - `src/chatbot/kakao_colab_bot.py`, `src/pipeline.py`, `src/reports/issue_summary.py`의 책임을 단계적으로 분리한다.
3. **운영 산출물과 인코딩 안정화**
   - `result/` 산출물 정책, CSV `utf-8-sig`, 문서 UTF-8 저장, latest 승격 조건을 명확히 한다.
4. **외부 API 의존성 격리**
   - yfinance, DART, Naver, OpenAI/llama.cpp, ngrok은 테스트에서 기본 mock/disable되도록 유지하고, 실패 시 파이프라인 핵심 경로가 중단되지 않게 한다.

---

## 2. 최우선 가드레일

아래 항목은 모든 개선 작업 전에 먼저 확인해야 한다.

| 가드레일 | 유지해야 할 계약 | 대표 확인 지점 |
|---|---|---|
| 권고 정책 | `predicted_return` 외 값은 매수/매도/관망을 바꾸지 않는다. | `src/domain/signal_policy.py`, `tests/test_signal_policy_contract.py` |
| display-only 컨텍스트 | 뉴스/공시/뉴스임팩트 컬럼은 모델 입력, 랭킹, 권고에 들어가지 않는다. | `src/features/feature_selection.py`, `tests/test_display_only_feature_guard.py` |
| 산출물 위치 | 생성 CSV/JSON은 `result/` 아래에 둔다. | `src/reports/output.py`, `src/reports/run_artifacts.py` |
| CSV 인코딩 | CSV 산출물은 `utf-8-sig`를 사용한다. | 리포트/산출물 쓰기 함수 |
| 테스트 격리 | 외부 API는 기본적으로 mock 또는 disable 가능해야 한다. | data, news_impact, chatbot 관련 테스트 |
| 샘플 실행 안전성 | 샘플/smoke 실행은 production latest를 덮어쓰지 않는다. | `RunArtifactManager` 관련 테스트 |

---

## 3. 우선순위별 개선안

### P0. 정책 무결성 회귀 방지 강화

**문제**

프로젝트의 핵심 계약은 “권고는 `predicted_return`만 사용한다”는 점이다. 이미 테스트가 존재하지만, 챗봇/리포트/추천 경로가 늘어날수록 뉴스나 이슈 요약이 간접적으로 순위·추천·권고에 섞일 위험이 있다.

**제안**

- `signal_policy` 계약 테스트에 다음 케이스를 명시적으로 추가한다.
  - 같은 `predicted_return`, 다른 `up_probability`
  - 같은 `predicted_return`, 다른 `news_impact_score`
  - 같은 `predicted_return`, 다른 이슈 요약 텍스트
  - 같은 `predicted_return`, 다른 공시/뉴스 존재 여부
- 추천/챗봇 응답 테스트에서 display-only 컬럼이 권고 라벨을 바꾸지 않는지 확인한다.
- 문서와 코드 주석에 “display-only는 설명용이며 정책 입력이 아님”을 반복 명시한다.

**대상 파일**

- `src/domain/signal_policy.py`
- `src/features/feature_selection.py`
- `src/reports/news_impact_context.py`
- `src/chatbot/kakao_colab_bot.py`
- `tests/test_signal_policy_contract.py`
- `tests/test_display_only_feature_guard.py`

**검증**

```powershell
pytest tests/test_signal_policy_contract.py tests/test_display_only_feature_guard.py
```

---

### P1. `src/chatbot/kakao_colab_bot.py` 단계적 분해

**문제**

`src/chatbot/kakao_colab_bot.py`는 가장 큰 단일 모듈이며 Flask 서버, Kakao 응답 포맷, 캐시, 보안, ngrok, 파이프라인 잡 실행, 추천 응답을 함께 담당한다. 변경 충돌과 회귀 위험이 높다.

**제안 분해 방향**

| 새 책임 단위 | 역할 |
|---|---|
| `chatbot/app.py` | Flask 앱 생성과 route 등록 |
| `chatbot/handlers.py` | Kakao intent별 요청 처리 |
| `chatbot/responses.py` | Kakao 응답 JSON 포맷팅 |
| `chatbot/cache.py` | 결과 CSV 캐시 로드/무효화 |
| `chatbot/security.py` | HMAC, IP/CIDR, 시크릿 레닥션 |
| `chatbot/jobs.py` | 백그라운드 파이프라인 실행/상태 관리 |
| `chatbot/ngrok.py` | ngrok 터널 실행 보조 |

**권장 방식**

한 번에 대규모 이동하지 말고, 테스트가 있는 작은 단위부터 함수 이동을 반복한다.

1. 순수 포맷 함수부터 `responses.py`로 이동
2. 보안 관련 순수 함수와 검증 함수를 `security.py`로 이동
3. 캐시 로직을 `cache.py`로 이동
4. job 실행 로직을 `jobs.py`로 이동
5. 마지막에 Flask app factory를 분리

**검증**

```powershell
pytest tests/test_kakao_colab_bot.py
pytest tests/test_pipeline_smoke.py
```

---

### P1. `src/pipeline.py` 오케스트레이션 분리

**문제**

`pipeline.py`는 CLI 파싱, 데이터 로드, 피처 생성, 검증, 모델 학습, 예측, 산출물 작성까지 넓은 책임을 가진다. 파이프라인 변경 시 영향 범위가 크다.

**제안 분해 방향**

| 단위 | 역할 |
|---|---|
| `pipeline_cli.py` | CLI parser와 인자 정규화 |
| `pipeline_data.py` | 입력 로드, 유니버스 필터, 실데이터 refresh |
| `pipeline_features.py` | 피처 생성과 피처 컬럼 선택 |
| `pipeline_validation.py` | walk-forward, OOF, 보정, 백테스트 |
| `pipeline_training.py` | 최종 모델 학습과 persistence |
| `pipeline_outputs.py` | 최신 예측, report JSON, artifact 작성 |

**권장 방식**

- 기존 public entrypoint `src.pipeline:main`은 유지한다.
- 먼저 side-effect가 적은 helper부터 이동한다.
- 함수 이동 후에는 import 경로만 바꾸고 동작은 바꾸지 않는다.
- 각 단계마다 smoke 테스트를 실행한다.

**검증**

```powershell
pytest tests/test_pipeline_smoke.py
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

---

### P1. 문서/터미널 인코딩 점검

**문제**

분석 문서에서 PowerShell 기본 출력으로 한글이 깨져 보이는 현상이 있었다. 파일 자체는 UTF-8로 정상 판독 가능하지만, 운영자 환경에 따라 문서와 로그 가독성이 깨질 수 있다.

**제안**

- 문서 파일은 UTF-8로 저장한다.
- CSV는 기존 원칙대로 `utf-8-sig`를 유지한다.
- README 또는 운영 문서에 Windows PowerShell 권장 인코딩 설정을 추가한다.
- 문서 생성/검증 스크립트가 있다면 UTF-8 read/write를 명시한다.

**예시 운영 안내**

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

**검증**

```powershell
python - <<'PY'
from pathlib import Path
for path in ["README.md", "docs/CODEBASE_ANALYSIS_FULL_2026-06-22.md"]:
    Path(path).read_text(encoding="utf-8")
print("utf-8 docs ok")
PY
```

---

### P2. `src/reports/issue_summary.py` 책임 분리

**문제**

이슈 요약은 LLM/룰 기반 fallback, 프롬프트, 요약 포맷, 실패 처리, 표시 컨텍스트를 함께 다룰 가능성이 크다. display-only 경계를 더 선명하게 만들 필요가 있다.

**제안 분해 방향**

| 단위 | 역할 |
|---|---|
| `issue_summary/prompts.py` | 프롬프트 템플릿 |
| `issue_summary/llm.py` | LLM 호출과 실패 처리 |
| `issue_summary/rules.py` | 룰 기반 fallback |
| `issue_summary/formatting.py` | 사용자 표시 문구 |
| `issue_summary/types.py` | dataclass/TypedDict |

**추가 제안**

- 이슈 요약 산출물 필드명에 `display_only` 의미를 드러낸다.
- 요약 실패 시 빈 문자열보다 명확한 상태값을 둔다.
- LLM 응답이 권고 정책으로 전달되지 않는 테스트를 추가한다.

**검증**

```powershell
pytest tests -k "issue_summary or display_only"
```

---

### P2. 외부 API 의존성 표준화

**문제**

yfinance, DART, Naver, OpenAI/llama.cpp, ngrok 등 외부 의존성이 많다. 운영 기능에는 필요하지만 테스트와 샘플 실행에서는 재현성을 깨뜨릴 수 있다.

**제안**

- 외부 호출 옵션명을 일관화한다.
  - `--disable-external`
  - `--fetch-investor-context`
  - `--news-impact-report`
- 외부 호출 함수는 timeout, retry, fallback 결과를 명시한다.
- 테스트 fixture에서 네트워크 호출 차단을 기본값으로 둔다.
- 외부 API 실패는 `report_json` 또는 manifest에 경고로 남기되 핵심 예측 경로는 가능한 한 유지한다.

**검증**

```powershell
pytest tests/test_pipeline_smoke.py
pytest tests -k "external or fetch or investor or news_impact"
```

---

### P2. `result/` 산출물 관리 정책 명문화

**문제**

`result/`는 생성물 영역이다. 산출물이 커지거나 오래된 실행 결과가 쌓이면 git 오염과 운영 혼동이 생길 수 있다.

**제안**

- 커밋 가능한 산출물과 커밋 금지 산출물을 문서화한다.
- `result/runs/<run_id>/` 보존 기간 또는 cleanup 명령을 명확히 한다.
- 샘플 smoke 산출물이 `latest`를 덮어쓰지 않는 조건을 README/문서에 명시한다.
- manifest에 schema version, config hash, git commit, row count, sha256을 계속 기록한다.

**검증**

```powershell
pytest tests -k "artifact or manifest or result"
```

---

### P3. `signal_policy.py` row/vectorized 경로 정리

**문제**

row 단위 하위호환 함수와 vectorized 함수가 함께 있어 동작 차이가 생길 수 있다.

**제안**

- canonical 로직은 하나로 정한다.
- row 함수는 canonical vectorized 로직의 thin wrapper로 유지한다.
- 같은 입력에 대해 row/vectorized 결과가 같은지 property-style 테스트를 추가한다.

**검증**

```powershell
pytest tests/test_signal_policy_contract.py
```

---

### P3. 모델/검증 메타데이터 가독성 개선

**문제**

모델 backend, feature hash, config hash, random seed, artifact version이 이미 기록되지만 운영자가 실패 원인을 빠르게 파악하려면 요약 가독성이 더 좋아질 수 있다.

**제안**

- report JSON에 다음 요약을 명시한다.
  - 모델 backend
  - feature count
  - walk-forward fold 수
  - OOF row count
  - calibration 사용 여부
  - backtest 유효/스킵 사유
  - external feature 사용/스킵 사유
- PM 리포트에 “데이터 부족/검증 스킵/외부 API 실패”를 별도 warning 섹션으로 노출한다.

**검증**

```powershell
pytest tests -k "metadata or report or pm_report"
```

---

## 4. 모듈별 수정 제안 요약

| 영역 | 현 상태 | 제안 | 우선순위 |
|---|---|---|---|
| 권고 정책 | `predicted_return` 중심 계약 존재 | 권고 불변성 테스트 확대 | P0 |
| 피처 선택 | display-only 제외 로직 존재 | 뉴스/공시/임팩트 접두사 회귀 테스트 확대 | P0 |
| 챗봇 | 대형 단일 모듈 | response/security/cache/jobs/app 순서로 분해 | P1 |
| 파이프라인 | 큰 오케스트레이터 | CLI/data/features/validation/training/output 분리 | P1 |
| 문서/인코딩 | UTF-8 파일이나 터미널 깨짐 가능 | Windows 출력 인코딩 안내 추가 | P1 |
| 이슈 요약 | 역할 집중 가능성 | prompts/llm/rules/formatting/types 분리 | P2 |
| 외부 API | 여러 공급자 의존 | timeout/retry/fallback/mock 표준화 | P2 |
| result 관리 | manifest와 latest 관리 존재 | cleanup/커밋 정책 명문화 | P2 |
| signal row/vectorized | 병존 | canonical 로직과 동등성 테스트 | P3 |
| 리포트 메타데이터 | 기본 메타데이터 존재 | 운영 warning/스킵 사유 가독성 개선 | P3 |

---

## 5. 권장 실행 로드맵

### 1단계: 안전망 강화

1. `predicted_return` 단독 권고 테스트 추가
2. display-only 컬럼 제외 테스트 확장
3. 챗봇/추천 응답에서 뉴스·공시가 권고를 바꾸지 않는 테스트 추가

완료 기준:

```powershell
pytest tests/test_signal_policy_contract.py tests/test_display_only_feature_guard.py
pytest tests/test_pipeline_smoke.py
```

### 2단계: 문서/운영 안정화

1. README 또는 운영 문서에 인코딩 안내 추가
2. `result/` 산출물 커밋/보존 정책 문서화
3. 외부 API 실패 시 report warning 표준화 계획 정리

완료 기준:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

### 3단계: 챗봇 분해

1. 응답 포맷 함수 분리
2. 보안 함수 분리
3. 캐시 함수 분리
4. job 실행 함수 분리
5. Flask app factory 분리

완료 기준:

```powershell
pytest tests/test_kakao_colab_bot.py
pytest tests/test_pipeline_smoke.py
```

### 4단계: 파이프라인 분해

1. CLI parser 분리
2. 데이터 로드/refresh 분리
3. 피처 생성 분리
4. 검증/백테스트 분리
5. 산출물 작성 분리

완료 기준:

```powershell
pytest
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

---

## 6. 변경 시 테스트 매트릭스

| 변경 유형 | 최소 테스트 |
|---|---|
| 권고/시그널 정책 | `pytest tests/test_signal_policy_contract.py` |
| 피처 선택/display-only | `pytest tests/test_display_only_feature_guard.py` |
| 파이프라인 흐름 | `pytest tests/test_pipeline_smoke.py` |
| 산출물/manifest | `pytest tests -k "artifact or manifest or output"` |
| 챗봇 | `pytest tests -k "kakao or chatbot"` |
| 뉴스 임팩트 | `pytest tests -k "news_impact"` |
| 외부 API fetch | `pytest tests -k "fetch or external or investor"` |
| 제출 전 | `pytest` 및 샘플 파이프라인 실행 |

샘플 파이프라인 실행:

```powershell
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

---

## 7. 작업 전 체크리스트

- [ ] 이 변경이 매수/매도/관망 정책에 영향을 주는가?
- [ ] `predicted_return` 외 컬럼이 권고나 랭킹에 들어가는 경로가 생기는가?
- [ ] 뉴스/공시/뉴스임팩트 컬럼은 display-only로 유지되는가?
- [ ] 외부 API 없이 테스트가 통과하는가?
- [ ] 생성 CSV/JSON은 `result/` 아래에 저장되는가?
- [ ] CSV는 `utf-8-sig`로 저장되는가?
- [ ] 샘플 실행이 production latest를 덮어쓰지 않는가?
- [ ] 새 설정 키가 있다면 기본값과 문서가 있는가?
- [ ] 관련 pytest가 추가 또는 갱신되었는가?

---

## 8. 결론

현재 코드베이스의 가장 큰 자산은 예측 성능 자체보다 **정책 경계가 명확하다**는 점이다. 앞으로의 개선은 이 장점을 보존하면서 대형 모듈을 작게 나누고, 외부 API와 운영 산출물의 불확실성을 줄이는 방향이 적합하다.

가장 먼저 할 일은 기능 추가가 아니라 안전망 확장이다. `predicted_return` 단독 권고 계약과 display-only 컨텍스트 격리를 테스트로 더 강하게 고정한 뒤, 챗봇과 파이프라인을 단계적으로 분해하는 순서를 권장한다.
