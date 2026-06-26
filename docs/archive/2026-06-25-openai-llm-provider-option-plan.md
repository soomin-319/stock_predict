# OpenAI API 옵션 추가 구현 플랜 (공용 provider 스위치 / A안)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans 로 이 플랜을 태스크 단위로 구현하세요. 각 단계는 체크박스(`- [ ]`)로 추적합니다. AGENTS.md에 따라 **서브에이전트 금지·순차 실행**.

**Goal:** 현재 로컬 gemma(llama.cpp, port 8001)로만 도는 LLM 기능을, **명시적 전환**으로 OpenAI API도 쓸 수 있게 만든다. 두 LLM 소비자(`news_impact` 점수/판정 경로, `issue_summary` 종목 이슈 요약 경로)가 **단일 provider 설정 하나**(`llama_cpp` 기본 ↔ `openai`)를 함께 따른다. 종가 예측(`predicted_return`) 산식·입력은 불변.

**Architecture (A안 — 공용 스위치 하나):** 이미 존재하는 `LLMConfig`/`load_llm_config`(`src/news_impact/llm_config.py`)를 **provider 진실원본**으로 재사용한다. 새 추상화 레이어·팩토리는 만들지 않는다(YAGNI). 파이프라인이 단일 `LLMConfig`를 해석해 두 소비자에 전달한다.

- `news_impact` 경로는 **이미 provider 인지** 상태다: `LLMConfig`에 `provider`/`base_url`/`api_key`가 있고, `LlamaCppClient`가 OpenAI 호환 `/chat/completions`를 호출하며 `_auth_headers`가 OpenAI Bearer를 붙인다. 예시 설정 `configs/news_impact.{gemma,openai}.example.json`도 존재. → **검증·테스트 보강 위주.**
- `issue_summary` 경로는 **현재 OpenAI SDK 전용**(`responses.create`/`chat.completions`)이라 로컬 gemma로 못 돈다. → **여기에 provider/base_url 주입을 추가하는 것이 이 플랜의 핵심 신규 코드.**

**Tech Stack:** Python 3.10+(이 PC 3.14), pandas, `openai` SDK(issue_summary), 표준 `urllib`(news_impact). 신규 의존성 없음. 로컬 gemma·OpenAI 모두 OpenAI 호환 API 사용.

## Global Constraints

- **가드레일(불변):** 뉴스/공시 LLM 출력은 **표시·참고용**이다. provider를 무엇으로 바꾸든 점수 산식, 예측 입력, `predicted_return`은 절대 바뀌지 않는다. 요약 산출의 결정론·감사가능성·누출안전성을 유지한다(`참고용·예측값 미반영` 문구 보존).
- **기본값은 항상 로컬 gemma:** provider 미지정 시 `llama_cpp`/`http://localhost:8001/v1`/`gemma-4-26b-a4b`. 키가 우연히 환경에 있다고 자동으로 OpenAI로 넘어가지 않는다.
- **LLM 요약은 opt-in 유지:** `issue_summary` LLM 호출은 **명시적으로 설정을 줄 때만** 켜진다. 무설정 시 기존 규칙기반 폴백 동작을 그대로 둔다(불필요한 gemma 호출·지연 추가 금지 — `docs/2026-06-25-gemma-llm-latency-optimization.md` 취지와 일치).
- **하위 계약 보존:** `append_issue_summary_columns`·`_llm_symbol_issue_summary`·`PipelineRuntimeConfig` 등에 추가하는 모든 신규 인자/필드는 **기본값을 둬서** 기존 호출부·테스트가 깨지지 않게 한다. 기존 플래그(`--news-impact-llm-config`, `--openai-api-key`, `--openai-model`)는 **하위호환으로 계속 동작**한다.
- **시크릿은 env로:** `OPENAI_API_KEY`는 argv가 아닌 환경변수/`.env`로 주입(`ps`/히스토리 노출 방지). `PipelineRuntimeConfig.build_subprocess_env`가 이미 `OPENAI_API_KEY`를 자식 프로세스 env로 전달함 — 이 관례를 따른다.
- **출력·인코딩:** 생성 CSV/JSON은 `result/` 하위, CSV는 `utf-8-sig`.
- **테스트 실행(이 PC):** `result/` ACL deny 회피를 위해 basetemp 명시: `pytest <경로> -v --basetemp=.tmp_pytest`. 라이브 LLM 호출은 모킹/비활성화(가짜 클라이언트·전송).
- **PR 생성:** 변경 시 커밋·푸시·PR 필수. 제목·본문 자동 작성 후 `& "C:\Program Files\GitHub CLI\gh.exe" pr create`로 생성, `/pull/<번호>` URL 제공.

## 파일 구조

- 수정: `src/reports/issue_summary.py` — `_llm_symbol_issue_summary`/`append_issue_summary_columns`/`_append_issue_summary_columns_sequential`에 `provider`·`base_url` 추가, OpenAI SDK 클라이언트 `base_url` 생성, provider별 호출 경로 분기, LLM 게이트·모델 해석 수정, 캐시 키에 `provider`/`base_url` 반영.
- 수정: `src/pipeline.py` — 단일 `LLMConfig` 해석(`run_daily_pipeline`), 두 소비자에 전달(`_maybe_append_issue_summary` 인자 확장), `effective_*` 해석부 갱신.
- 수정: `src/pipeline_cli.py` — `--llm-config` 단일 스위치 플래그 추가(기존 플래그 유지).
- 수정: `src/chatbot/runtime_config.py` — `PipelineRuntimeConfig`에 `llm_config` 추가, `build_command`가 `--llm-config` 전달.
- 추가(문서 예시 재사용): `configs/news_impact.{gemma,openai}.example.json` 을 **공용 LLM 설정 예시**로 문서화(스키마 동일 — 신규 파일 불필요). 필요 시 `configs/llm.{gemma,openai}.example.json` 심볼릭 예시 추가는 선택.
- 수정/추가 테스트: `tests/test_issue_summary.py`, `tests/test_news_impact_llm_config.py`, `tests/test_news_impact_llm_cache.py`, `tests/test_pipeline_smoke.py`(스모크), 신규 `tests/test_llm_provider_wiring.py`(파이프라인이 단일 설정을 두 소비자에 전달하는지).
- 수정: `docs/FULL_LOCAL_RUN_WITH_GEMMA_KAKAO.md` 또는 신규 운영 노트 — OpenAI 전환 사용법.

---

### Task 1: issue_summary 를 provider 인지로 (코어 신규 코드)

**문제:** `_llm_symbol_issue_summary`(issue_summary.py:377)는 `from openai import OpenAI` 후 `OpenAI(api_key=api_key)`로 **클라우드 OpenAI에만** 붙는다. 로컬 gemma로 못 돌고, `_call_llm_text`/`_call_llm_json`은 항상 `responses.create`를 먼저 시도해 gemma에선 **매번 실패 후 chat 폴백**으로 왕복을 낭비한다.

**Files:**
- Modify: `src/reports/issue_summary.py`
- Test: `tests/test_issue_summary.py`

**Interfaces:**
- `_llm_symbol_issue_summary(..., api_key, model, provider="openai", base_url=None)` — `base_url`이 있으면 `OpenAI(api_key=..., base_url=...)`로 생성. gemma는 키가 없을 수 있으므로 `api_key`가 비면 더미(예: `"not-needed"`)를 사용.
- `_call_llm_text`/`_call_llm_json(..., provider)` — `provider == "llama_cpp"`면 `responses.create`를 **건너뛰고** `chat.completions`로 직행. `openai`면 기존대로 responses→chat 폴백.
- `append_issue_summary_columns(..., provider="openai", base_url=None)` / `_append_issue_summary_columns_sequential(...)` 동일 인자 추가(기본값으로 하위호환).
- LLM 게이트 변경: 기존 `use_llm = bool(openai_api_key and resolved_model and context ...)` →
  `llm_enabled = bool(resolved_model and context and "source_type" in context.columns and (provider == "llama_cpp" or openai_api_key))`.
  즉 **gemma는 키 없이도** 활성, OpenAI는 키 있을 때만 활성.
- 캐시 키(`_issue_summary_cache_key`)에 `provider`·`base_url`을 포함해 provider 간 캐시 충돌 방지.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_issue_summary.py 에 추가
class _FakeChat:
    def __init__(self, recorder): self._r = recorder
    class _Completions:
        ...
    # chat.completions.create 호출을 기록하고 JSON/텍스트 반환

class _FakeResponses:
    def create(self, **kw):
        raise RuntimeError("responses API not implemented")  # gemma 모사

def test_gemma_provider_skips_responses_and_uses_base_url(monkeypatch):
    calls = {"responses": 0, "chat": 0, "base_url": None}
    # OpenAI 생성자를 가짜로 패치해 base_url 캡처
    # provider="llama_cpp" 로 _call_llm_text 호출 시 responses.create 가 0회여야 함
    ...
    assert calls["responses"] == 0
    assert calls["chat"] >= 1
    assert calls["base_url"] == "http://localhost:8001/v1"

def test_openai_provider_uses_responses_first():
    # provider="openai" 면 responses.create 우선 시도
    ...

def test_issue_summary_gemma_enabled_without_api_key():
    # provider=llama_cpp, openai_api_key=None 이어도 LLM 경로가 켜지는지
    ...
```

- [ ] **Step 2: 테스트 통과(구현)** — 위 인터페이스대로 `issue_summary.py` 수정. responses 분기, base_url 주입, 게이트·캐시 키 갱신.
- [ ] **Step 3: 리팩터 점검** — `_call_llm_text`/`_call_llm_json` 중복 최소화, 더미 키 상수화, 기존 OpenAI 경로 회귀 없음 확인.
- [ ] **Step 4: 검증** — `pytest tests/test_issue_summary.py -v --basetemp=.tmp_pytest` 그린.

---

### Task 2: 단일 LLM 설정을 두 소비자에 배선 (공용 스위치)

**문제:** 현재 news_impact는 `--news-impact-llm-config`(LLMConfig), issue_summary는 `--openai-api-key`/`--openai-model`로 **따로** 설정된다. A안은 **하나의 스위치**가 둘 다 제어해야 한다.

**Files:**
- Modify: `src/pipeline_cli.py`(플래그), `src/pipeline.py`(해석·전달)
- Test: 신규 `tests/test_llm_provider_wiring.py`, `tests/test_pipeline_smoke.py`

**Interfaces:**
- 신규 플래그 `--llm-config <path>`(`pipeline_cli.py`): 주면 `load_llm_config`로 단일 `LLMConfig`를 만들고 **두 소비자 모두** 그 provider/base_url/model/api_key를 쓴다.
- 해석 우선순위(명시적, 자동전환 없음):
  1. `--llm-config` 제공 → 그 `LLMConfig`가 news_impact·issue_summary 공통 소스.
  2. 미제공 시 하위호환: `--news-impact-llm-config`는 news_impact만, `--openai-api-key/--openai-model`는 issue_summary만(=`provider="openai"`)을 종전대로 구동.
  3. 둘 다 없으면 LLM off(규칙기반/rule 폴백) — 기존 동작.
- `run_daily_pipeline`/`_maybe_append_issue_summary`(pipeline.py:701, 1059): 해석된 `provider`·`base_url`·`model`·`api_key`를 `append_issue_summary_columns`로 전달. issue_summary 모델 해석은 `config.model`(예: openai면 `gpt-5-mini`, gemma면 `gemma-4-26b-a4b`)을 사용하도록 `effective_openai_model` 로직 보강.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_llm_provider_wiring.py
def test_llm_config_drives_both_consumers(tmp_path, monkeypatch):
    cfg = tmp_path / "llm.json"
    cfg.write_text(json.dumps({
        "llm_provider": "openai",
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-5-mini",
    }), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # run_daily_pipeline(... llm_config=cfg ...) 호출 시
    # news_impact 와 issue_summary 양쪽에 동일 provider/base_url/model 전달되는지 캡처

def test_no_config_keeps_rule_based_default():
    # 무설정이면 두 경로 모두 LLM off(rule) — 기존 동작 보존

def test_backcompat_openai_flags_still_work():
    # --openai-api-key/--openai-model 만 줘도 issue_summary가 openai 로 동작
```

- [ ] **Step 2: 구현** — 플래그 추가, 해석부, 전달.
- [ ] **Step 3: 리팩터 점검** — 해석 로직을 작은 헬퍼(`_resolve_llm_config(args/env)`)로 빼 가독성 확보.
- [ ] **Step 4: 검증** — `pytest tests/test_llm_provider_wiring.py tests/test_pipeline_smoke.py -v --basetemp=.tmp_pytest`.

---

### Task 3: news_impact OpenAI 경로 end-to-end 검증·보강

**문제:** 토대는 있으나(`LLMConfig`+`LlamaCppClient`+`_auth_headers`+예시 설정) OpenAI 실경로 회귀 테스트가 약하다. `verify_model_alias`가 OpenAI의 큰 `/models` 응답·인증을 제대로 처리하는지, `response_format={"type":"json_object"}`가 OpenAI에서 동작하는지 보장 필요.

**Files:**
- Modify(필요 시): `src/news_impact/llm_client.py`, `src/news_impact/llm_config.py`
- Test: `tests/test_news_impact_llm_config.py`, `tests/test_news_impact_llm_cache.py`

**Interfaces / 확인 항목:**
- `load_llm_config`가 `configs/news_impact.openai.example.json`에서 provider=openai, api_key는 `OPENAI_API_KEY`(또는 `LLM_API_KEY`) env에서 해석되는지(이미 `_api_key_value`가 처리 — 테스트로 고정).
- `UrllibJsonTransport`가 `_auth_headers`로 `Authorization: Bearer`를 모든 요청에 부착하는지(가짜 전송으로 헤더 캡처).
- `verify_model_alias`: OpenAI `/models` data 리스트에 설정 모델이 있으면 통과, 없으면 `LLMModelAliasError` 명확. (대형 응답에서도 동작 — 현 로직 OK, 테스트로 고정.)

- [ ] **Step 1: 실패하는 테스트 작성** — 가짜 `JsonTransport`로 OpenAI 응답·헤더를 모사하는 케이스 추가(provider=openai 해석, bearer 헤더, alias 검증).
- [ ] **Step 2: 구현/수정** — 테스트가 드러낸 갭만 최소 수정.
- [ ] **Step 3: 검증** — `pytest tests/test_news_impact_llm_config.py tests/test_news_impact_llm_cache.py -v --basetemp=.tmp_pytest`.

---

### Task 4: 챗봇 런타임 배선

**문제:** `PipelineRuntimeConfig.build_command`(runtime_config.py:45)는 `--openai-model`·`--news-impact-llm-config`만 전달한다. 단일 `--llm-config` 스위치를 챗봇에서도 노출해야 한다.

**Files:**
- Modify: `src/chatbot/runtime_config.py`
- Test: `tests/test_kakao_colab_bot.py`

**Interfaces:**
- `PipelineRuntimeConfig.llm_config: str | None = None` 추가. `build_command`가 값 있으면 `--llm-config <path>` 추가.
- `build_subprocess_env`는 이미 `OPENAI_API_KEY`를 자식 env로 전달(변경 불필요, 회귀 테스트만).

- [ ] **Step 1: 실패하는 테스트** — `llm_config` 설정 시 argv에 `--llm-config`가 들어가고, `OPENAI_API_KEY`가 env로만 전달(argv 비노출)되는지.
- [ ] **Step 2: 구현.**
- [ ] **Step 3: 검증** — `pytest tests/test_kakao_colab_bot.py -v --basetemp=.tmp_pytest`.

---

### Task 5: 문서·운영 가이드·가드레일 재확인

**Files:**
- Modify: `docs/FULL_LOCAL_RUN_WITH_GEMMA_KAKAO.md`(또는 신규 `docs/LLM_PROVIDER_SWITCH.md`)

- [ ] **Step 1:** OpenAI 전환 사용법 문서화 —
  - 로컬 gemma(기본): 설정 없이 또는 `--llm-config configs/news_impact.gemma.example.json`.
  - OpenAI: `OPENAI_API_KEY` env 설정 후 `--llm-config configs/news_impact.openai.example.json`(기본 모델 `gpt-5-mini`). 키는 `.env`/env로만, argv 금지.
  - 하위호환 플래그(`--news-impact-llm-config`, `--openai-api-key/--openai-model`) 동작 표.
- [ ] **Step 2:** 가드레일 재명시 — provider 전환은 **표시/요약 품질·비용·지연**에만 영향, 예측값·랭킹 산식 불변. 비용 주의(OpenAI는 종목당 호출과금).
- [ ] **Step 3:** 전체 테스트 + 스모크 파이프라인 —
  - `pytest tests -v --basetemp=.tmp_pytest`
  - `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`

---

## 완료 기준 (Definition of Done)

- [ ] `--llm-config` 하나로 news_impact·issue_summary가 **동시에** gemma↔OpenAI 전환됨(명시적, 기본 gemma).
- [ ] issue_summary가 **로컬 gemma로도** LLM 요약 생성(키 불필요), OpenAI로도 동작(`gpt-5-mini` 기본).
- [ ] gemma 경로에서 `responses.create` 불필요 왕복 제거.
- [ ] 기존 플래그·호출부·테스트 전부 회귀 없음(기본값 보존).
- [ ] 시크릿 argv 비노출, CSV `utf-8-sig`, `result/` 외 산출 없음.
- [ ] `pytest tests` 그린 + 스모크 파이프라인 성공.
- [ ] 가드레일 보존: `predicted_return`·점수 산식·예측 입력 불변(표시 전용).
- [ ] 변경 커밋·푸시·PR 생성 후 `/pull/<번호>` URL 제공.
