# 로컬 Gemma 요약·뉴스임팩트 지연 최적화 — 분석 + 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans 로 §5 구현 플랜을 태스크 단위로 구현하세요. 각 단계는 체크박스(`- [ ]`) 문법으로 추적합니다.

작성일: 2026-06-25 · 통합본(분석 #341 + 구현 플랜 #342) · 최신화: 2026-06-25 (코드 라인/`provider`·`base_url` 인자 반영)

**Goal:** 로컬 gemma 기반 종목 이슈 요약·뉴스임팩트 판정의 LLM 호출 수·토큰·캐시 동작을 고쳐 재실행 지연을 대폭 줄인다(점수 산식·예측 입력은 불변).

**Architecture:** 본 문서 **§2 근본 원인**의 우선순위(P0 캐시 안정화 → P0 팬아웃 차단 → P1 토큰상한·요약통합 → P2 본문절단)를 따라, 콘텐츠 해시 캐시를 **안정 디렉터리**로 옮기고, 미매핑 기사의 전체-워치리스트 팬아웃을 제거하며, 출력 토큰 상한과 요약 1회화로 호출당·호출수 비용을 동시에 낮춘다. 모든 변경은 기존 표시/예측 분리 가드를 보존한다. §1~§3은 **왜·무엇을 먼저** 고치는지(진단), §4~§5는 **어디서·어떻게**(실행)다.

**Tech Stack:** Python 3.14, pandas, 표준 라이브러리 `urllib`(llama.cpp OpenAI 호환 API), 기존 `src/news_impact/*` · `src/reports/issue_summary.py` 배관.

## Global Constraints

- **Python 버전:** 3.14.x — 신규 의존성 없음(표준 라이브러리만 사용).
- **테스트 실행(이 PC):** `result/` 폴더 ACL deny를 피하려고 쓰기 가능한 basetemp 명시: `pytest <경로> -v --basetemp=.tmp_pytest`.
- **표시 vs 예측 가드(불변):** 뉴스/공시 임팩트는 **표시 전용**이다. 이 플랜은 호출 수·토큰·캐시·동시성만 바꾸며 점수 산식과 예측 입력 분리(`참고용·예측값 미반영`)는 절대 건드리지 않는다.
- **하위 계약 보존:** `DailyPipelineInputs`·`append_issue_summary_columns`·`LLMConfig`에 추가하는 모든 신규 인자/필드는 **기본값을 두어** 기존 호출부·테스트가 깨지지 않게 한다.
- **`provider`/`base_url` 인자 보존:** OpenAI provider 옵션 작업 이후 `issue_summary` 호출부·헬퍼(`_llm_symbol_issue_summary`, `_call_llm_json`, `_call_llm_text`, `append_issue_summary_columns`)에 `provider`/`base_url` 인자가 추가돼 있다. 아래 Task의 호출부 수정은 **이 인자들을 제거하지 말고 그대로 둔 채** `llm_cache_dir` 등만 추가한다.
- **캐시 안정 경로:** 뉴스임팩트 LLM 캐시 기본 경로는 `result/runtime/llm_cache/news_impact`(cwd 상대 — 기존 `result/runtime` 관례와 일치). 이슈 요약은 `result/runtime/llm_cache/issue_summary`.
- **PR 생성:** 제목·본문 자동 작성 후 `& "C:\Program Files\GitHub CLI\gh.exe" pr create`로 직접 생성, 결과 `/pull/<번호>` URL 제공.

---

## 1. 배경 분석 — 느린 경로는 두 갈래다

체감하는 "gemma 처리"는 서로 다른 두 LLM 서브시스템이다. 각각의 호출 수·토큰·캐시 동작이 다르므로 분리해서 본다.

| 경로 | 코드 | 호출 형태 | gemma 연결 |
|---|---|---|---|
| **(A) 뉴스/공시 임팩트 판정** | `src/news_impact/pipeline.py:_build_llm_judged_events`(L273) → `LlamaCppClient.chat_json` | **뉴스 N × 종목 M 중첩 루프, 완전 순차** | 직접(`localhost:8001`) |
| **(B) 종목 이슈 요약** | `src/reports/issue_summary.py:_llm_symbol_issue_summary`(L403) | **종목당 2회 순차**(공시 요약 + 뉴스 요약) | OpenAI 호환 클라(`provider=llama_cpp`+`base_url`로 gemma 지정 시) |

두 경로 모두 **호출 1건당 수 초~수십 초**가 드는 로컬 26B(MoE A4B) 추론을, **불필요하게 많이·길게·반복** 호출하는 것이 지연의 본질이다. 아래 원인을 임팩트 순으로 정리한다.

---

## 2. 근본 원인 (임팩트 순)

### 🔴 P0-1. 응답 캐시가 매 실행마다 폐기된다 (가장 큰 숨은 비용)

뉴스임팩트 gemma 경로는 `output_dir`을 **임시 디렉터리**로 잡고, LLM 응답 캐시를 그 하위에 만든다.

- `src/reports/news_impact_context.py:219` — `tempfile.TemporaryDirectory(prefix="news_impact_gemma_")`를 `output_dir`로 전달
- `src/news_impact/pipeline.py:388` — `_build_impact_judge_llm`이 캐시를 `output_dir / "llm_cache" / "impact_judgments"`에 생성

→ `with` 블록을 빠져나오는 순간 캐시 디렉터리가 통째로 삭제된다. **동일 기사·동일 종목 판정을 매 요청마다 처음부터 재계산**한다. `FileLLMResponseCache`는 콘텐츠 해시 기반(`llm_client.py:_cache_key`)이라, 안정 경로에만 두면 같은 기사는 즉시 히트한다.

이슈 요약 경로(B)도 마찬가지다. `append_issue_summary_columns`는 `llm_cache_dir` 인자를 지원하지만(`issue_summary.py:682`, 캐시 적용 `:719-722`), **운영 호출부 두 곳이 모두 이 인자를 넘기지 않는다**:
- `src/pipeline.py:792-801`
- `src/chatbot/kakao_colab_bot.py:1336-1345`(타임아웃 백그라운드), `:1360-1366`(동기)

→ 이슈 요약 LLM 캐시도 사실상 **죽어 있다**. 매 실행이 전량 재요약.

**효과:** 반복 실행(같은 종목/같은 날)에서 LLM 호출을 0에 가깝게 줄임. 단일 변경으로 재실행 지연이 가장 크게 떨어진다. → **Task 1·2**

---

### 🔴 P0-2. 뉴스 N × 종목 M 팬아웃 + 완전 순차

`_build_llm_judged_events`(`pipeline.py:273-342`)는 기사마다 대상 종목을 순회하며 종목별로 gemma를 호출한다(호출부 `:286`). 그리고 대상 종목 선정 로직이 폭발한다:

```python
# pipeline.py:345-348
def _target_tickers_for_news(item: NewsItem, watchlist_tickers: list[str]) -> list[str]:
    if item.ticker and item.ticker in watchlist_tickers:
        return [item.ticker]
    return watchlist_tickers   # ← 매핑 안 된 기사는 "전체 워치리스트"로 팬아웃
```

→ 종목이 특정되지 않은 시황/섹터 기사 1건이 **워치리스트 전 종목 수만큼 gemma 판정**을 유발한다. 워치리스트 W종목 × 미매핑 기사 K건 = **K×W회 호출, 전부 순차**. 게다가 같은 기사 본문을 종목 라인만 바꿔 매번 풀 프리필한다.

**개선 방향:**
1. **사전 매핑으로 팬아웃 차단** — 기사 본문/제목에 회사명이 등장하는 종목만 대상으로 좁힌다(`companies`는 이미 `_build_llm_judged_events` 스코프에 있음). 무관 종목 판정 자체를 없앤다. 0매칭(시황 등)이면 기존처럼 전체로 폴백. → **Task 4**
2. **(후속) 기사 단위 1회 판정 후 귀속** — 임팩트 방향/점수는 대부분 기사 본문이 결정한다. 기사(또는 클러스터)당 1회만 판정하고 종목에 귀속하면 추가로 W배 절감.
3. **(후속) 판정 전 의미 클러스터링으로 중복 기사 병합** — 현재 `_assign_unique_news_cluster_ids`(`:351`)는 기사마다 고유 클러스터를 부여(중복 제거 효과 없음). 판정 **전에** 군집화하면 같은 사건의 중복 기사를 1회로 합칠 수 있다.

**효과:** 호출 수를 수배~수십 배 절감. 로컬처럼 호출 1건이 비싼 환경에서 가장 직접적인 레버.

---

### 🟠 P1-1. 임팩트 판정에 출력 토큰 상한이 없다

`LlamaCppClient.chat_json`(`llm_client.py:242-251`)의 페이로드에는 `temperature`만 있고 **`max_tokens`/`n_predict`가 없다**. 디코딩 시간은 출력 토큰 수에 거의 비례하는데, 요구 JSON에는 자유서술 필드(`reason`, `why_may_be_wrong`)가 포함되어(`impact_judge.py:LLM_REQUIRED_KEYS`) 모델이 장황해질 수 있다.

**개선:** 페이로드에 `max_tokens`(예: 200~300) 추가. 이슈 요약은 `max_output_tokens=700`(`issue_summary.py:461,466`)으로 잡혀 있는데 불릿 요약엔 과하므로 통합 호출에서 600 이하로 축소. → **Task 3**(임팩트), **Task 5**(요약 통합 시 함께)

---

### 🟠 P1-2. 이슈 요약은 종목당 2회 호출 — 1회로 합칠 수 있다

`_llm_symbol_issue_summary`(`issue_summary.py:403`)는 공시 요약·뉴스 요약을 **순차 2회** 호출한다(`:469`, `:472`).

**개선:** 단일 프롬프트로 `{disclosure_summary, news_summary}`를 한 번에 받는 JSON 호출로 통합(이미 `_call_llm_json`(`:250`) + `response_format=json_object` 경로가 `provider` 인지 호출을 지원). 종목당 호출 절반으로. → **Task 5**

---

### 🟠 P1-3. 클라이언트 순차 호출 ↔ 서버 단일 슬롯 (비코드/운영)

임팩트 판정은 완전 순차이고, 이슈 요약은 `summary_n_jobs`로 스레드풀을 지원하지만 **기본값 1**이다. 동시성으로 이득을 보려면 **양쪽을 동시에** 맞춰야 한다:

- **서버:** `llama-server`를 `--parallel N --cont-batching`(+ 충분한 `-c` 컨텍스트)으로 기동해 슬롯 N개를 만든다. 현재 기동은 `-a gemma-4-26b-a4b --port 8001`만으로(메모리 기준) 단일 슬롯·연속배치 없음.
- **클라이언트:** 동시 요청 발행(임팩트 판정도 `ThreadPoolExecutor`화, 이슈 요약은 `summary_n_jobs` 상향).

**주의(로컬 특성):** GPU 슬롯이 남을 때는 연속배치가 처리량을 거의 선형으로 올린다. 그러나 **순수 CPU/컴퓨트 바운드**면 슬롯을 늘려도 총 처리량은 거의 그대로다 — 이때는 "호출 수·토큰 줄이기 + 프리픽스 캐시"가 더 효과적이다. 즉 P1-3은 **GPU 여유가 있을 때만** 큰 레버. → **부록 A**

---

### 🟡 P2-1. 시스템 프롬프트 프리필 반복 / 기사 길이

임팩트 판정 시스템 프롬프트는 약 **2,832자**(`src/news_impact/prompts/news_impact_llm_prompt.md` + 가드)로 매 호출 동일하게 전송된다.

- llama-server는 슬롯별 **최장 공통 프리픽스**를 재사용(`cache_prompt` 기본 on)하므로, 같은 슬롯에서 시스템 프롬프트를 맨 앞에 바이트 동일로 두면 재프리필을 피한다(현 코드는 이미 시스템 프롬프트를 첫 메시지로 둠 — 유지할 것).
- 다만 **기사 본문은 호출마다 달라** 프리필이 불가피하다. 본문을 **리드 + 핵심 문장 위주로 1,000~1,500자 절단**하면 프리필 토큰이 줄어 임팩트 판정·요약 모두 빨라진다. → **Task 6**
- 연속배치(다중 슬롯) 사용 시 슬롯 간 프리픽스 공유는 제한적이므로 P1-3과 트레이드오프 고려.

---

### 🟡 P2-2. 모델·서버 플래그 / 태스크 티어링 (비코드/운영)

26B(MoE) Q4가 **부분 오프로드로 CPU에 흘러내리면** 디코딩이 급락한다(메모리상 과거 OOM/한글경로 기동 이슈 이력). 점검·튜닝 항목:

- `-ngl 99`로 가능한 한 전부 GPU 오프로드(VRAM 한계 확인). `-fa`(flash attention), `-b`/`-ub`(배치/유배치) 상향으로 프리필 가속, `-t`는 물리 코어 수.
- **태스크 티어링:** 요약·클러스터 라벨처럼 쉬운 작업은 **소형 gemma(2B/4B)**로, 미묘한 임팩트 판정만 26B로. 요약을 소형 모델로 돌리면 수 배 빨라진다.
- 서버는 8001에 상주(모델 재로딩 없음) — 유지. → **부록 A**

---

## 3. 우선순위 요약 (구현 상태)

| 순위 | 항목 | 변경 위치 | 기대 효과 | 난이도 | 상태 |
|---|---|---|---|---|---|
| **P0-1** | 응답 캐시를 안정 디렉터리로 | `news_impact_context.py:219`, `pipeline.py:388·99`, `pipeline.py:792`, `kakao_colab_bot.py:1336·1360` | 반복 실행 지연 대폭↓ | 낮음 | ⬜ Task 1·2 |
| **P0-2** | N×M 팬아웃 차단(회사명 사전 매핑) | `pipeline.py:286·345` | 호출 수 수배~수십배↓ | 중 | ⬜ Task 4 |
| **P1-1** | 임팩트 출력 토큰 상한 | `llm_config.py:24·61`, `llm_client.py:251` | 호출당 디코딩↓ | 낮음 | ⬜ Task 3 |
| **P1-2** | 이슈 요약 2→1 호출 통합 | `issue_summary.py:403·469-477` | 요약 호출 절반 | 낮음 | ⬜ Task 5 |
| **P1-3** | 서버 `--parallel --cont-batching` + 클라 동시성 | 서버 기동, `pipeline.py`/`issue_summary.py` | GPU 여유 시 처리량↑ | 중 | ⬜ 부록 A(비코드) |
| **P2-1** | 기사 본문 절단 / 프리픽스 캐시 유지 | `pipeline.py:358` | 프리필 토큰↓ | 낮음 | ⬜ Task 6 |
| **P2-2** | GPU 오프로드 점검 + 태스크 티어링 | 서버 기동 / 설정 | 호출당 latency↓ | 중 | ⬜ 부록 A(비코드) |

> 현재(2026-06-25 기준) 코드 상태: **위 코드 항목(P0-1·P0-2·P1-1·P1-2·P2-1) 모두 미구현.** 아래 §5 Task 1~7로 구현한다.

---

## 4. 파일 구조

- 수정: `src/news_impact/pipeline.py` — `DailyPipelineInputs`에 `llm_cache_dir` 추가, `_build_impact_judge_llm`이 안정 캐시 디렉터리 사용, `_target_tickers_for_news` 팬아웃 차단, `_llm_article_text_and_flags` 본문 절단.
- 수정: `src/reports/news_impact_context.py` — gemma 런타임이 안정 캐시 디렉터리를 `DailyPipelineInputs`에 전달.
- 수정: `src/news_impact/llm_config.py` · `src/news_impact/llm_client.py` — `max_tokens` 설정·전송.
- 수정: `src/reports/issue_summary.py` — 공시+뉴스 요약 1회 JSON 호출 통합(출력 토큰 상한 축소).
- 수정: `src/pipeline.py` · `src/chatbot/kakao_colab_bot.py` — 이슈 요약 호출부에 `llm_cache_dir` 전달(`provider`/`base_url` 인자는 유지).
- 테스트: `tests/test_news_impact_llm_cache.py`, `tests/test_news_impact_llm_config.py`, `tests/test_issue_summary.py`, `tests/test_news_impact_full_package.py`, `tests/test_pipeline_smoke.py`, `tests/test_kakao_colab_bot.py`(기존 파일에 케이스 추가).

---

## 5. 구현 플랜 (TDD)

### Task 1: 뉴스임팩트 LLM 캐시를 안정 디렉터리로 (P0-1a)

**문제:** gemma 런타임이 `output_dir`을 `tempfile.TemporaryDirectory`로 잡고(`news_impact_context.py:219`), 임팩트 판정 캐시를 그 하위에 만들어(`pipeline.py:388`) 매 실행 삭제 → 동일 기사 매번 재판정.

**Files:**
- Modify: `src/news_impact/pipeline.py` (`DailyPipelineInputs` L60-67, `_build_impact_judge_llm` L387-389, `run_daily_pipeline` L99-102)
- Modify: `src/reports/news_impact_context.py:230-239` (gemma 런타임 호출)
- Test: `tests/test_news_impact_full_package.py`

**Interfaces:**
- Produces: `DailyPipelineInputs.llm_cache_dir: str | Path | None = None`. `None`이면 기존처럼 `output_dir / "llm_cache" / "impact_judgments"`를, 값이 있으면 그 경로를 임팩트 판정 캐시 루트로 사용.
- Consumes(Task 외부): `append_llm_news_impact_context_with_runtime`가 안정 경로 `result/runtime/llm_cache/news_impact`를 기본으로 주입.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_news_impact_full_package.py 에 추가
from pathlib import Path
from src.news_impact.pipeline import DailyPipelineInputs


def test_daily_pipeline_inputs_accepts_stable_llm_cache_dir(tmp_path):
    inputs = DailyPipelineInputs(
        run_date="2026-06-25",
        watchlist_path=tmp_path / "wl.csv",
        company_master_path=tmp_path / "cm.csv",
        input_fixture_path=tmp_path / "fx.json",
        output_dir=tmp_path / "out",
        llm_cache_dir=tmp_path / "stable_cache",
    )
    assert Path(inputs.llm_cache_dir) == tmp_path / "stable_cache"
```

- [ ] **Step 2: 테스트를 돌려 실패 확인**

Run: `pytest tests/test_news_impact_full_package.py::test_daily_pipeline_inputs_accepts_stable_llm_cache_dir -v --basetemp=.tmp_pytest`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'llm_cache_dir'`

- [ ] **Step 3: 최소 구현 작성**

`src/news_impact/pipeline.py`의 `DailyPipelineInputs`에 필드 추가(`semantic_cluster_llm` 다음, L67 아래):

```python
    semantic_cluster_llm: SemanticClusterLLM | None = None
    llm_cache_dir: str | Path | None = None
```

`_build_impact_judge_llm`(L387-389) 시그니처·본문 교체:

```python
def _build_impact_judge_llm(
    llm_config: LLMConfig,
    output_dir: Path,
    llm_cache_dir: str | Path | None = None,
) -> LlamaCppClient:
    cache_root = (
        Path(llm_cache_dir)
        if llm_cache_dir is not None
        else output_dir / "llm_cache" / "impact_judgments"
    )
    return LlamaCppClient(llm_config, cache=FileLLMResponseCache(cache_root))
```

`run_daily_pipeline`의 호출부(L99-102)를 교체:

```python
    impact_judge_llm = inputs.impact_judge_llm or _build_impact_judge_llm(
        llm_config=llm_config,
        output_dir=output_dir,
        llm_cache_dir=inputs.llm_cache_dir,
    )
```

- [ ] **Step 4: 테스트를 돌려 통과 확인**

Run: `pytest tests/test_news_impact_full_package.py::test_daily_pipeline_inputs_accepts_stable_llm_cache_dir -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 5: gemma 런타임이 안정 경로를 주입하도록 수정**

`src/reports/news_impact_context.py`의 `DailyPipelineInputs(...)` 생성부(L230-239)에 `llm_cache_dir` 추가. (파일 상단 `import`에 `from pathlib import Path`가 있는지 확인하고 없으면 추가.)

```python
            result = _run_daily_pipeline(
                DailyPipelineInputs(
                    run_date=run_date,
                    watchlist_path=bundle.watchlist_path,
                    company_master_path=bundle.company_master_path,
                    input_fixture_path=bundle.fixture_path,
                    output_dir=tmp,
                    semantic_clustering=False,
                    llm_config_path=llm_config_path,
                    llm_cache_dir=Path("result/runtime/llm_cache/news_impact"),
                )
            )
```

- [ ] **Step 6: 전체 패키지 회귀 실행**

Run: `pytest tests/test_news_impact_full_package.py -v --basetemp=.tmp_pytest`
Expected: PASS (신규 + 기존)

- [ ] **Step 7: 커밋**

```bash
git add src/news_impact/pipeline.py src/reports/news_impact_context.py tests/test_news_impact_full_package.py
git commit -m "perf(news-impact): persist gemma judgment cache across runs"
```

---

### Task 2: 이슈 요약 캐시 호출부 연결 (P0-1b)

**문제:** `append_issue_summary_columns`는 `llm_cache_dir` 캐시를 지원하지만(`issue_summary.py:682`, `tests/test_issue_summary.py::test_append_issue_summary_columns_reuses_llm_cache`로 검증됨), 운영 호출부가 인자를 넘기지 않아 캐시가 비활성.

**Files:**
- Modify: `src/pipeline.py:792-801`
- Modify: `src/chatbot/kakao_colab_bot.py:1336-1345`(백그라운드) 및 `:1360-1366`(동기)
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: `append_issue_summary_columns(..., llm_cache_dir: str | Path | None = None)`(기존 시그니처, `issue_summary.py:682`).
- **주의:** 두 호출부의 기존 인자(`provider`/`base_url`/`summary_n_jobs` 등)는 그대로 둔다 — `llm_cache_dir`만 추가.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_pipeline_smoke.py`에 추가(안정 캐시 경로 헬퍼의 존재·반환값 검증):

```python
def test_issue_summary_cache_dir_is_stable_path():
    import src.pipeline as pipeline_mod

    assert pipeline_mod._issue_summary_cache_dir() == "result/runtime/llm_cache/issue_summary"
```

> 호출부에 실제로 `llm_cache_dir` 키워드가 전달되는지는 기존 `test_pipeline_smoke.py`의 스모크 테스트가 `append_issue_summary_columns`를 `**kwargs` 흡수 가짜로 monkeypatch하므로 회귀로 보장된다(신규 키워드가 들어가도 깨지지 않음). 이 단위 테스트는 헬퍼 계약만 고정한다.

- [ ] **Step 2: 테스트를 돌려 실패 확인**

Run: `pytest tests/test_pipeline_smoke.py::test_issue_summary_cache_dir_is_stable_path -v --basetemp=.tmp_pytest`
Expected: FAIL — `AttributeError: module 'src.pipeline' has no attribute '_issue_summary_cache_dir'`

- [ ] **Step 3: 최소 구현 작성**

`src/pipeline.py`에 헬퍼 추가(파일 상단 import에 `from pathlib import Path` 있음):

```python
def _issue_summary_cache_dir() -> str:
    return "result/runtime/llm_cache/issue_summary"
```

`append_issue_summary_columns` 호출(L792-801)에 한 줄 추가(기존 `provider`/`base_url`/`summary_n_jobs` 유지):

```python
        pred_df = append_issue_summary_columns(
            pred_df,
            context_raw_df=context_raw_df,
            openai_api_key=effective_openai_api_key,
            openai_model=effective_openai_model,
            provider=issue_summary_provider,
            base_url=issue_summary_base_url,
            summarize_symbols=issue_summary_symbols,
            summary_n_jobs=issue_summary_n_jobs,
            llm_cache_dir=_issue_summary_cache_dir(),
        )
```

- [ ] **Step 4: 테스트를 돌려 통과 확인**

Run: `pytest tests/test_pipeline_smoke.py::test_issue_summary_cache_dir_is_stable_path -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 5: 챗봇 호출부에도 캐시 디렉터리 전달**

`src/chatbot/kakao_colab_bot.py`의 동기 경로(L1360-1366)에 `llm_cache_dir` 추가:

```python
                summarized_df = append_issue_summary_columns(
                    base,
                    context_raw_df=same_day.copy(),
                    openai_api_key=self.runtime_config.openai_api_key,
                    openai_model=self.runtime_config.openai_model,
                    summarize_symbols=[symbol],
                    llm_cache_dir="result/runtime/llm_cache/issue_summary",
                )
```

타임아웃 백그라운드 경로(L1336-1345)의 `self._run_in_background_with_timeout(append_issue_summary_columns, base, ...)` 호출 키워드 인자에도 동일하게 추가:

```python
                    summarize_symbols=[symbol],
                    llm_cache_dir="result/runtime/llm_cache/issue_summary",
                )
```

- [ ] **Step 6: 챗봇 회귀 실행**

Run: `pytest tests/test_kakao_colab_bot.py -v --basetemp=.tmp_pytest`
Expected: PASS (기존 테스트가 `append_issue_summary_columns`를 monkeypatch로 가짜 대체 — `**kwargs` 흡수하므로 신규 키워드 안전)

- [ ] **Step 7: 커밋**

```bash
git add src/pipeline.py src/chatbot/kakao_colab_bot.py tests/test_pipeline_smoke.py
git commit -m "perf(issue-summary): enable response cache at pipeline and bot call sites"
```

---

### Task 3: 임팩트 판정 출력 토큰 상한 (P1-1)

**문제:** `LlamaCppClient.chat_json` 페이로드에 `max_tokens`가 없어(`llm_client.py:242-251`) 자유서술 필드(`reason`, `why_may_be_wrong`)로 디코딩이 길어질 수 있음.

**Files:**
- Modify: `src/news_impact/llm_config.py` (`LLMConfig` L16-24, `load_llm_config` L49-62)
- Modify: `src/news_impact/llm_client.py` (`chat_json` L242-251)
- Modify: `configs/news_impact.gemma.example.json`
- Test: `tests/test_news_impact_llm_config.py`, `tests/test_news_impact_llm_cache.py`

**Interfaces:**
- Produces: `LLMConfig.max_tokens: int | None = None`. `None`이면 페이로드에 `max_tokens` 미포함(기존 동작 유지), 정수면 `payload["max_tokens"]`로 전송.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_news_impact_llm_cache.py`에 추가(전송 페이로드 캡처용 가짜 transport):

```python
from src.news_impact.llm_client import LlamaCppClient
from src.news_impact.llm_config import LLMConfig


class _CapturingTransport:
    def __init__(self):
        self.last_payload = None

    def get_json(self, url, timeout_seconds):
        return {"data": [{"id": "gemma-4-26b-a4b"}]}

    def post_json(self, url, payload, timeout_seconds):
        self.last_payload = payload
        return {
            "model": "gemma-4-26b-a4b",
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
        }


def test_chat_json_sends_max_tokens_when_configured():
    config = LLMConfig(
        provider="llama_cpp",
        base_url="http://localhost:8001/v1",
        model="gemma-4-26b-a4b",
        temperature=0.1,
        max_retries=0,
        json_schema_required=True,
        max_tokens=256,
    )
    transport = _CapturingTransport()
    client = LlamaCppClient(config, transport=transport)
    client.chat_json("sys", "user")
    assert transport.last_payload["max_tokens"] == 256


def test_chat_json_omits_max_tokens_when_unset():
    config = LLMConfig.default()
    transport = _CapturingTransport()
    client = LlamaCppClient(config, transport=transport)
    client.chat_json("sys", "user")
    assert "max_tokens" not in transport.last_payload
```

> `LlamaCppClient(config, transport=...)` 주입 지점이 현재 시그니처와 다르면, 테스트의 transport 주입 방식만 실제 생성자에 맞춰 조정한다(페이로드 캡처 의도는 동일).

- [ ] **Step 2: 테스트를 돌려 실패 확인**

Run: `pytest tests/test_news_impact_llm_cache.py::test_chat_json_sends_max_tokens_when_configured tests/test_news_impact_llm_cache.py::test_chat_json_omits_max_tokens_when_unset -v --basetemp=.tmp_pytest`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'max_tokens'`

- [ ] **Step 3: 최소 구현 작성**

`src/news_impact/llm_config.py`의 `LLMConfig`에 필드 추가(`timeout_seconds` 다음, L24 아래):

```python
    timeout_seconds: float = 60.0
    max_tokens: int | None = None
```

`load_llm_config`의 `return LLMConfig(...)` 끝(L61 `timeout_seconds=...` 다음)에 추가:

```python
        timeout_seconds=float(raw_config.get("timeout_seconds", default.timeout_seconds)),
        max_tokens=(
            int(raw_config["max_tokens"]) if raw_config.get("max_tokens") is not None else None
        ),
    )
```

`src/news_impact/llm_client.py`의 `chat_json`에서 `if self._config.json_schema_required:` 블록 아래(L251 다음, `cache_key` 계산 전)에 추가:

```python
        if self._config.json_schema_required:
            payload["response_format"] = {"type": "json_object"}
        if self._config.max_tokens is not None:
            payload["max_tokens"] = self._config.max_tokens
```

> `max_tokens`를 payload에 넣으면 `_cache_key`가 바뀌어 기존 캐시는 1회 무효화된다(설정이 출력에 영향을 주므로 의도된 동작).

- [ ] **Step 4: 테스트를 돌려 통과 확인**

Run: `pytest tests/test_news_impact_llm_cache.py::test_chat_json_sends_max_tokens_when_configured tests/test_news_impact_llm_cache.py::test_chat_json_omits_max_tokens_when_unset -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 5: 예시 설정에 상한 명시**

`configs/news_impact.gemma.example.json`에 `"max_tokens": 256` 키 추가(기존 JSON 객체에 한 줄 추가, 콤마 주의).

- [ ] **Step 6: 설정 로딩 회귀 실행**

Run: `pytest tests/test_news_impact_llm_config.py -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add src/news_impact/llm_config.py src/news_impact/llm_client.py configs/news_impact.gemma.example.json tests/test_news_impact_llm_cache.py
git commit -m "perf(news-impact): cap impact-judge output tokens via config"
```

---

### Task 4: 미매핑 기사의 전체-워치리스트 팬아웃 차단 (P0-2)

**문제:** `_target_tickers_for_news`가 ticker 없는 기사를 **전체 워치리스트**로 팬아웃해(`pipeline.py:345-348`) 기사 1건이 종목 수만큼 순차 판정을 유발. 가장 큰 호출수 레버.

**접근:** 기사 본문/제목에 회사명이 등장하는 종목만 대상으로 좁힌다. 매칭이 0건이면(시황 등) 기존처럼 전체로 폴백. `companies`는 이미 `_build_llm_judged_events`(L277) 스코프에 있다.

**Files:**
- Modify: `src/news_impact/pipeline.py` (`_build_llm_judged_events` 호출부 L286, `_target_tickers_for_news` L345-348)
- Test: `tests/test_news_impact_full_package.py`

**Interfaces:**
- Produces: `_target_tickers_for_news(item, watchlist_tickers, companies=None) -> list[str]`. 종목별 회사명(`companies[ticker]["company"]`)이 기사 `title`/`raw_text`/`summary`에 포함되면 그 종목만 반환. 매칭 0건이면 전체 워치리스트 반환(기존 폴백 보존).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_news_impact_full_package.py 에 추가
from types import SimpleNamespace
from src.news_impact.pipeline import _target_tickers_for_news


def _news_stub(title, summary="", raw_text="", ticker=""):
    # _target_tickers_for_news는 ticker/title/raw_text/summary 속성만 읽으므로
    # 제약 많은 NewsItem 대신 가벼운 스텁으로 검증한다.
    return SimpleNamespace(title=title, summary=summary, raw_text=raw_text, ticker=ticker)


def test_target_tickers_narrows_to_company_name_match():
    companies = {
        "005930": {"company": "삼성전자"},
        "000660": {"company": "SK하이닉스"},
    }
    item = _news_stub("삼성전자, 신규 HBM 수주")
    assert _target_tickers_for_news(item, ["005930", "000660"], companies) == ["005930"]


def test_target_tickers_falls_back_to_full_watchlist_when_no_match():
    companies = {"005930": {"company": "삼성전자"}, "000660": {"company": "SK하이닉스"}}
    item = _news_stub("코스피 외국인 순매수 전환")
    assert _target_tickers_for_news(item, ["005930", "000660"], companies) == ["005930", "000660"]
```

- [ ] **Step 2: 테스트를 돌려 실패 확인**

Run: `pytest tests/test_news_impact_full_package.py::test_target_tickers_narrows_to_company_name_match tests/test_news_impact_full_package.py::test_target_tickers_falls_back_to_full_watchlist_when_no_match -v --basetemp=.tmp_pytest`
Expected: FAIL — `TypeError: _target_tickers_for_news() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: 최소 구현 작성**

`src/news_impact/pipeline.py`의 `_target_tickers_for_news`(L345-348) 교체:

```python
def _target_tickers_for_news(
    item: NewsItem,
    watchlist_tickers: list[str],
    companies: dict[str, dict[str, str]] | None = None,
) -> list[str]:
    if item.ticker and item.ticker in watchlist_tickers:
        return [item.ticker]
    if companies:
        haystack = " ".join(
            str(part) for part in (item.title, getattr(item, "raw_text", ""), item.summary) if part
        )
        matched = [
            ticker
            for ticker in watchlist_tickers
            if (companies.get(ticker, {}).get("company") or "") and companies[ticker]["company"] in haystack
        ]
        if matched:
            return matched
    return watchlist_tickers
```

`_build_llm_judged_events`의 호출부(L286)를 교체:

```python
        for ticker in _target_tickers_for_news(item, watchlist_tickers, companies):
```

- [ ] **Step 4: 테스트를 돌려 통과 확인**

Run: `pytest tests/test_news_impact_full_package.py::test_target_tickers_narrows_to_company_name_match tests/test_news_impact_full_package.py::test_target_tickers_falls_back_to_full_watchlist_when_no_match -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 5: 전체 패키지 회귀 실행**

Run: `pytest tests/test_news_impact_full_package.py tests/test_news_impact_context.py -v --basetemp=.tmp_pytest`
Expected: PASS (기존 동작이 회사명 매칭/폴백으로 보존됨)

- [ ] **Step 6: 커밋**

```bash
git add src/news_impact/pipeline.py tests/test_news_impact_full_package.py
git commit -m "perf(news-impact): narrow news judging to name-matched tickers, keep market-news fallback"
```

---

### Task 5: 이슈 요약 2회 → 1회 JSON 호출 통합 (P1-2)

**문제:** `_llm_symbol_issue_summary`가 공시 요약·뉴스 요약을 **순차 2회** 호출(`issue_summary.py:469`, `:472`). 1회 JSON 호출로 통합해 종목당 호출 절반.

**Files:**
- Modify: `src/reports/issue_summary.py` (`_llm_symbol_issue_summary` L403-, 신규 통합 프롬프트)
- Test: `tests/test_issue_summary.py`

**Interfaces:**
- Consumes: 기존 `_call_llm_json(client, model, prompt, max_output_tokens=400, provider="openai") -> dict | None`(`issue_summary.py:250`). **`provider=provider_norm`를 반드시 전달**(gemma는 `chat.completions`+`json_object` 경로).
- Produces: 단일 호출이 `{"disclosure_summary": str, "news_summary": str}`를 반환. 누락 키는 기존 `_ensure_non_empty_issue_block` 폴백으로 보강.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_issue_summary.py 에 추가
import src.reports.issue_summary as issue_mod
import pandas as pd


def test_llm_symbol_issue_summary_uses_single_call(monkeypatch):
    calls = {"json": 0, "text": 0}

    def _fake_json(client, model, prompt, max_output_tokens=400, provider="openai"):
        calls["json"] += 1
        return {
            "disclosure_summary": "[공시 요약]\n- 공급계약 체결",
            "news_summary": "[뉴스 요약]\n- 메모리 수요 증가",
        }

    def _fake_text(*a, **k):
        calls["text"] += 1
        return "should-not-be-called"

    class _FakeOpenAI:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(issue_mod, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(issue_mod, "_call_llm_json", _fake_json)
    monkeypatch.setattr(issue_mod, "_call_llm_text", _fake_text)

    events = pd.DataFrame(
        [{"Date": "2026-06-25", "Symbol": "005930.KS", "source_type": "news",
          "title": "삼성전자 메모리 수요 증가", "published_at": "2026-06-25T09:00:00"}]
    )
    summary = issue_mod._llm_symbol_issue_summary(
        symbol="005930.KS", symbol_name="삼성전자", events=events, api_key="sk-test", model="gemma",
    )
    assert summary is not None
    assert calls["json"] == 1
    assert calls["text"] == 0
    assert "공급계약" in summary.disclosure_summary
    assert "메모리" in summary.news_summary
```

- [ ] **Step 2: 테스트를 돌려 실패 확인**

Run: `pytest tests/test_issue_summary.py::test_llm_symbol_issue_summary_uses_single_call -v --basetemp=.tmp_pytest`
Expected: FAIL — 현재는 `_call_llm_text`가 2회 호출되어 `calls["text"] != 0`

- [ ] **Step 3: 최소 구현 작성**

`src/reports/issue_summary.py`에 통합 프롬프트 상수 추가(`NEWS_SUMMARY_PROMPT` 아래):

```python
COMBINED_SUMMARY_PROMPT = """너는 한국 상장사의 공시와 뉴스를 투자 전문가에게 전달하기 위해 정리하는 AI다.
입력에 없는 정보는 추가하지 마라. 수치는 정확히 유지하라. 해석·전망·투자판단은 하지 마라.
반드시 JSON만 반환하라. 키는 disclosure_summary, news_summary 두 개다.
각 값은 '[공시 요약]' / '[뉴스 요약]' 헤더로 시작하고, 핵심 사실을 '- ' 불릿으로 최대 5줄 적는다.
핵심이 없으면 해당 값은 헤더 + '- 확인된 핵심 공시 내용 없음'(또는 뉴스) 한 줄로 한다."""
```

`_llm_symbol_issue_summary`에서 `disclosure_payload`/`news_payload` 빌더(L429-446)와 `_call_llm_text_with_provider` 래퍼·2회 호출 블록(L455-477)을 **단일 JSON 호출**로 교체. `fallback_disclosure_lines`/`fallback_news_lines`(L448-453)와 이후 `_ensure_non_empty_issue_block` 폴백(L479-492)은 그대로 유지:

```python
    combined_payload = json.dumps(
        {
            "symbol": symbol,
            "symbol_name": symbol_name,
            "date_kst": structured_payload.get("date_kst", ""),
            "disclosures": disclosures,
            "news_clusters": structured_payload.get("news_clusters", []),
        },
        ensure_ascii=False,
    )
    try:
        parsed = _call_llm_json(
            client,
            model,
            f"{COMBINED_SUMMARY_PROMPT}\n\n[입력 데이터]\n{combined_payload}",
            max_output_tokens=600,
            provider=provider_norm,
        ) or {}
    except Exception as exc:
        print(f"[ISSUE SUMMARY] LLM 요약 실패 ({symbol}): {type(exc).__name__}: {exc}")
        return None
    disclosure_summary = str(parsed.get("disclosure_summary") or "").strip() or None
    news_summary = str(parsed.get("news_summary") or "").strip() or None
```

- [ ] **Step 4: 테스트를 돌려 통과 확인**

Run: `pytest tests/test_issue_summary.py::test_llm_symbol_issue_summary_uses_single_call -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 5: 이슈 요약 전체 회귀 실행**

Run: `pytest tests/test_issue_summary.py -v --basetemp=.tmp_pytest`
Expected: PASS (캐시 재사용·요청종목 제한 등 기존 케이스 유지)

- [ ] **Step 6: 커밋**

```bash
git add src/reports/issue_summary.py tests/test_issue_summary.py
git commit -m "perf(issue-summary): merge disclosure+news into one LLM call"
```

---

### Task 6: 임팩트 판정 기사 본문 절단 (P2-1)

**문제:** 기사 본문이 길수록 프리필 토큰이 늘어 호출당 latency 증가. 본문을 앞부분 위주로 절단해 프리필을 줄인다(판정 정보의 대부분은 리드에 있음).

**Files:**
- Modify: `src/news_impact/pipeline.py` (`_llm_article_text_and_flags` L358-366)
- Test: `tests/test_news_impact_full_package.py`

**Interfaces:**
- Produces: `_llm_article_text_and_flags`가 반환하는 텍스트는 최대 `MAX_ARTICLE_CHARS`(=1500)자로 절단되고, 절단 시 `"article_truncated"` 플래그 추가.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_news_impact_full_package.py 에 추가
from types import SimpleNamespace
from src.news_impact.pipeline import _llm_article_text_and_flags, MAX_ARTICLE_CHARS


def test_article_text_is_truncated_with_flag():
    # _llm_article_text_and_flags는 quality_flags/raw_text/summary 속성만 읽는다.
    long_item = SimpleNamespace(
        quality_flags=(), raw_text="가" * (MAX_ARTICLE_CHARS + 500), summary=""
    )
    text, flags = _llm_article_text_and_flags(long_item)
    assert len(text) == MAX_ARTICLE_CHARS
    assert "article_truncated" in flags
```

- [ ] **Step 2: 테스트를 돌려 실패 확인**

Run: `pytest tests/test_news_impact_full_package.py::test_article_text_is_truncated_with_flag -v --basetemp=.tmp_pytest`
Expected: FAIL — `ImportError: cannot import name 'MAX_ARTICLE_CHARS'`

- [ ] **Step 3: 최소 구현 작성**

`src/news_impact/pipeline.py` 상단 상수 영역에 추가:

```python
MAX_ARTICLE_CHARS = 1500
```

`_llm_article_text_and_flags`(L358-366) 교체:

```python
def _llm_article_text_and_flags(item: NewsItem) -> tuple[str, tuple[str, ...]]:
    base_flags = tuple(str(flag) for flag in item.quality_flags)
    if item.raw_text:
        text = item.raw_text
        flags = base_flags + detect_prompt_injection(text)
    else:
        text = item.summary
        flags = base_flags + ("summary_only_no_full_text", "needs_full_text_review")
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS]
        flags = flags + ("article_truncated",)
    return text, _dedupe_flags(flags)
```

- [ ] **Step 4: 테스트를 돌려 통과 확인**

Run: `pytest tests/test_news_impact_full_package.py::test_article_text_is_truncated_with_flag -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 5: 전체 패키지 회귀 실행**

Run: `pytest tests/test_news_impact_full_package.py -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add src/news_impact/pipeline.py tests/test_news_impact_full_package.py
git commit -m "perf(news-impact): truncate long article text to cut prefill tokens"
```

---

### Task 7: 전체 회귀 + 문서 갱신

**Files:**
- Modify: `docs/2026-06-25-gemma-llm-latency-optimization.md` (본 문서 §3 표 구현 상태 갱신)

- [ ] **Step 1: 관련 테스트 전부 실행**

Run:
```bash
pytest tests/test_news_impact_full_package.py tests/test_news_impact_llm_cache.py tests/test_news_impact_llm_config.py tests/test_issue_summary.py tests/test_pipeline_smoke.py tests/test_kakao_colab_bot.py -v --basetemp=.tmp_pytest
```
Expected: 전부 PASS

- [ ] **Step 2: 스모크 파이프라인 실행**

Run: `python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json`
Expected: 정상 종료(외부 LLM 비활성 경로에서도 회귀 없음).

- [ ] **Step 3: 본 문서에 구현 상태 반영**

§3 우선순위 표의 구현 완료 행 상태를 `⬜`에서 `✅ (구현됨)`로 바꾸고 커밋 해시/PR 포인터를 한 줄 추가.

- [ ] **Step 4: 커밋**

```bash
git add docs/2026-06-25-gemma-llm-latency-optimization.md
git commit -m "docs: mark gemma latency optimizations as implemented"
```

---

## 부록 A: 운영 튜닝 체크리스트 (비코드 — TDD 대상 아님)

§2의 P1-3·P2-2는 코드가 아니라 `llama-server` 기동/하드웨어 설정이다. 별도로 추적·측정한다.

- [ ] **서버 연속 배치:** `llama-server`를 `--parallel N --cont-batching`(+ 충분한 `-c`)로 기동. GPU 슬롯 여유가 있을 때만 처리량↑. 클라이언트 동시성(`summary_n_jobs` 상향, 임팩트 판정 스레드풀화)과 함께여야 효과.
- [ ] **GPU 오프로드 점검:** `-ngl 99`로 가능한 한 전부 오프로드. 부분 오프로드로 CPU에 흘러내리면 디코딩 급락(과거 OOM/한글경로 기동 이슈 이력 참고).
- [ ] **프리필 가속:** `-fa`(flash attention), `-b`/`-ub` 상향, `-t`=물리 코어 수.
- [ ] **태스크 티어링:** 요약·클러스터 라벨은 소형 gemma(2B/4B)로, 임팩트 판정만 26B로. 요약 모델 분리 시 수 배 가속.
- [ ] **측정 우선:** 변경 전후로 (a) 임팩트 판정 호출 수, (b) 이슈 요약 호출 수, (c) 호출당 평균 latency, (d) 캐시 히트율을 로깅해 지배 경로와 개선폭을 정량 확인.

---

## 권장 실행 순서

1. **먼저 측정한다.** 1회 실행에서 (a) 임팩트 판정 호출 수, (b) 이슈 요약 호출 수, (c) 호출당 평균 latency, (d) 캐시 히트율을 로깅. 어느 경로가 지배적인지부터 확정한다(가설: 미매핑 기사가 많으면 P0-2가, 반복 실행이면 P0-1이 지배적).
2. **Task 1·2 (P0-1 캐시 안정화)** — 가장 싸고 즉효. 안정 경로로 캐시를 이동하고 호출부에 `llm_cache_dir` 전달.
3. **Task 4 (P0-2 팬아웃 차단)** — 회사명 매칭으로 무관 판정 제거(가장 큰 알고리즘 레버).
4. **Task 3·5 (P1)** — 토큰 상한·요약 통합(저난이도 즉효).
5. **Task 6 (P2-1)** — 본문 절단.
6. **(GPU 여유 확인 후) 부록 A** — 서버 연속배치 + 클라 동시성, 오프로드 점검, 태스크 티어링.

## 성공 기준 (Success Criteria)

1. §5의 모든 `pytest` 명령이 PASS.
2. 동일 종목/동일 날짜 **재실행** 시 임팩트 판정·이슈 요약 LLM 호출이 캐시 히트로 0에 수렴(`result/runtime/llm_cache/` 하위에 항목 생성·재사용 확인).
3. 미매핑 시황 기사가 더 이상 전체 워치리스트로 팬아웃하지 않음(회사명 매칭 종목만 판정, 0매칭이면 기존 폴백 유지).
4. 표시/예측 분리 가드·점수 산식 불변(뉴스/공시 영향은 표시 전용 유지).
