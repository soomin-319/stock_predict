# 로컬 Gemma 요약·뉴스임팩트 지연 최적화 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans 로 이 플랜을 태스크 단위로 구현하세요. 각 단계는 체크박스(`- [ ]`) 문법으로 추적합니다.

**Goal:** 로컬 gemma 기반 종목 이슈 요약·뉴스임팩트 판정의 LLM 호출 수·토큰·캐시 동작을 고쳐 재실행 지연을 대폭 줄인다(점수 산식·예측 입력은 불변).

**Architecture:** 분석 문서 `docs/GEMMA_LLM_LATENCY_OPTIMIZATION.md`의 우선순위(P0 캐시 안정화 → P0 팬아웃 차단 → P1 토큰상한·요약통합 → P2 본문절단)를 따라, 콘텐츠 해시 캐시를 **안정 디렉터리**로 옮기고, 미매핑 기사의 전체-워치리스트 팬아웃을 제거하며, 출력 토큰 상한과 요약 1회화로 호출당·호출수 비용을 동시에 낮춘다. 모든 변경은 기존 표시/예측 분리 가드를 보존한다.

**Tech Stack:** Python 3.14, pandas, 표준 라이브러리 `urllib`(llama.cpp OpenAI 호환 API), 기존 `src/news_impact/*` · `src/reports/issue_summary.py` 배관.

## Global Constraints

- **Python 버전:** 3.14.5 — 신규 의존성 없음(표준 라이브러리만 사용).
- **테스트 실행(이 PC):** `result/` 폴더 ACL deny를 피하려고 쓰기 가능한 basetemp 명시: `pytest <경로> -v --basetemp=.tmp_pytest`.
- **표시 vs 예측 가드(불변):** 뉴스/공시 임팩트는 **표시 전용**이다. 이 플랜은 호출 수·토큰·캐시·동시성만 바꾸며 점수 산식과 예측 입력 분리(`참고용·예측값 미반영`)는 절대 건드리지 않는다.
- **하위 계약 보존:** `DailyPipelineInputs`·`append_issue_summary_columns`·`LLMConfig`에 추가하는 모든 신규 인자/필드는 **기본값을 두어** 기존 호출부·테스트가 깨지지 않게 한다.
- **캐시 안정 경로:** 뉴스임팩트 LLM 캐시 기본 경로는 `result/runtime/llm_cache/news_impact`(cwd 상대 — 기존 `result/runtime` 관례와 일치). 이슈 요약은 `result/runtime/llm_cache/issue_summary`.
- **PR 생성:** 제목·본문 자동 작성 후 `& "C:\Program Files\GitHub CLI\gh.exe" pr create`로 직접 생성, 결과 `/pull/<번호>` URL 제공.

## 파일 구조

- 수정: `src/news_impact/pipeline.py` — `DailyPipelineInputs`에 `llm_cache_dir` 추가, `_build_impact_judge_llm`이 안정 캐시 디렉터리 사용, `_target_tickers_for_news` 팬아웃 차단.
- 수정: `src/reports/news_impact_context.py` — gemma 런타임이 안정 캐시 디렉터리를 `DailyPipelineInputs`에 전달.
- 수정: `src/news_impact/llm_config.py` · `src/news_impact/llm_client.py` — `max_tokens` 설정·전송.
- 수정: `src/reports/issue_summary.py` — 공시+뉴스 요약 1회 JSON 호출 통합, 출력 토큰 상한 축소.
- 수정: `src/pipeline.py` · `src/chatbot/kakao_colab_bot.py` — 이슈 요약 호출부에 `llm_cache_dir` 전달.
- 수정: `src/news_impact/mapper.py`(또는 신규 헬퍼) — 기사→관련 종목 사전 매핑(팬아웃 차단용).
- 테스트: `tests/test_news_impact_llm_cache.py`, `tests/test_news_impact_llm_config.py`, `tests/test_issue_summary.py`, `tests/test_news_impact_full_package.py`(기존 파일에 케이스 추가).

---

### Task 1: 뉴스임팩트 LLM 캐시를 안정 디렉터리로 (P0-1a)

**문제:** gemma 런타임이 `output_dir`을 `tempfile.TemporaryDirectory`로 잡고(`news_impact_context.py:219`), 임팩트 판정 캐시를 그 하위에 만들어(`pipeline.py:388`) 매 실행 삭제 → 동일 기사 매번 재판정.

**Files:**
- Modify: `src/news_impact/pipeline.py` (`DailyPipelineInputs`, `_build_impact_judge_llm`, `run_daily_pipeline`)
- Modify: `src/reports/news_impact_context.py:218-240` (gemma 런타임 호출)
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

`src/news_impact/pipeline.py`의 `DailyPipelineInputs`에 필드 추가(`semantic_cluster_llm` 다음 줄):

```python
    semantic_cluster_llm: SemanticClusterLLM | None = None
    llm_cache_dir: str | Path | None = None
```

`_build_impact_judge_llm` 시그니처와 본문 교체:

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
    cache = FileLLMResponseCache(cache_root)
    return LlamaCppClient(llm_config, cache=cache)
```

`run_daily_pipeline` 내부 호출부(`impact_judge_llm = inputs.impact_judge_llm or _build_impact_judge_llm(...)`)를 교체:

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

`src/reports/news_impact_context.py`의 `append_llm_news_impact_context_with_runtime`에서 `DailyPipelineInputs(...)` 생성부(현재 line 230-240)에 `llm_cache_dir` 추가. 파일 상단 `import`에 `from pathlib import Path`가 이미 있으므로 그대로 사용:

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

**문제:** `append_issue_summary_columns`는 `llm_cache_dir` 캐시를 지원하지만(`tests/test_issue_summary.py::test_append_issue_summary_columns_reuses_llm_cache`로 검증됨), 운영 호출부 두 곳이 인자를 넘기지 않아 캐시가 비활성.

**Files:**
- Modify: `src/pipeline.py:750-757`
- Modify: `src/chatbot/kakao_colab_bot.py:1335-1346` 및 `:1360-1366`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: `append_issue_summary_columns(..., llm_cache_dir: str | Path | None = None)`(기존 시그니처, `issue_summary.py:631`).

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

`src/pipeline.py`에 헬퍼 추가(파일 상단 import에 `from pathlib import Path`가 이미 있음):

```python
def _issue_summary_cache_dir() -> str:
    return "result/runtime/llm_cache/issue_summary"
```

`append_issue_summary_columns` 호출(line 750-757)에 인자 추가:

```python
        pred_df = append_issue_summary_columns(
            pred_df,
            context_raw_df=context_raw_df,
            openai_api_key=effective_openai_api_key,
            openai_model=effective_openai_model,
            summarize_symbols=issue_summary_symbols,
            summary_n_jobs=issue_summary_n_jobs,
            llm_cache_dir=_issue_summary_cache_dir(),
        )
```

- [ ] **Step 4: 테스트를 돌려 통과 확인**

Run: `pytest tests/test_pipeline_smoke.py::test_issue_summary_cache_dir_is_stable_path -v --basetemp=.tmp_pytest`
Expected: PASS

- [ ] **Step 5: 챗봇 호출부에도 캐시 디렉터리 전달**

`src/chatbot/kakao_colab_bot.py`의 동기 경로(line 1360-1366)에 `llm_cache_dir` 추가:

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

타임아웃 백그라운드 경로(line 1335-1346)의 `self._run_in_background_with_timeout(append_issue_summary_columns, base, ...)` 호출에도 동일 키워드 인자 추가:

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

**문제:** `LlamaCppClient.chat_json` 페이로드에 `max_tokens`가 없어(`llm_client.py:242-249`) 자유서술 필드(`reason`, `why_may_be_wrong`)로 디코딩이 길어질 수 있음.

**Files:**
- Modify: `src/news_impact/llm_config.py` (`LLMConfig`, `load_llm_config`)
- Modify: `src/news_impact/llm_client.py` (`chat_json`)
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

- [ ] **Step 2: 테스트를 돌려 실패 확인**

Run: `pytest tests/test_news_impact_llm_cache.py::test_chat_json_sends_max_tokens_when_configured tests/test_news_impact_llm_cache.py::test_chat_json_omits_max_tokens_when_unset -v --basetemp=.tmp_pytest`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'max_tokens'`

- [ ] **Step 3: 최소 구현 작성**

`src/news_impact/llm_config.py`의 `LLMConfig`에 필드 추가(`timeout_seconds` 다음):

```python
    timeout_seconds: float = 60.0
    max_tokens: int | None = None
```

`load_llm_config`의 `return LLMConfig(...)`에 추가:

```python
        timeout_seconds=float(raw_config.get("timeout_seconds", default.timeout_seconds)),
        max_tokens=(
            int(raw_config["max_tokens"]) if raw_config.get("max_tokens") is not None else None
        ),
    )
```

`src/news_impact/llm_client.py`의 `chat_json`에서 payload 구성 직후(`if self._config.json_schema_required:` 블록 아래)에 추가:

```python
        if self._config.json_schema_required:
            payload["response_format"] = {"type": "json_object"}
        if self._config.max_tokens is not None:
            payload["max_tokens"] = self._config.max_tokens
```

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

**접근:** 기사 본문/제목에 종목명·티커가 등장하는 종목만 대상으로 좁힌다. 매칭이 0건이면(시황 등) 기존처럼 전체로 폴백하되, 그 동작은 **설정 플래그로 제어**해 점수 표시 회귀를 통제한다.

**Files:**
- Modify: `src/news_impact/pipeline.py` (`_build_llm_judged_events`, `_target_tickers_for_news`)
- Test: `tests/test_news_impact_full_package.py`

**Interfaces:**
- Produces: `_target_tickers_for_news(item, watchlist_tickers, companies) -> list[str]`. 종목별 회사명(`companies[ticker]["company"]`)이 기사 `title`/`raw_text`/`summary`에 포함되면 그 종목만 반환. 매칭 0건이면 전체 워치리스트 반환(기존 폴백 보존).

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

`src/news_impact/pipeline.py`의 `_target_tickers_for_news` 교체:

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

`_build_llm_judged_events`의 호출부(`for ticker in _target_tickers_for_news(item, watchlist_tickers):`)를 교체:

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

**문제:** `_llm_symbol_issue_summary`가 공시 요약·뉴스 요약을 **순차 2회** 호출(`issue_summary.py:424-435`). 1회 JSON 호출로 통합해 종목당 호출 절반.

**Files:**
- Modify: `src/reports/issue_summary.py` (`_llm_symbol_issue_summary`, 신규 통합 프롬프트)
- Test: `tests/test_issue_summary.py`

**Interfaces:**
- Consumes: 기존 `_call_llm_json(client, model, prompt, max_output_tokens) -> dict | None`(`issue_summary.py:240`).
- Produces: 단일 호출이 `{"disclosure_summary": str, "news_summary": str}`(및 선택 키)를 반환. 누락 키는 기존 `_ensure_non_empty_issue_block` 폴백으로 보강.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_issue_summary.py 에 추가
import src.reports.issue_summary as issue_mod
import pandas as pd


def test_llm_symbol_issue_summary_uses_single_call(monkeypatch):
    calls = {"json": 0, "text": 0}

    def _fake_json(client, model, prompt, max_output_tokens=400):
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

`_llm_symbol_issue_summary`의 두 `_call_llm_text` 호출 블록(line 423-438)을 단일 JSON 호출로 교체:

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
        ) or {}
    except Exception as exc:
        print(f"[ISSUE SUMMARY] LLM 요약 실패 ({symbol}): {type(exc).__name__}: {exc}")
        return None
    disclosure_summary = str(parsed.get("disclosure_summary") or "").strip() or None
    news_summary = str(parsed.get("news_summary") or "").strip() or None
```

이후 기존 `_ensure_non_empty_issue_block(...)` 폴백 블록(line 440-453)은 그대로 유지한다.

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
- Modify: `src/news_impact/pipeline.py` (`_llm_article_text_and_flags`)
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

`src/news_impact/pipeline.py` 상단 상수 영역(`RULE_BASED_FLAGS` 근처)에 추가:

```python
MAX_ARTICLE_CHARS = 1500
```

`_llm_article_text_and_flags` 교체:

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
- Modify: `docs/GEMMA_LLM_LATENCY_OPTIMIZATION.md` (구현 완료 항목 표식)

- [ ] **Step 1: 관련 테스트 전부 실행**

Run:
```bash
pytest tests/test_news_impact_full_package.py tests/test_news_impact_llm_cache.py tests/test_news_impact_llm_config.py tests/test_issue_summary.py tests/test_pipeline_smoke.py tests/test_kakao_colab_bot.py -v --basetemp=.tmp_pytest
```
Expected: 전부 PASS

- [ ] **Step 2: 분석 문서에 구현 상태 반영**

`docs/GEMMA_LLM_LATENCY_OPTIMIZATION.md`의 §3 우선순위 표에서 구현 완료한 P0-1/P0-2/P1-1/P1-2/P2-1 행에 `✅ (구현됨, 이 플랜)` 표식과 본 플랜 경로 한 줄 포인터를 추가.

- [ ] **Step 3: 커밋**

```bash
git add docs/GEMMA_LLM_LATENCY_OPTIMIZATION.md
git commit -m "docs: mark gemma latency optimizations as implemented"
```

---

## 부록 A: 운영 튜닝 체크리스트 (비코드 — TDD 대상 아님)

분석의 P1-3·P2-2는 코드가 아니라 `llama-server` 기동/하드웨어 설정이다. 별도로 추적·측정한다.

- [ ] **서버 연속 배치:** `llama-server`를 `--parallel N --cont-batching`(+ 충분한 `-c`)로 기동. GPU 슬롯 여유가 있을 때만 처리량↑. 클라이언트 동시성(`issue_summary_n_jobs` 상향, 임팩트 판정 스레드풀화)과 함께여야 효과.
- [ ] **GPU 오프로드 점검:** `-ngl 99`로 가능한 한 전부 오프로드. 부분 오프로드로 CPU에 흘러내리면 디코딩 급락(과거 OOM/한글경로 기동 이슈 이력 참고).
- [ ] **프리필 가속:** `-fa`(flash attention), `-b`/`-ub` 상향, `-t`=물리 코어 수.
- [ ] **태스크 티어링:** 요약·클러스터 라벨은 소형 gemma(2B/4B)로, 임팩트 판정만 26B로. 요약 모델 분리 시 수 배 가속.
- [ ] **측정 우선:** 변경 전후로 (a) 임팩트 판정 호출 수, (b) 이슈 요약 호출 수, (c) 호출당 평균 latency, (d) 캐시 히트율을 로깅해 지배 경로와 개선폭을 정량 확인.

## 성공 기준 (Success Criteria)

1. 위 모든 `pytest` 명령이 PASS.
2. 동일 종목/동일 날짜 **재실행** 시 임팩트 판정·이슈 요약 LLM 호출이 캐시 히트로 0에 수렴(`result/runtime/llm_cache/` 하위에 항목 생성·재사용 확인).
3. 미매핑 시황 기사가 더 이상 전체 워치리스트로 팬아웃하지 않음(회사명 매칭 종목만 판정, 0매칭이면 기존 폴백 유지).
4. 표시/예측 분리 가드·점수 산식 불변(뉴스/공시 영향은 표시 전용 유지).
