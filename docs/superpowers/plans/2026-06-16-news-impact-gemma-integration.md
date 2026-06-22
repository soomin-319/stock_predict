# News-Impact gemma 통합 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 카카오 챗봇이 요청 종목을 예측할 때 뉴스/공시 임팩트 점수를 규칙 기반 대신 로컬 gemma(`gemma-4-26b-a4b`) LLM 판정으로 산출하되, 실패 시 규칙 기반으로 안전하게 폴백한다.

**Architecture:** `src/news_impact`의 기존 gemma 파이프라인(`run_daily_pipeline`)을 챗봇 예측 subprocess(`src/pipeline.py`)에서 인-프로세스로 호출한다. `context_raw_df`를 파이프라인이 요구하는 fixture(JSON)+watchlist/company-master(CSV)로 변환하고, 파이프라인이 쓰는 `report.json`을 기존 `append_news_impact_context()` 소비 경로에 그대로 먹인다. gemma 발화는 단일-종목 경로(`_start_prediction_job`)에만 연결하고 부트스트랩/실패 시 규칙 기반을 유지한다.

**Tech Stack:** Python 3.12+, pandas, pytest, stdlib `urllib`(LLM transport), 로컬 llama.cpp(OpenAI 호환 `/v1`).

---

## 선행 조건 (코드 작성 전 1회)

- 로컬 gemma 서버가 `http://localhost:8001/v1`에서 `gemma-4-26b-a4b`를 서빙 중이어야 함.
- 연결 검증:
  ```bash
  python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json
  ```
  기대 출력: `{"status": "ok", ...}` (JSON 한 줄). 실패하면 서버부터 띄운 뒤 진행.

## 파일 구조

- **신규** `src/reports/news_impact_fixture.py` — `context_raw_df` → fixture/watchlist/company-master 변환 (단일 책임).
- **수정** `src/reports/news_impact_context.py` — gemma 분기 함수 `append_llm_news_impact_context` 추가 (기존 규칙 기반/리포트 소비 함수와 한곳).
- **수정** `src/pipeline.py` — `--news-impact-llm-config` 인자 + 분기 배선.
- **수정** `src/chatbot/kakao_colab_bot.py` — `PipelineRuntimeConfig` 필드 + `build_command`/`_start_prediction_job` 배선.
- **신규 테스트** `tests/test_news_impact_fixture.py`, 기존 `tests/test_news_impact_context.py`/`tests/test_kakao_colab_bot.py`에 케이스 추가.

참고 — 변경하지 않음 (의도적):
- `_attach_news_impact_score`(`kakao_colab_bot.py:1576`)의 인-프로세스 라이브 부착은 1.5~2.5초 타임아웃이라 26B를 동기 호출할 수 없음 → 규칙 기반 폴백으로 유지.
- 부트스트랩 `_start_bootstrap_job`(`internal:prewarm_prediction_cache`)은 전 종목이라 규칙 기반 유지.

---

## Task 1: fixture 빌더

**Files:**
- Create: `src/reports/news_impact_fixture.py`
- Test: `tests/test_news_impact_fixture.py`

`context_raw_df` 컬럼: `Date, Symbol, source_type, title, published_at, provider, url, raw_id`.
`NewsItem`/`DisclosureItem` 스키마 검증(`src/news_impact/schema.py`)을 통과해야 하므로 tz-aware datetime, `signal_at == max(published_at, collected_at)`, ticker 6자리 숫자를 보장한다. 검증을 위해 **파이프라인 자체 리더로 되읽어** 통과 여부를 단언한다.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_news_impact_fixture.py
import json
from pathlib import Path

import pandas as pd

from src.news_impact.pipeline import (
    _news_item,
    _disclosure_item,
    _read_company_master,
    _read_json_object,
    _read_watchlist_tickers,
)
from src.reports.news_impact_fixture import build_news_impact_fixture


def _context_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "2026-06-16",
                "Symbol": "005930.KS",
                "source_type": "news",
                "title": "삼성전자 HBM 공급계약 체결",
                "published_at": "2026-06-16T09:30:00+09:00",
                "provider": "naver",
                "url": "https://news.example/1",
                "raw_id": "n1",
            },
            {
                "Date": "2026-06-16",
                "Symbol": "005930.KS",
                "source_type": "disclosure",
                "title": "[정정]단일판매ㆍ공급계약체결",
                "published_at": "",
                "provider": "dart",
                "url": "https://dart.example/2",
                "raw_id": "20260616000123",
            },
        ]
    )


def test_build_fixture_passes_pipeline_readers(tmp_path):
    bundle = build_news_impact_fixture(
        context_raw_df=_context_df(),
        symbols=["005930.KS"],
        symbol_name_map={"005930.KS": "삼성전자"},
        run_date="2026-06-16",
        output_dir=tmp_path,
    )

    fixture = _read_json_object(Path(bundle.fixture_path))
    news = [_news_item(row) for row in fixture["news"]]
    disclosures = [_disclosure_item(row) for row in fixture["disclosures"]]

    assert len(news) == 1
    assert len(disclosures) == 1
    # signal_at 불변식 + tz-aware 가 통과(_news_item/_disclosure_item __post_init__에서 검증)
    assert disclosures[0].ticker == "005930"
    assert disclosures[0].is_correction is True
    assert _read_watchlist_tickers(Path(bundle.watchlist_path)) == ["005930"]
    master = _read_company_master(Path(bundle.company_master_path))
    assert master["005930"]["company"] == "삼성전자"
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_news_impact_fixture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.reports.news_impact_fixture'`

- [ ] **Step 3: 최소 구현**

```python
# src/reports/news_impact_fixture.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.news_impact.market_clock import KST


@dataclass(frozen=True)
class NewsImpactFixtureBundle:
    fixture_path: Path
    watchlist_path: Path
    company_master_path: Path


def build_news_impact_fixture(
    *,
    context_raw_df: pd.DataFrame,
    symbols: Iterable[str],
    symbol_name_map: dict[str, str],
    run_date: str,
    output_dir: str | Path,
) -> NewsImpactFixtureBundle:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = {str(s) for s in symbols if str(s).strip()}

    df = context_raw_df.copy()
    df["Symbol"] = df["Symbol"].astype(str)
    if target:
        df = df[df["Symbol"].isin(target)]
    df["source_type"] = df["source_type"].astype(str)

    news_rows = []
    disclosure_rows = []
    for _, row in df.iterrows():
        published_at = _resolve_datetime(row.get("published_at"), run_date)
        signal_at = published_at  # collected_at == published_at → signal_at == max(...)
        source_type = str(row.get("source_type"))
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        if source_type == "news":
            news_rows.append(
                {
                    "source": str(row.get("provider") or "naver"),
                    "title": title,
                    "summary": "",
                    "url": str(row.get("url") or ""),
                    "original_url": str(row.get("url") or "") or None,
                    "publisher_domain": None,
                    "publisher_domain_source": None,
                    "publisher_confidence": 0.0,
                    "published_at": published_at.isoformat(),
                    "timestamp_source": "naver_pubDate" if _has_value(row.get("published_at")) else "manual",
                    "collected_at": published_at.isoformat(),
                    "signal_at": signal_at.isoformat(),
                    "market_session": "regular",
                    "raw_text": None,
                    "storage_policy": "metadata_only",
                    "quality_flags": ["title_only"],
                }
            )
        elif source_type == "disclosure":
            ticker = _ticker(str(row.get("Symbol")))
            if ticker is None:
                continue
            disclosure_rows.append(
                {
                    "source": str(row.get("provider") or "dart"),
                    "receipt_no": str(row.get("raw_id") or f"synthetic-{ticker}-{len(disclosure_rows)}"),
                    "corp_code": "",
                    "ticker": ticker,
                    "disclosure_title": title,
                    "disclosure_at": published_at.isoformat(),
                    "collected_at": published_at.isoformat(),
                    "signal_at": signal_at.isoformat(),
                    "is_correction": "정정" in title,
                    "original_receipt_no": None,
                    "url": str(row.get("url") or ""),
                    "quality_flags": [],
                }
            )

    fixture_path = out / "fixture.json"
    fixture_path.write_text(
        json.dumps({"news": news_rows, "disclosures": disclosure_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    watchlist_path = out / "watchlist.csv"
    company_master_path = out / "company_master.csv"
    tickers = [t for t in (_ticker(str(s)) for s in (target or df["Symbol"].unique())) if t]
    _write_csv(watchlist_path, ["ticker"], [{"ticker": t} for t in dict.fromkeys(tickers)])
    _write_csv(
        company_master_path,
        ["ticker", "company", "market", "sector"],
        [
            {
                "ticker": t,
                "company": symbol_name_map.get(_symbol_for_ticker(t, target), t),
                "market": _market(_symbol_for_ticker(t, target)),
                "sector": "",
            }
            for t in dict.fromkeys(tickers)
        ],
    )
    return NewsImpactFixtureBundle(fixture_path, watchlist_path, company_master_path)


def _has_value(value: object) -> bool:
    return bool(str(value or "").strip())


def _resolve_datetime(value: object, run_date: str) -> datetime:
    text = str(value or "").strip()
    if text:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=KST)
    base = datetime.fromisoformat(f"{run_date}T00:00:00")
    return datetime.combine(base.date(), time(9, 0), tzinfo=KST)


def _ticker(symbol: str) -> str | None:
    head = symbol.split(".", 1)[0]
    return head if len(head) == 6 and head.isdigit() else None


def _symbol_for_ticker(ticker: str, target: set[str]) -> str:
    for symbol in target:
        if symbol.split(".", 1)[0] == ticker:
            return symbol
    return ticker


def _market(symbol: str) -> str:
    return "KOSDAQ" if symbol.upper().endswith(".KQ") else "KOSPI"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_news_impact_fixture.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/reports/news_impact_fixture.py tests/test_news_impact_fixture.py
git commit -m "feat(news-impact): add context->gemma fixture builder"
```

---

## Task 2: gemma 분기 함수 (+폴백)

**Files:**
- Modify: `src/reports/news_impact_context.py` (함수 추가)
- Test: `tests/test_news_impact_context.py` (케이스 추가)

`run_daily_pipeline`을 인-프로세스로 호출하고, 그 산출물 `report.json`을 기존 `append_news_impact_context`로 소비한다. 테스트 격리를 위해 `_run_daily_pipeline`을 주입 가능한 인자로 둔다(기본은 실제 함수). 어떤 예외든 잡아 규칙 기반으로 폴백.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_news_impact_context.py 에 추가
import json
import types
from pathlib import Path

import pandas as pd

from src.reports.news_impact_context import append_llm_news_impact_context


def _pred_df() -> pd.DataFrame:
    return pd.DataFrame([{"Date": "2026-06-16", "Symbol": "005930.KS", "종목명": "삼성전자"}])


def _context_df() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "Date": "2026-06-16", "Symbol": "005930.KS", "source_type": "news",
            "title": "삼성전자 HBM 공급계약", "published_at": "2026-06-16T09:30:00+09:00",
            "provider": "naver", "url": "https://news.example/1", "raw_id": "n1",
        }]
    )


def test_append_llm_news_impact_uses_report_json(tmp_path):
    def fake_run(inputs):
        report = Path(inputs.output_dir) / "report.json"
        report.write_text(json.dumps({"rows": [{
            "date": "2026-06-16", "ticker": "005930",
            "news_disclosure_score": 42.0, "top_reason": "공급계약",
            "event_count": 1, "risk_flags": "llm_judged",
        }]}), encoding="utf-8")
        return types.SimpleNamespace(artifact_paths={"report.json": report})

    out = append_llm_news_impact_context(
        _pred_df(), _context_df(),
        llm_config_path="configs/news_impact.gemma.example.json",
        symbols=["005930.KS"], symbol_name_map={"005930.KS": "삼성전자"},
        run_date="2026-06-16", _run_daily_pipeline=fake_run,
    )
    assert "news_impact_final_score" in out.columns
    assert float(out.iloc[0]["news_impact_final_score"]) == 42.0


def test_append_llm_news_impact_falls_back_on_error():
    def boom(inputs):
        raise RuntimeError("gemma server down")

    out = append_llm_news_impact_context(
        _pred_df(), _context_df(),
        llm_config_path="configs/news_impact.gemma.example.json",
        symbols=["005930.KS"], symbol_name_map={"005930.KS": "삼성전자"},
        run_date="2026-06-16", _run_daily_pipeline=boom,
    )
    # 폴백: 규칙 기반 표시 컬럼이 생성됨
    assert "뉴스/공시 영향 점수" in out.columns
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_news_impact_context.py -k append_llm -v`
Expected: FAIL — `ImportError: cannot import name 'append_llm_news_impact_context'`

- [ ] **Step 3: 최소 구현**

```python
# src/reports/news_impact_context.py 상단 import 에 추가
import tempfile

from src.news_impact.pipeline import DailyPipelineInputs, run_daily_pipeline
from src.reports.news_impact_fixture import build_news_impact_fixture
```

```python
# src/reports/news_impact_context.py 에 함수 추가
def append_llm_news_impact_context(
    pred_df: pd.DataFrame,
    context_raw_df: pd.DataFrame | None,
    *,
    llm_config_path: str,
    symbols,
    symbol_name_map: dict[str, str],
    run_date: str,
    _run_daily_pipeline=run_daily_pipeline,
) -> pd.DataFrame:
    """gemma 파이프라인으로 임팩트를 판정해 붙인다. 실패 시 규칙 기반 폴백."""
    if pred_df.empty or context_raw_df is None or context_raw_df.empty:
        return append_generated_news_impact_context(pred_df, context_raw_df)
    try:
        with tempfile.TemporaryDirectory(prefix="news_impact_gemma_") as tmp:
            bundle = build_news_impact_fixture(
                context_raw_df=context_raw_df,
                symbols=symbols,
                symbol_name_map=symbol_name_map,
                run_date=run_date,
                output_dir=tmp,
            )
            result = _run_daily_pipeline(
                DailyPipelineInputs(
                    run_date=run_date,
                    watchlist_path=bundle.watchlist_path,
                    company_master_path=bundle.company_master_path,
                    input_fixture_path=bundle.fixture_path,
                    output_dir=tmp,
                    semantic_clustering=False,
                    llm_config_path=llm_config_path,
                )
            )
            report_path = result.artifact_paths["report.json"]
            return append_news_impact_context(pred_df, report_path)
    except Exception as exc:  # 서버 무응답/alias/타임아웃/스키마 검증 등 전부 폴백
        print(f"[NEWS IMPACT][gemma] 실패 → 규칙 기반 폴백: {type(exc).__name__}: {exc}")
        return append_generated_news_impact_context(pred_df, context_raw_df)
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_news_impact_context.py -k append_llm -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/reports/news_impact_context.py tests/test_news_impact_context.py
git commit -m "feat(news-impact): add gemma-backed context with rule-based fallback"
```

---

## Task 3: `src/pipeline.py` 인자/분기 배선

**Files:**
- Modify: `src/pipeline.py` (import, argparse, `run_pipeline`, 내부 예측 함수, 738-741 분기)
- Test: `tests/test_news_impact_context.py` (CLI 인자 케이스 추가)

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_news_impact_context.py 에 추가
from inspect import signature

from src.pipeline import build_cli_parser, run_pipeline


def test_pipeline_accepts_news_impact_llm_config_flag():
    parser = build_cli_parser()
    args = parser.parse_args(["--news-impact-llm-config", "configs/news_impact.gemma.example.json"])
    assert args.news_impact_llm_config == "configs/news_impact.gemma.example.json"
    assert "news_impact_llm_config" in signature(run_pipeline).parameters
```

> 파서 생성 함수는 `src/pipeline.py:1168` `build_cli_parser()`, `run_pipeline`은 1168행 이전 `src/pipeline.py:972`. 기존 `tests/test_news_impact_context.py:138`과 동일 패턴.

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_news_impact_context.py -k news_impact_llm_config -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'news_impact_llm_config'`

- [ ] **Step 3: 최소 구현**

`src/pipeline.py` 상단 import에 추가:
```python
from src.reports.news_impact_context import (
    append_generated_news_impact_context,
    append_news_impact_context,
    append_llm_news_impact_context,
)
```

argparse에 인자 추가 (기존 `--news-impact-report` 정의 바로 아래, 약 1178행):
```python
parser.add_argument(
    "--news-impact-llm-config",
    default=None,
    help="Optional llama.cpp/gemma LLM config for on-demand news-impact judging",
)
```

`run_pipeline(...)` 시그니처에 파라미터 추가 (기존 `news_impact_report: str | None = None,` 옆, 약 999행):
```python
news_impact_llm_config: str | None = None,
```

`run_pipeline` 내부에서 예측 함수로 전달 (기존 `news_impact_report=news_impact_report,` 호출부, 약 1137행):
```python
news_impact_report=news_impact_report,
news_impact_llm_config=news_impact_llm_config,
```

내부 예측 함수 시그니처에 파라미터 추가 (기존 `news_impact_report: str | None,` 옆, 약 699행):
```python
news_impact_llm_config: str | None,
```

분기 수정 (738-741행):
```python
if news_impact_report:
    pred_df = append_news_impact_context(pred_df, news_impact_report)
elif news_impact_llm_config:
    pred_df = append_llm_news_impact_context(
        pred_df,
        context_raw_df,
        llm_config_path=news_impact_llm_config,
        symbols=issue_summary_symbols,
        symbol_name_map=symbol_name_map,
        run_date=str(pred_df["Date"].astype(str).max()),
    )
else:
    pred_df = append_generated_news_impact_context(pred_df, context_raw_df)
```

`main()`에서 argparse 결과 전달 (기존 `news_impact_report=args.news_impact_report,` 옆, 약 1315행):
```python
news_impact_report=args.news_impact_report,
news_impact_llm_config=args.news_impact_llm_config,
```

> `symbol_name_map`은 해당 예측 함수 내 약 725행에서 이미 만들어져 있다(`symbol_name_map = get_symbol_name_map(...)`). 분기가 그 정의 이후(738행)이므로 그대로 참조 가능.

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_news_impact_context.py -v`
Expected: PASS (신규 + 기존 케이스 모두)

- [ ] **Step 5: 커밋**

```bash
git add src/pipeline.py tests/test_news_impact_context.py
git commit -m "feat(pipeline): wire --news-impact-llm-config into prediction"
```

---

## Task 4: 챗봇 `build_command`/`_start_prediction_job` 배선

**Files:**
- Modify: `src/chatbot/kakao_colab_bot.py` (`PipelineRuntimeConfig` 필드, `build_command`, `_start_prediction_job`)
- Test: `tests/test_kakao_colab_bot.py` (케이스 추가)

gemma 플래그는 **단일-종목(non-bootstrap)** 작업에만 추가한다.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_kakao_colab_bot.py 에 추가
from src.chatbot.kakao_colab_bot import PipelineRuntimeConfig


def test_build_command_includes_llm_config_when_enabled():
    cfg = PipelineRuntimeConfig(news_impact_llm_config="configs/news_impact.gemma.example.json")
    cmd = cfg.build_command("005930.KS", enable_news_impact_llm=True)
    assert "--news-impact-llm-config" in cmd
    assert "configs/news_impact.gemma.example.json" in cmd


def test_build_command_excludes_llm_config_when_disabled():
    cfg = PipelineRuntimeConfig(news_impact_llm_config="configs/news_impact.gemma.example.json")
    cmd = cfg.build_command("005930.KS", enable_news_impact_llm=False)
    assert "--news-impact-llm-config" not in cmd


def test_build_command_excludes_llm_config_when_unset():
    cfg = PipelineRuntimeConfig(news_impact_llm_config=None)
    cmd = cfg.build_command("005930.KS", enable_news_impact_llm=True)
    assert "--news-impact-llm-config" not in cmd
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k build_command_includes_llm_config -v`
Expected: FAIL — `TypeError: build_command() got an unexpected keyword argument 'enable_news_impact_llm'`

- [ ] **Step 3: 최소 구현**

`PipelineRuntimeConfig`에 필드 추가 (기존 필드 블록, 약 92행 `extra_args` 위):
```python
news_impact_llm_config: str | None = None
```

`build_command` 시그니처 + 본문 수정 (94-125행):
```python
def build_command(
    self,
    symbol: str,
    add_symbols: list[str] | None = None,
    issue_summary_symbols: list[str] | None = None,
    enable_news_impact_llm: bool = False,
) -> list[str]:
    normalized_add_symbols = [str(s) for s in (add_symbols or [symbol]) if str(s).strip()]
    normalized_issue_symbols = [str(s) for s in (issue_summary_symbols or [symbol]) if str(s).strip()]
    cmd = [
        self.python_executable,
        "src/pipeline.py",
        "--input",
        self.input_csv,
        "--add-symbols",
        *normalized_add_symbols,
        "--issue-summary-symbols",
        *normalized_issue_symbols,
    ]
    if self.fetch_investor_context:
        cmd.append("--fetch-investor-context")
        if not self.enable_investor_disclosure:
            cmd.append("--disable-disclosure-context")
        if self.dart_corp_map_csv:
            cmd.extend(["--dart-corp-map-csv", self.dart_corp_map_csv])
    if self.openai_model:
        cmd.extend(["--openai-model", self.openai_model])
    if enable_news_impact_llm and self.news_impact_llm_config:
        cmd.extend(["--news-impact-llm-config", self.news_impact_llm_config])
    if not self.use_external:
        cmd.append("--disable-external")
    if self.report_json:
        cmd.extend(["--report-json", self.report_json])
    cmd.extend(self.extra_args)
    return [str(part) for part in cmd]
```

`_start_prediction_job` 수정 (1109-1125행): 부트스트랩 여부를 한 번 계산해 사용하고 플래그로 전달.
```python
def _start_prediction_job(self, symbol: str) -> bool:
    issue_summary_symbols = [symbol]
    add_symbols = [symbol]
    is_bootstrap = self._is_bootstrap_required()
    if is_bootstrap:
        bootstrap_symbols = self._load_bootstrap_symbols_from_krx_map()
        if bootstrap_symbols:
            add_symbols = bootstrap_symbols
            self._console_log(
                f"{self._display_code(symbol)} 최초 예측 요청: {len(add_symbols)}개 심볼에 대해 예측을 실행합니다."
            )
        self._bootstrap_all_symbols_done = True

    command = self.runtime_config.build_command(
        symbol,
        add_symbols=add_symbols,
        issue_summary_symbols=issue_summary_symbols,
        enable_news_impact_llm=not is_bootstrap,
    )
```
(이하 기존 본문 유지)

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_kakao_colab_bot.py -k build_command -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/chatbot/kakao_colab_bot.py tests/test_kakao_colab_bot.py
git commit -m "feat(chatbot): pass gemma llm-config for on-demand single-symbol jobs"
```

---

## Task 5: 런타임 설정 + 문서 + e2e 확인

**Files:**
- Modify: `README.md` (gemma on-demand 사용법 한 단락)
- (런타임 구성처) `PipelineRuntimeConfig(news_impact_llm_config="configs/news_impact.gemma.example.json")` 설정

- [ ] **Step 1: README에 사용법 추가**

`configs/news_impact.gemma.example.json` 설명(약 127행) 인근에 추가:
```markdown
### On-demand gemma news-impact (챗봇)
1. 로컬 llama.cpp로 `gemma-4-26b-a4b`를 `http://localhost:8001/v1`에 서빙.
2. 연결 확인: `python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json`
3. 챗봇 구성에서 `PipelineRuntimeConfig(news_impact_llm_config="configs/news_impact.gemma.example.json")` 설정.
4. 이후 단일 종목 예측/“최신화” 시 gemma로 뉴스/공시 임팩트를 판정하며, 서버 무응답 시 규칙 기반으로 자동 폴백.
   부트스트랩(전 종목 prewarm)은 규칙 기반을 유지.
```

- [ ] **Step 2: 전체 테스트**

Run: `python -m pytest tests/test_news_impact_fixture.py tests/test_news_impact_context.py tests/test_kakao_colab_bot.py -v`
Expected: PASS (전부)

- [ ] **Step 3: 단일 종목 e2e (gemma 서버 필요)**

Run:
```bash
python src/pipeline.py --input data/real_ohlcv.csv \
  --add-symbols 005930.KS --issue-summary-symbols 005930.KS \
  --fetch-investor-context --disable-external \
  --news-impact-llm-config configs/news_impact.gemma.example.json \
  --report-json pipeline_report_with_context.json
```
Expected: 정상 종료. 콘솔/결과에 `005930.KS`의 뉴스/공시 영향 점수가 생성됨. gemma 서버를 끄고 재실행하면 `[NEWS IMPACT][gemma] 실패 → 규칙 기반 폴백` 로그와 함께 규칙 기반 점수가 채워짐(흐름 미중단).

- [ ] **Step 4: 커밋**

```bash
git add README.md
git commit -m "docs(news-impact): document on-demand gemma usage"
```

---

## Self-Review 메모 (작성자 점검 완료)

- **스펙 커버리지**: §5.1 fixture→Task1, §5.2 분기/폴백→Task2, §5.3 pipeline→Task3, §5.4 build_command→Task4, §6 스키마 매핑→Task1 코드, §7 매핑(watchlist 단일 티커)→Task1/Task3, §8 폴백→Task2, §10 테스트→각 Task, §11 선행조건→선행/Task5.
- **리스크**: `report.json` 키 정합성은 `report.py:REPORT_COLUMNS`로 확인 완료(`append_news_impact_rows`가 `news_disclosure_score→news_impact_final_score` 매핑). `_run_daily_pipeline`은 뉴스 클러스터를 LLM 판정하므로 단일 종목이면 호출 수가 적다(허용).
- **명칭 일관성**: `append_llm_news_impact_context`, `build_news_impact_fixture`, `NewsImpactFixtureBundle`, `enable_news_impact_llm`, `news_impact_llm_config`를 전 Task에서 동일하게 사용.
- **명칭 확정**: `src/pipeline.py` 파서는 `build_cli_parser()`(1168행), `run_pipeline`(972행) — Task3에 반영 완료.
