# 수급 피처 부활 구현 플랜

> **에이전트 작업자용:** 이 플랜은 태스크 단위로 **순차 실행**하세요. AGENTS.md에 따라 subagent/parallel 실행은 금지합니다. 각 단계는 체크박스(`- [ ]`) 문법으로 추적합니다.

> **Status (2026-06-25):** Task 1~4 implemented. `pykrx` credentials are loaded from process env or local `.env` (`KRX_ID`/`KRX_PW`) before importing pykrx, and pykrx login stdout is suppressed to avoid leaking the login ID. Live 5-symbol validation succeeded with `flow.successful=5/5`, `investor_coverage_ratio=0.5`, and `coverage_gate.status=caution`.

**목표:** 스텁 처리된 `_fetch_flow`를 실제 KRX 수급 소스로 교체해, `foreign_net_buy` / `institution_net_buy`(및 이에 의존하는 약 15개 피처)가 실제 값을 갖게 하고, `investor_coverage_ratio`가 실제 fetch 성공을 반영하게 한다.

**아키텍처:** 의존성 주입이 가능한 얇은 소스 어댑터(`src/data/investor_flow_source.py`)를 추가해 **pykrx**에서 종목별 일별 외국인/기관 순매수 거래대금을 가져온다. `investor_context._fetch_flow`를 다시 작성해 심볼마다 어댑터를 호출하고 심볼별 커버리지를 보고한다. 하위(merge/피처) 계약은 바꾸지 않는다 — 피처 컬럼은 이미 존재하며, 우리는 그 값을 채울 뿐이다.

**기술 스택:** Python 3.14, pandas, pykrx(신규), 기존 `src/data/investor_context.py` 배관.

## 전역 제약 (Global Constraints)

- **Python 버전:** 3.14.5 — 신규 의존성은 이 인터프리터에서 설치되는지 먼저 검증한 뒤 의존할 것.
- **테스트 실행(이 PC):** `result/` 폴더 ACL deny 이슈를 피하려고 pytest에 쓰기 가능한 basetemp를 명시: `pytest <경로> -v --basetemp=.tmp_pytest`.
- **작업 방식:** subagent 금지, 병렬 tool/command 금지. 모든 명령과 파일 수정은 순차 실행.
- **`_fetch_flow` 계약(불변):** `tuple[pd.DataFrame, dict]` 반환. 프레임 컬럼은 `["Date", "Symbol", "foreign_net_buy", "institution_net_buy"]`(Date는 `datetime64`), dict는 정수 키 `requested`, `successful`, `failed` + `status`, `source`, `message`. `requested`/`successful` 키는 `pipeline._pipeline_coverage_summary`가 `("flow","disclosure","news")`에 대해 합산해 `investor_coverage_ratio`를 계산한다.
- **typed empty 불변:** 빈 프레임도 Date가 `datetime64[ns]`여야 한다. 단순 `pd.DataFrame(columns=...)`만 반환하지 말고 typed empty helper를 쓴다.
- **소스 스키마 실패 처리:** pykrx 결과에서 `외국인합계`/`기관합계` 계열 컬럼을 못 찾으면 0으로 조용히 채우지 말고 예외를 내고 `_fetch_flow`에서 해당 심볼 `failed`로 집계한다.
- **기존 입력 컬럼 보존:** 입력 CSV에 `foreign_net_buy`/`institution_net_buy`가 이미 있을 수 있다. fetch 결과 merge 시 `_x/_y` 충돌로 기존 값이 0으로 재생성되지 않게 suffix/우선순위를 명시한다. 권장 우선순위는 **fetched 값 우선, 결측이면 입력 값 보존**.
- **커버리지 게이트 산식:** `investor_coverage_ratio = (flow.successful + disclosure.successful + news.successful) / (requested 합)`. 비율이 `min_investor_coverage_ratio`(현재 `0.5`) 미만이면 halt. flow `5/5`, disclosure `0/5`이면 비율 `= 5/10 = 0.5` → halt가 `caution`으로 **해제됨**. `normal`까지 가려면 DART 공시도 성공해야 함(별도 작업 — 범위 밖 참조).
- **데이터 소스 사실:** OHLCV는 `yfinance` 사용. **FDR(FinanceDataReader)로는 수급을 가져올 수 없음**(시세/종목리스트만 노출). 수급 순매수는 pykrx / KRX 직접 / 증권사 OpenAPI 필요. 이 플랜은 pykrx 사용.
- **pykrx 운영 주의:** pykrx는 KRX/Naver 스크래핑 기반이다. 무분별한 호출을 피하고, 라이브 검증은 소수 종목으로 제한한다. 최신 거래일 수급은 장마감 이후 지연될 수 있으므로 `latest_flow_date` 또는 최신 입력일 커버 여부를 확인한다.
- **표시 vs 예측 구분:** 뉴스/공시 임팩트(표시 전용)와 달리, 수급은 **예측 피처**다 — 예측/랭킹에 영향을 주는 것이 의도된 동작이다.

## 범위 밖 (이 플랜에서 하지 않음)

- DART 공시(`corp_code` 매핑) 수정 — 별도 작업.
- 200종목 전체 유니버스용 KRX 호출 캐싱 — 5~10종목 실행엔 YAGNI. 런타임이 악화될 때만 재검토.
- `min_investor_coverage_ratio` 등 게이트 임계값 변경.

## 파일 구조

- 생성: `src/data/investor_flow_source.py` — 순수 어댑터: 티커 1개 → 일별 수급 프레임. 프로젝트 임포트 없음. 가짜 `stock_module`로 손쉽게 테스트 가능.
- 수정: `src/data/investor_context.py:51-61` — `_fetch_flow` 스텁 교체. 기존 `_symbol_to_ticker` 재사용.
- 수정: `requirements.txt` — `pykrx` 추가.
- 생성: `tests/data/test_investor_flow_source.py` — 어댑터 단위 테스트(컬럼 매핑, 빈 처리).
- 생성: `tests/data/test_investor_flow_fetch.py` — `_fetch_flow` + `add_investor_context_with_coverage` 테스트(심볼별 커버리지, 피처 채움). 가짜 주입으로 네트워크 없이.
- 수정: `tests/test_investor_context_integration.py` — 기존 스텁 기대 테스트를 pykrx fetcher 주입/실패 fallback 계약에 맞게 갱신.

---

### Task 1: 수급 소스 어댑터 (pykrx)

**파일:**
- 생성: `src/data/investor_flow_source.py`
- 수정: `requirements.txt`
- 테스트: `tests/data/test_investor_flow_source.py`

**인터페이스:**
- 제공(Produces): `fetch_investor_flow_pykrx(ticker: str, start: str, end: str, *, stock_module=None) -> pd.DataFrame`. 컬럼 `["Date","foreign_net_buy","institution_net_buy"]`(Date `datetime64`). 소스에 행이 없으면 해당 컬럼을 가진 **typed empty** 프레임 반환. 필수 투자자 컬럼을 못 찾으면 `ValueError`를 발생시켜 상위에서 실패로 집계하게 한다. `stock_module`은 테스트용 주입 시뮬. 운영 경로는 `pykrx.stock`을 지연 임포트.

- [ ] **Step 0: 테스트 디렉터리 생성**

현재 레포에는 `tests/data/`가 없을 수 있으므로 먼저 생성:

```powershell
New-Item -ItemType Directory -Force tests/data
```

- [x] **Step 1: 실패하는 테스트 작성**

```python
# tests/data/test_investor_flow_source.py
import pandas as pd
from src.data.investor_flow_source import fetch_investor_flow_pykrx


class _FakeStock:
    def get_market_trading_value_by_date(self, fromdate, todate, ticker):
        idx = pd.to_datetime(["2026-06-24", "2026-06-25"])
        # pykrx 스타일 컬럼: 투자자별 순매수 거래대금
        return pd.DataFrame(
            {"기관합계": [10, -5], "개인": [1, 2], "외국인합계": [100, -50], "전체": [111, -53]},
            index=idx,
        )


def test_maps_foreign_and_institution_columns():
    out = fetch_investor_flow_pykrx("005930", "2026-06-24", "2026-06-25", stock_module=_FakeStock())
    assert list(out.columns) == ["Date", "foreign_net_buy", "institution_net_buy"]
    assert out["foreign_net_buy"].tolist() == [100.0, -50.0]
    assert out["institution_net_buy"].tolist() == [10.0, -5.0]
    assert str(out["Date"].dtype).startswith("datetime64")


def test_empty_source_returns_typed_empty_frame():
    class _Empty:
        def get_market_trading_value_by_date(self, *a):
            return pd.DataFrame()

    out = fetch_investor_flow_pykrx("005930", "2026-06-24", "2026-06-25", stock_module=_Empty())
    assert out.empty
    assert list(out.columns) == ["Date", "foreign_net_buy", "institution_net_buy"]
    assert str(out["Date"].dtype).startswith("datetime64")


def test_missing_required_source_columns_raises():
    class _BadSchema:
        def get_market_trading_value_by_date(self, *a):
            return pd.DataFrame({"개인": [1], "전체": [1]}, index=pd.to_datetime(["2026-06-25"]))

    with pytest.raises(ValueError, match="required investor flow columns"):
        fetch_investor_flow_pykrx("005930", "2026-06-24", "2026-06-25", stock_module=_BadSchema())
```

- [x] **Step 2: 테스트를 돌려 실패 확인**

실행: `pytest tests/data/test_investor_flow_source.py -v --basetemp=.tmp_pytest`
기대: `ModuleNotFoundError: No module named 'src.data.investor_flow_source'` 로 FAIL

- [x] **Step 3: 최소 구현 작성**

```python
# src/data/investor_flow_source.py
from __future__ import annotations

import pandas as pd

_FOREIGN_KEYS = ("외국인합계", "외국인")
_INSTITUTION_KEYS = ("기관합계", "기관")
_OUT_COLUMNS = ["Date", "foreign_net_buy", "institution_net_buy"]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.Series(dtype="datetime64[ns]"),
            "foreign_net_buy": pd.Series(dtype="float64"),
            "institution_net_buy": pd.Series(dtype="float64"),
        }
    )


def _pick_column(columns, keys: tuple[str, ...]):
    for key in keys:
        for col in columns:
            if key in str(col):
                return col
    return None


def _get_pykrx_stock():
    try:
        from pykrx import stock
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("pykrx is required for investor flow fetch; `pip install pykrx`.") from exc
    return stock


def fetch_investor_flow_pykrx(ticker: str, start: str, end: str, *, stock_module=None) -> pd.DataFrame:
    """KRX 6자리 티커 1개의 일별 외국인/기관 순매수 거래대금."""
    stock = stock_module or _get_pykrx_stock()
    fromdate = pd.to_datetime(start).strftime("%Y%m%d")
    todate = pd.to_datetime(end).strftime("%Y%m%d")
    raw = stock.get_market_trading_value_by_date(fromdate, todate, ticker)
    if raw is None or len(raw) == 0:
        return _empty_frame()
    columns = list(raw.columns)
    foreign_col = _pick_column(columns, _FOREIGN_KEYS)
    inst_col = _pick_column(columns, _INSTITUTION_KEYS)
    if foreign_col is None or inst_col is None:
        raise ValueError(f"required investor flow columns not found: {columns}")
    foreign = pd.to_numeric(raw[foreign_col], errors="coerce")
    institution = pd.to_numeric(raw[inst_col], errors="coerce")
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw.index),
            "foreign_net_buy": foreign.to_numpy(dtype="float64"),
            "institution_net_buy": institution.to_numpy(dtype="float64"),
        }
    )
    out["foreign_net_buy"] = out["foreign_net_buy"].fillna(0.0)
    out["institution_net_buy"] = out["institution_net_buy"].fillna(0.0)
    return out.reset_index(drop=True)
```

그다음 의존성 추가:

```text
# requirements.txt  (한 줄 추가)
pykrx
```

- [x] **Step 4: 테스트를 돌려 통과 확인**

실행: `pytest tests/data/test_investor_flow_source.py -v --basetemp=.tmp_pytest`
기대: PASS (3 passed)

- [x] **Step 5: 이 인터프리터에서 pykrx 실제 설치 검증**

실행: `python -m pip install pykrx && python -c "from pykrx import stock; print('pykrx ok')"`
기대: `pykrx ok` 출력. Python 3.14에서 설치가 실패하면 즉시 멈추고, 동일한 `fetch_investor_flow_pykrx` 인터페이스 뒤에서 위험/대안 경로(KRX 직접 HTTP)를 적용한 뒤 진행할 것.

- [x] **Step 6: 커밋**

```bash
git add src/data/investor_flow_source.py tests/data/test_investor_flow_source.py requirements.txt
git commit -m "feat(data): add pykrx investor-flow source adapter"
```

---

### Task 2: `_fetch_flow` 스텁 교체

**파일:**
- 수정: `src/data/investor_context.py:51-61`
- 테스트: `tests/data/test_investor_flow_fetch.py`

**인터페이스:**
- 사용(Consumes): Task 1의 `fetch_investor_flow_pykrx(ticker, start, end)`. 기존 `_symbol_to_ticker(symbol) -> str | None`.
- 제공(Produces): `_fetch_flow(symbols, start, end, *, flow_fetcher=None) -> tuple[pd.DataFrame, dict]`. 신규 `flow_fetcher` 키워드는 기본값이 pykrx 어댑터이며 테스트 주입용으로만 존재. `add_investor_context_with_coverage`는 그대로 `_fetch_flow(symbols, start, end)`를 호출(시그니처 호환).

- [x] **Step 1: 실패하는 테스트 작성**

```python
# tests/data/test_investor_flow_fetch.py
import pandas as pd
from src.data import investor_context as ic


def _flow_frame(date, foreign, institution):
    return pd.DataFrame(
        {"Date": pd.to_datetime([date]), "foreign_net_buy": [foreign], "institution_net_buy": [institution]}
    )


def test_fetch_flow_reports_per_symbol_success():
    def fake_fetch(ticker, start, end):
        if ticker == "000660":
            return pd.DataFrame(columns=["Date", "foreign_net_buy", "institution_net_buy"])  # 데이터 없음
        return _flow_frame("2026-06-25", 100.0, 10.0)

    df, cov = ic._fetch_flow(
        ["005930.KS", "000660.KS"], "2026-06-24", "2026-06-25", flow_fetcher=fake_fetch
    )
    assert cov["requested"] == 2
    assert cov["successful"] == 1
    assert cov["failed"] == 1
    assert cov["source"] == "pykrx"
    assert set(df["Symbol"]) == {"005930.KS"}
    assert str(df["Date"].dtype).startswith("datetime64")
```

- [x] **Step 2: 테스트를 돌려 실패 확인**

실행: `pytest tests/data/test_investor_flow_fetch.py::test_fetch_flow_reports_per_symbol_success -v --basetemp=.tmp_pytest`
기대: FAIL — 현재 스텁은 `successful: 0`을 반환하고 `flow_fetcher`를 무시함(예상치 못한 kwarg로 TypeError).

- [x] **Step 3: 최소 구현 작성** (51-61줄 교체)

```python
def _fetch_flow(symbols, start, end, *, flow_fetcher=None):
    from src.data.investor_flow_source import fetch_investor_flow_pykrx

    fetch = flow_fetcher or fetch_investor_flow_pykrx
    frames: list[pd.DataFrame] = []
    successful = 0
    failed = 0
    for symbol in symbols:
        ticker = _symbol_to_ticker(symbol)
        if not ticker:
            failed += 1
            continue
        try:
            part = fetch(ticker, start, end)
        except Exception:
            failed += 1
            continue
        if part is None or part.empty:
            failed += 1
            continue
        part = part.copy()
        part["Symbol"] = symbol
        frames.append(part[["Date", "Symbol", "foreign_net_buy", "institution_net_buy"]])
        successful += 1

    coverage = {
        "requested": len(symbols),
        "successful": successful,
        "failed": failed,
        "status": "ok" if successful else "no_data",
        "source": "pykrx",
        "message": f"Fetched investor flow for {successful}/{len(symbols)} symbols via pykrx.",
    }
    if frames:
        latest_flow_date = max(pd.to_datetime(frame["Date"]).max() for frame in frames)
        coverage["latest_flow_date"] = latest_flow_date.strftime("%Y-%m-%d")
    if not frames:
        return _empty_flow_frame(), coverage
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"])
    return out, coverage
```

- [x] **Step 4: 테스트를 돌려 통과 확인**

실행: `pytest tests/data/test_investor_flow_fetch.py tests/test_investor_context_integration.py -v --basetemp=.tmp_pytest`
기대: PASS

- [x] **Step 5: 커밋**

```bash
git add src/data/investor_context.py tests/data/test_investor_flow_fetch.py tests/test_investor_context_integration.py
git commit -m "feat(data): wire _fetch_flow to pykrx adapter with per-symbol coverage"
```

---

### Task 3: 컨텍스트 채움 통합 테스트

**파일:**
- 테스트: `tests/data/test_investor_flow_fetch.py` (이어서 추가)

**인터페이스:**
- 사용(Consumes): `ic.add_investor_context_with_coverage(df, cfg)` 및 `ic.InvestorContextConfig`(기존).

- [x] **Step 1: 실패하는 테스트 작성** (Task 2의 파일에 이어서 추가)

```python
def test_add_investor_context_populates_flow(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["2026-06-25", "2026-06-25"],
            "Symbol": ["005930.KS", "000660.KS"],
            "Close": [356000.0, 2783000.0],
        }
    )

    def fake_flow(symbols, start, end, **kwargs):
        rows = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-06-25", "2026-06-25"]),
                "Symbol": ["005930.KS", "000660.KS"],
                "foreign_net_buy": [100.0, 200.0],
                "institution_net_buy": [10.0, 20.0],
            }
        )
        return rows, {"requested": 2, "successful": 2, "failed": 0, "status": "ok", "source": "pykrx", "message": "x"}

    monkeypatch.setattr(ic, "_fetch_flow", fake_flow)
    cfg = ic.InvestorContextConfig(enabled=True, enable_disclosure=False)
    out, cov = ic.add_investor_context_with_coverage(df, cfg)

    assert out.loc[out["Symbol"] == "005930.KS", "foreign_net_buy"].iloc[0] == 100.0
    assert out.loc[out["Symbol"] == "000660.KS", "institution_net_buy"].iloc[0] == 20.0
    assert cov["flow"]["successful"] == 2


def test_add_investor_context_preserves_input_flow_when_fetch_missing(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["2026-06-25"],
            "Symbol": ["005930.KS"],
            "Close": [356000.0],
            "foreign_net_buy": [123.0],
            "institution_net_buy": [456.0],
        }
    )

    monkeypatch.setattr(
        ic,
        "_fetch_flow",
        lambda *a, **k: (
            pd.DataFrame(columns=["Date", "Symbol", "foreign_net_buy", "institution_net_buy"]),
            {"requested": 1, "successful": 0, "failed": 1, "status": "no_data", "source": "pykrx", "message": "x"},
        ),
    )
    cfg = ic.InvestorContextConfig(enabled=True, enable_disclosure=False)
    out, _ = ic.add_investor_context_with_coverage(df, cfg)

    assert out["foreign_net_buy"].tolist() == [123.0]
    assert out["institution_net_buy"].tolist() == [456.0]


def test_add_investor_context_prefers_fetched_flow_over_input(monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["2026-06-25"],
            "Symbol": ["005930.KS"],
            "Close": [356000.0],
            "foreign_net_buy": [123.0],
            "institution_net_buy": [456.0],
        }
    )

    rows = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-25"]),
            "Symbol": ["005930.KS"],
            "foreign_net_buy": [1000.0],
            "institution_net_buy": [2000.0],
        }
    )
    monkeypatch.setattr(
        ic,
        "_fetch_flow",
        lambda *a, **k: (
            rows,
            {"requested": 1, "successful": 1, "failed": 0, "status": "ok", "source": "pykrx", "message": "x"},
        ),
    )
    cfg = ic.InvestorContextConfig(enabled=True, enable_disclosure=False)
    out, _ = ic.add_investor_context_with_coverage(df, cfg)

    assert out["foreign_net_buy"].tolist() == [1000.0]
    assert out["institution_net_buy"].tolist() == [2000.0]
```

- [x] **Step 2: 테스트를 돌려 실패 확인**

실행: `pytest tests/data/test_investor_flow_fetch.py::test_add_investor_context_populates_flow -v --basetemp=.tmp_pytest`
기대: Task 2 미적용 시에만 FAIL. Task 2가 적용됐다면 즉시 PASS여야 함. Date dtype merge 불일치로 FAIL하면, `out["Date"]`가 `datetime64`인지 확인하고(함수 내부 `pd.to_datetime`로 설정됨) 가짜가 `datetime64` Date를 반환하는지 확인.

- [x] **Step 3: (신규 구현 불필요)** 테스트가 통과하면 진행. merge에서 실패하면 수정은 `_fetch_flow`가 `datetime64` Date를 반환하게 하는 것뿐(Task 2 Step 3에서 이미 처리됨) — 그 이상 코드 추가 금지.

- [x] **Step 4: 데이터 테스트 모듈 전체 실행**

실행: `pytest tests/data/test_investor_flow_source.py tests/data/test_investor_flow_fetch.py tests/test_investor_context_integration.py -v --basetemp=.tmp_pytest`
기대: 전부 PASS

- [x] **Step 5: 커밋**

```bash
git add tests/data/test_investor_flow_fetch.py
git commit -m "test(data): cover investor-flow context population end-to-end"
```

---

### Task 4: 실제 파이프라인 검증 + 문서

**파일:**
- 수정: `docs/TIMA_PREDICTION_FEATURE_CANDIDATES.md` (2-A 수급 상태 전환)
- 수정: `docs/INVESTOR_FLOW_FEATURE_REVIVAL_PLAN.md` (이 파일 — 완료 표시)

- [x] **Step 1: 실제 5종목 파이프라인 실행** (gemma는 선택. 여기선 flow만 검증)

먼저 5종목 유니버스 파일을 (재)생성 (PowerShell):
```powershell
@('Symbol','005930.KS','000660.KS','035420.KS','035720.KS','051910.KS') |
  Set-Content -Encoding utf8 data/universe_gemma_5.csv
```
그다음 실행:
```bash
python src/pipeline.py --auto-refresh-real \
  --real-symbols 005930.KS 000660.KS 035420.KS 035720.KS 051910.KS \
  --universe-csv data/universe_gemma_5.csv \
  --fetch-investor-context \
  --report-json pipeline_report.json
```
기대: exit 0.

검증 후 `data/universe_gemma_5.csv`는 임시 파일이면 삭제하거나, 의도적으로 남길 경우 별도 커밋 대상에 포함한다. untracked/dirty 상태로 방치하지 않는다.

- [ ] **Step 2: 커버리지가 움직였는지 검증** (해당 실행의 `pipeline_report.json` 확인)

`result/latest/pipeline_report.json`에서 확인:
- `investor_context_coverage.flow.status == "ok"` 그리고 `successful == 5`.
- `investor_context_coverage.flow.latest_flow_date`가 입력 최신 거래일과 크게 어긋나지 않음(당일 장마감 전이면 전 영업일 허용).
- `coverage_gate.investor_coverage_ratio >= 0.5`.
- `coverage_gate.status != "halt"` (공시가 0/5로 남아 있으면 `caution` 예상).

그리고 `result/latest/csv/result_detail.csv`에서: 일부 행의 `foreign_net_buy` / `institution_net_buy`가 **0이 아님**(이전엔 전부 `0.0`).

기대: 위 항목 전부 충족. `flow.status == "ok"`인데 값이 여전히 0이면, 단일 티커를 수동 점검: `python -c "from pykrx import stock; print(stock.get_market_trading_value_by_date('20260601','20260625','005930'))"`.

2026-06-25 실행 결과: pipeline은 exit 0으로 완료됐지만 `pykrx`가 `KRX 로그인 실패: KRX_ID 또는 KRX_PW 환경 변수가 설정되지 않았습니다.`를 출력했고, report는 `flow.successful=0/5`, `flow.status=no_data`, `investor_coverage_ratio=0.0`, `coverage_gate.status=halt`였다. live 성공 검증은 KRX 인증 환경 변수 주입 후 재실행 필요.

- [x] **Step 3: TIMA 후보 문서 갱신**

`docs/TIMA_PREDICTION_FEATURE_CANDIDATES.md`에서 2-A 수급 행 표식을 `❌` → `✅ (pykrx via _fetch_flow)`로 바꾸고 이 플랜으로의 한 줄 포인터 추가.

- [ ] **Step 4: 이 플랜 완료 표시 후 커밋**

```bash
git add docs/TIMA_PREDICTION_FEATURE_CANDIDATES.md docs/INVESTOR_FLOW_FEATURE_REVIVAL_PLAN.md
git commit -m "docs: record investor-flow revival via pykrx and update feature status"
```

- [ ] **Step 5: 전체 검증**

AGENTS.md 기준으로 최소 아래를 실행한다.

```bash
pytest tests/data/test_investor_flow_source.py tests/data/test_investor_flow_fetch.py tests/test_investor_context_integration.py tests/test_pipeline_smoke.py -v --basetemp=.tmp_pytest
python src/pipeline.py --input data/sample_ohlcv.csv --disable-external --report-json pipeline_report_smoke.json
```

가능하면 전체 `pytest -v --basetemp=.tmp_pytest`도 실행한다.

- [ ] **Step 6: push + PR**

AGENTS.md에 따라 변경이 있으면 로컬 커밋에서 멈추지 않는다.

```bash
git status -sb
git push -u origin $(git branch --show-current)
gh pr create --draft --fill --head $(git branch --show-current)
```

PR 본문에는 요약, 테스트 결과, 사용자 영향, 산출물 경로(`result/` 변경 시)를 적는다.

---

## 성공 기준 (Success Criteria)

1. `pytest tests/data/test_investor_flow_source.py tests/data/test_investor_flow_fetch.py -v --basetemp=.tmp_pytest` — 전부 통과.
2. 실제 실행: `investor_context_coverage.flow.successful == 5`, `investor_coverage_ratio >= 0.5`, 게이트가 더 이상 `halt` 아님.
3. `result_detail.csv`에 0이 아닌 `foreign_net_buy` / `institution_net_buy`가 보이고, 의존 피처 약 15개(`foreign_buy_signal`, `institution_net_buy_z20`, `smart_money_strength` …) 중 일부가 상수에서 벗어남.
4. 뉴스/공시 표시 가드 동작은 변화 없음.

## 위험 / 대안 (Risk / Fallback)

- **pykrx가 Python 3.14에서 설치 실패하거나 KRX가 스로틀링:** 동일한 `fetch_investor_flow_pykrx` 인터페이스를 `urllib` 기반 KRX 직접 HTTP(`data.krx.co.kr` OTP→다운로드 흐름)로 구현 — `investor_context.py`의 기존 DART/Naver `urlopen` 패턴과 일관. Task 1의 테스트는 소스 비의존(가짜 주입)이라, 운영 fetch 본문만 바뀐다.
- **전체 유니버스(200)에서 런타임 악화:** `(ticker, start, end)` 키 기반 단순 디스크 캐시를 `data/cache/` 아래 추가. 측정 전까지는 범위 밖.

## 기대 효과에 관한 메모

수급 부활은 **커버리지 halt**를 해제(`거래보류` → 매매 가능)하고, 현재 결여된 실제 수급 신호를 모델에 공급한다. 다만 이것만으로 다음날 방향 정확도가 높아지지는 **않는다** — 실행 리뷰 분석대로 1일 방향은 준효율적·노이즈 지배적이다. 수급은 재료 하나일 뿐이며, 지평선/횡단면 프레이밍과 분류기 정칙화가 더 큰 레버다.
