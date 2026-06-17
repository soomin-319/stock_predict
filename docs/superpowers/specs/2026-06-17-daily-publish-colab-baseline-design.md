# 일일 Publish → GitHub 기준데이터 → Colab 서빙 설계

- 날짜: 2026-06-17
- 상태: 승인됨 (설계) — 스펙 리뷰 대기
- 작성 맥락: 현재 Colab 봇은 실행할 때마다 기본 200종목(KOSPI200)을 로컬에서 직접 예측(부트스트랩)하고, 산출물은 `result/`(gitignore)에만 남아 GitHub로 공유되지 않는다. 운영 흐름을 "로컬에서 하루 1회 수동 publish → GitHub `published/`에 커밋 → Colab은 GitHub 기준데이터를 기본 서빙, 사용자가 요청한 종목만 세션에서 예측" 으로 전환한다.

## 1. 목표 / 요구사항

1. 기본종목 200개(KOSPI200) 예측을 **로컬에서 수동 1회 실행**해 GitHub에 저장한다.
2. Colab 서버는 GitHub에 저장된 파일을 바탕으로 예측 결과를 보여준다.
3. Colab 기본 동작은 **GitHub에서 불러온 기준데이터 표시**이며, 기존 기본종목 200개 예측은 사용자가 명시적으로 요청하지 않는 한 실행하지 않는다.
4. 사용자 요청 시에만 특정 종목 예측·최신화를 진행한다(세션 한정, GitHub push 없음).

## 2. 확정된 결정

| 항목 | 결정 |
| --- | --- |
| 배치 실행 위치 | 로컬 PC, **수동 단일 명령**(자동 cron 아님) → git push |
| GitHub 저장 형태 | `main` 브랜치의 추적 폴더 `published/` |
| 히스토리 | `published/latest/` + `published/history/<trading_date>/`(거래일당 1스냅샷) + `published/index.json` |
| 배치 뉴스 모드 | gemma(`configs/news_impact.gemma.example.json`), 서버 다운 시 규칙기반 자동 폴백 |
| Colab 기본 | `published/latest/` 베이스라인 서빙, 자동 부트스트랩 OFF |
| 온디맨드 결과 | Colab 세션 임시(`result/`)에만 두고 베이스라인 위 오버레이, push 없음 |
| 서빙 구조 | published 기준 + 세션 오버레이(종목코드 기준, 세션 우선) |
| 뉴스 점수 성격 | 표시용 유지, `predicted_return`·추천·신호 정책 미반영 |

## 3. 배경 — 현재 무엇을 바꾸나 (근거)

- 봇 자동 부트스트랩이 2곳에 있다.
  - 런치 시: `launch_colab_kakao_bot` → `should_prewarm and cfg.bootstrap_on_launch` → `_start_bootstrap_job` → `prewarm_prediction_cache` → 전 종목 `run_colab_pipeline` (`src/chatbot/kakao_colab_bot.py:2111`, `:1271`, `:2047`).
  - 첫 종목 요청 시: `_is_bootstrap_required()` → `_start_prediction_job`에서 `--add-symbols`에 전 종목 주입 (`kakao_colab_bot.py:1100`, `:1129`).
- 봇이 결과를 읽는 소스는 `result/`(`_resolve_result_path`/`_load_cached_result_simple`, `kakao_colab_bot.py:278`, `:747`)이며 `result/`는 `.gitignore` 대상이라 GitHub로 공유되지 않는다.
- 온디맨드 단일 종목 예측은 이미 `--add-symbols <symbol>` + gemma(`--news-impact-llm-config`)로 동작한다(`build_command`, `kakao_colab_bot.py:95`, `:1139`). 요구사항 #4는 거의 충족 상태.
- 따라서 변경의 핵심은 (a) 로컬 publish 산출 + `published/` 커밋, (b) 봇/Colab의 기본 소스를 `published/latest/`로 바꾸고 세션 결과를 오버레이, (c) 자동 부트스트랩 기본 OFF 이다.

## 4. 아키텍처 & 데이터 흐름

```
[로컬 PC]
  stock-predict-publish
    ├ 1. 200 KOSPI200 OHLCV 갱신 (--auto-refresh-real 기본 / --full-refresh)
    ├ 2. run_pipeline(투자자컨텍스트 ON, news_impact_llm_config=gemma)
    │      → result/latest/ (promoted=true 운영 산출물)
    ├ 3. 게시 세트 복사: result/latest/ → published/latest/ AND published/history/<trading_date>/
    │      + publish_meta.json 작성, published/index.json 갱신
    └ 4. git add published/ && commit && push   (--no-push/--dry-run 차단)

[Colab]
  git pull
    → load_published_predictions(date=None)   # published/latest/ 베이스라인 표시 (파이프라인 미실행)
    → launch_colab_kakao_bot(prewarm_cache=False)   # 자동 부트스트랩 OFF
        └ 사용자 종목 요청 → _start_prediction_job(--add-symbols 1종목, gemma)
             → result/ (세션) → 베이스라인 위 오버레이 → 서빙
```

- Colab은 원격 런타임이라 로컬 gemma(`localhost:8001`)에 접근 불가. Colab 세션 온디맨드는 gemma 설정이 닿지 않으면 규칙기반으로 자동 폴백한다(기존 폴백 경로). 베이스라인의 gemma 점수는 로컬 publish 시 산출된 값이 그대로 표시된다.

## 5. 컴포넌트

### 5.1 Publish 명령 (신규) — `src/ops/publish_predictions.py`
- 콘솔 스크립트: `pyproject.toml [project.scripts]`에 `stock-predict-publish = "src.ops.publish_predictions:main"` 추가.
- 책임:
  1. 200개 KOSPI200 심볼 OHLCV 갱신(publish 입력 = `data/real_ohlcv.csv`, 봇 세션 입력과 분리됨). 기본 `--auto-refresh-real`(증분 append), `--full-refresh` 시 전체 재수집. 심볼 소스는 기존 기본 유니버스 로직 재사용(`_fallback_symbols_from_input_or_default` / `data/kospi200_symbol_name_map.csv`).
  2. `run_pipeline(...)` 호출: `use_investor_context=True`, `news_impact_llm_config` = `--news-mode gemma`(기본)이면 `configs/news_impact.gemma.example.json`, `--news-mode rule`이면 None. 산출물은 `result/latest/`(promoted 운영).
  3. 게시 세트를 `published/latest/`와 `published/history/<trading_date>/` 양쪽에 복사하고 메타/인덱스 작성.
  4. `git add published/ && git commit -m "..." && git push`. `--no-push`/`--dry-run` 시 push/commit 생략.
- 게시 세트(각 폴더 동일):
  - `csv/result_simple.csv`, `csv/result_detail.csv`, `csv/result_news.csv`, `csv/result_disclosure.csv`
  - `manifest.json`, `pipeline_report.json`
  - `publish_meta.json`
- `publish_meta.json` 스키마:
  ```json
  {
    "generated_at_kst": "2026-06-17T18:05:00+09:00",
    "trading_date": "2026-06-17",
    "news_mode": "gemma",            // 실제 사용된 모드(폴백 시 "rule_based")
    "source_run_id": "<manifest.run_id>",
    "symbol_count": 200,
    "git": {"commit": "<sha or null until committed>", "branch": "main"}
  }
  ```
- `trading_date` 결정: `manifest`/`result_detail.csv`의 최신 예측 기준일(KST)을 사용. 같은 거래일 재publish 시 그 날짜 history 폴더를 덮어쓴다(거래일당 1스냅샷).
- 플래그: `--news-mode {gemma,rule}`(기본 gemma), `--full-refresh`, `--no-push`, `--dry-run`, `--config-json`(파이프라인 override 전달).
- 중단 조건: 파이프라인 manifest 상태가 `pass`/`warning`이 아니거나 `promoted!=true`이면 publish를 **중단**하고 `published/`를 변경하지 않는다(비운영 데이터 커밋 금지).

### 5.2 `published/` 폴더 규약 (신규 추적 디렉터리)
- `published/latest/` — 최신 게시본. Colab 기본 읽기 대상.
- `published/history/<trading_date>/` — 거래일별 스냅샷(예: `published/history/2026-06-17/`).
- `published/index.json` — 가용 날짜 목록 + 날짜별 메타.
  ```json
  {
    "latest": "2026-06-17",
    "entries": [
      {"trading_date": "2026-06-17", "generated_at_kst": "...", "news_mode": "gemma", "symbol_count": 200, "source_run_id": "..."}
    ]
  }
  ```
- `.gitignore`는 `result/`만 무시하므로 `published/`는 그대로 추적된다(확인 필요, 추가 규칙 불필요).

### 5.3 봇 — published 베이스라인 + 세션 오버레이 (`src/chatbot/kakao_colab_bot.py`)
- `PipelineRuntimeConfig` 변경:
  - 추가: `published_dir: str = "published/latest"`.
  - 기본값 전환(부트스트랩 OFF): `bootstrap_on_launch=False`, `prewarm_default_predictions=False`, `bootstrap_default_symbols=False`.
  - **세션 입력 분리**: `input_csv` 기본값을 publish용 200종목 파일(`data/real_ohlcv.csv`)이 아닌 **세션 전용 경로**(예: `result/session/session_ohlcv.csv`, `result/` 하위라 이미 gitignore)로 변경. 근거: `--add-symbols`는 `--input` CSV에 종목을 누적·재예측하므로, 입력이 publish의 200종목을 담고 있으면 온디맨드 1종목 요청이 200개를 재실행한다. 세션 입력을 분리하면 Colab/로컬 어느 환경에서도 요청 종목만 예측한다. 세션 입력은 베이스라인 표시와 무관(베이스라인은 `published/`에서 읽음)하며, 온디맨드로 요청된 종목만 점진 누적된다.
- 베이스라인 로더(신규/확장): `published_dir`에서 `csv/result_simple.csv`(+detail/news/disclosure)와 `manifest.json`을 읽어 운영 베이스라인으로 사용. `published/latest/manifest.json`은 publish가 운영 manifest를 복사하므로 `_is_operational_manifest` 통과.
- `_load_cached_result_simple()` 변경: **베이스라인(published) + 세션(`result/`) 행을 종목코드 기준 오버레이**. 동일 종목코드는 세션 행(최신화/온디맨드 결과)이 우선. 세션 소스가 없으면 베이스라인만 반환.
- `_resolve_result_path()` / detail·news·disclosure 조회도 동일 우선순위(세션 우선, 없으면 published)로 정렬.
- `_is_bootstrap_required()` → 항상 `False`(첫 요청 시 전 종목 실행 제거).
- `launch_colab_kakao_bot` 런치 경로에서 자동 prewarm 미실행(기본값 OFF 반영).
- 온디맨드 단일 종목 잡(`_start_prediction_job`)은 그대로 유지. 세션 입력 CSV는 요청 종목만 담아 작게 유지 → 1종목만 예측(200개 재실행 방지).
- 베이스라인 부재 시: 친절 안내("기준데이터 없음 — 종목 요청 시 예측") 후에도 온디맨드는 정상 동작.

### 5.4 Colab 러너 (`colab/stock_predict_colab.py`)
- 신규 `load_published_predictions(date: str | None = None) -> dict[str, str]`:
  - `date=None` → `published/latest/`, 날짜 지정 → `published/history/<date>/`에서 게시 세트 경로 반환 + 프리뷰 출력. 파이프라인 **미실행**.
  - 폴더 부재 시 안내 메시지 + `index.json` 기반 가용 날짜 출력.
- 기본 Colab 흐름 문서/예시를 "`git pull` → `load_published_predictions()` → `launch_colab_kakao_bot(prewarm_cache=False)`"로 갱신.
- 기존 `run_colab_pipeline(...)`는 사용자가 **명시적으로 "기본 200 재실행"을 요청할 때만** 호출하는 옵션 함수로 유지(요구사항 #3).

### 5.5 문서 + 테스트
- README: "Daily Publish" 섹션 추가, "Kakao Bot"/Colab 사용법을 publish 기반 흐름으로 갱신. `docs/PROJECT_OPERATION_GUIDE.md`/`OPERATIONS.md`에 publish 절차 추가.
- 테스트(네트워크 없이):
  - publish 산출물 복사·`publish_meta.json`·`index.json` 작성(파이프라인은 smoke/fixture 또는 `run_pipeline` mock).
  - 같은 거래일 재publish 시 history 폴더 덮어쓰기/인덱스 갱신.
  - 봇 오버레이: 세션 `result/` 행이 published 베이스라인 행을 덮는지.
  - 부트스트랩 기본 OFF: 런치/첫 요청 시 전 종목 실행이 일어나지 않는지(`_is_bootstrap_required()` False).
  - Colab `load_published_predictions(date)` latest/history 로드 및 부재 안내.

## 6. 에러 처리

- Publish:
  - gemma 무응답/오류 → 파이프라인이 규칙기반으로 폴백, `publish_meta.news_mode="rule_based"` 기록.
  - manifest 비운영(`pass`/`warning` 아님 또는 `promoted!=true`) → publish 중단, `published/` 미변경.
  - `git push` 실패 → 커밋은 남기고 실패를 보고(사용자가 수동 재시도). `--no-push`면 커밋까지만.
- 봇:
  - `published/latest/` 없음/스키마 불일치 → 베이스라인 빈 상태로 안내, 온디맨드는 동작.
  - 세션 오버레이 로드 실패 → 베이스라인만 서빙(degraded but functional).

## 7. 범위 밖 (YAGNI)

- 자동 스케줄러(cron/작업 스케줄러) 설정 — 사용자가 수동 실행.
- Colab에서 GitHub로의 push(온디맨드 결과 영속화) — 세션 한정.
- 봇이 임의 과거 날짜를 카톡으로 서빙 — 히스토리는 Colab 조회/아카이브용. 기본 서빙은 항상 `latest`.
- 예측 모델/신호 정책 변경 — 본 작업은 게시·서빙 경로 변경에 한정.
