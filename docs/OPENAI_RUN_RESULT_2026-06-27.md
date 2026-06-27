# OPENAI_RUN 실행 기록 (2026-06-27)

`docs/OPENAI_RUN.md`의 절차(OpenAI API 기반 LLM 파이프라인)를 실제로 실행한 경과·결과·소요시간 로그입니다.

- **실행 시작:** 2026-06-27 13:05:32 (KST)
- **실행 종료:** 2026-06-27 13:14:10 (KST)
- **총 소요(검증/조사 포함):** 약 8.6분 (핵심 파이프라인 실행은 169초 ≈ 2.8분)
- **실행 환경:** Windows 11, PowerShell 5.1, Python 3.14.5, CPU 16코어, 작업 디렉터리 `C:\Users\카운\Desktop\stock_predict`
- **LLM provider:** OpenAI (`https://api.openai.com/v1`), 모델 `gpt-5-mini`, config `configs/news_impact.openai.example.json`
- **run_id:** `20260627T040721Z_bbd95775`
- **git_commit:** `761dfe2`

> 참고: 본 출력은 연구/운영 참고용입니다. 매수/매도/보유 판단은 `predicted_return` 기준이며, 뉴스/공시 LLM 출력은 산출물에서 **"참고용·예측값 미반영"** 으로 표기됩니다.
>
> 운영 메모: PowerShell 도구는 호출 간 환경변수를 유지하지 않으므로, 키가 필요한 단계마다 `.env`를 재로딩했습니다.

---

## 단계별 실행 결과 요약

벽시계 기준 전체 실행은 **13:05:32 → 13:14:10 (총 약 8분 38초 ≈ 8.6분)** 입니다. 이 중 실제 **명령 실행 시간은 약 177초(≈ 3.0분)**, 나머지 약 5.6분은 단계 사이의 출력 점검·유니버스 CSV 생성·리포트 확인 등 조사/검증 시간입니다.

| 단계 | 내용 | 결과 | 명령 실행 소요 | 누적(명령 기준) |
| --- | --- | --- | --- | --- |
| 0 | `.env` 로드 / 키 확인 | OK (`OPENAI_API_KEY` 등 로드됨) | < 1s | ~1s |
| 3 | 의존성 설치 (`requirements.txt`, `-e .`) | OK (exit 0, 전부 이미 충족) | 6.1s | ~7s |
| 4 | OpenAI `llm-smoke` 연결 확인 | OK (`status: ok`, `gpt-5-mini`) | 1.0s | ~8s |
| 5 | 5종목 예측 파이프라인 | **OK (exit 0, status: pass)** | 169.3s | ~177s |
| 6 | Kakao 챗봇 앱 구성 검증 (서버 미기동) | OK (Flask 앱 생성, 라우트 정상) | 0.5s | ~177s |

> 단계 1·2 번호는 `OPENAI_RUN.md`의 명령 블록 주석 번호를 따릅니다. "명령 실행 소요"는 각 명령 자체의 실행 시간, "누적"은 명령 실행 시간만 합산한 값입니다(단계 사이 조사 시간 제외).

### 전체 타임라인 (벽시계, KST)

| 시각 | 경과(시작 기준) | 이벤트 |
| --- | --- | --- |
| 13:05:32 | +0:00 | 실행 시작 (`.env` 로드 / 키 확인) |
| 13:06:52 | +1:20 | 5종목 파이프라인 시작 (`python src/pipeline.py …`) |
| 13:08:17 / 13:08:45 / 13:09:05 | +2:45 ~ +3:33 | OpenAI 이슈 요약 신규 캐시 3건 생성 (issue summary 실제 호출 시각) |
| 13:09:41 | +4:09 | 파이프라인 종료 (status: pass, 169.3초 소요) |
| 13:14:10 | +8:38 | 전체 실행 종료 (산출물/리포트 점검 포함) |

### 5단계 파이프라인 내부 소요 (`diagnostics.timings_seconds`)

파이프라인 169.3초의 내역입니다. **최종 학습·예측(`train_final_and_predict_latest`)이 전체의 약 2/3를 차지**합니다.

| 내부 단계 | 소요 | 비중(파이프라인 169.3s 대비) |
| --- | --- | --- |
| load_config_and_inputs (설정/입력 로드) | 0.45s | 0.3% |
| prepare_context (외부/투자자/뉴스 컨텍스트 수집) | 24.68s | 14.6% |
| build_feature_matrix (피처 행렬 생성) | 1.60s | 0.9% |
| validation_and_tuning (워크포워드 5폴드 검증·튜닝) | 24.22s | 14.3% |
| train_final_and_predict_latest (최종 학습·예측) | 114.23s | 67.5% |
| save_pipeline_artifacts + 오버헤드 | ~4.1s | 2.4% |
| **합계** | **169.3s** | **100%** |

> 측정된 내부 단계 합(`timings_seconds`)은 165.18초이며, 나머지 약 4.1초는 산출물 저장(`save_pipeline_artifacts`, 별도 시간 미기록)과 단계 간 오버헤드입니다.

---

## 0) 사전 점검 — `.env` / API 키 로드

| 항목 | 결과 |
| --- | --- |
| `.env` 파일 | present |
| `OPENAI_API_KEY` | loaded (length=164) |
| `NGROK_AUTH_TOKEN` | loaded |
| `DART_API_KEY` | loaded |
| `NAVER_CLIENT_ID` | loaded |

> 키 값 자체는 출력/기록하지 않았고 길이만 확인했습니다. (과거 메모와 달리 현재 `.env`에 OpenAI 키가 활성 상태로 존재함.)

## 3) 의존성 설치

```
python -m pip install -r requirements.txt   # exit 0
python -m pip install -e .                   # exit 0
```

- 모든 패키지가 이미 설치되어 있어 신규 다운로드 없음(`openai 2.41.0`, `flask 3.1.3`, `pyngrok 8.1.2`, `pykrx 1.2.8` 등 충족).
- 참고 경고(동작에 영향 없음): 사용자 site-packages 설치(`Defaulting to user installation`), `*.exe` 스크립트 경로가 PATH 미등록, pip 업그레이드 알림(26.1.1 → 26.1.2).

## 4) OpenAI 연결/모델 접근 확인 (`llm-smoke`)

```
python -m src.news_impact.run llm-smoke --config configs/news_impact.openai.example.json
```

출력:

```json
{"base_url": "https://api.openai.com/v1", "model": "gpt-5-mini", "provider": "openai", "status": "ok"}
```

→ 키 유효, `gpt-5-mini` 접근 가능. (exit 0)

## 5) 5종목 예측 파이프라인

대상 종목(`data/universe_openai_5.csv` 생성 후 사용):

| 종목코드 | 종목명 |
| --- | --- |
| 005930.KS | 삼성전자 |
| 000660.KS | SK하이닉스 |
| 035420.KS | NAVER |
| 035720.KS | 카카오 |
| 051910.KS | LG화학 |

실행 명령:

```
python src/pipeline.py `
  --auto-refresh-real `
  --real-symbols $TargetSymbols `
  --universe-csv data/universe_openai_5.csv `
  --fetch-investor-context `
  --issue-summary-symbols $TargetSymbols `
  --llm-config configs/news_impact.openai.example.json `
  --report-json pipeline_report.json
```

- 결과: **exit 0 / `status: pass`**, 12단계 정상 진행, 13:06:52 → 13:09:41 (169.3초).
- `--auto-refresh-real`로 `data/real_ohlcv.csv`를 당일(2026-06-27) 기준 증분 갱신.
- `input_as_of_date: 2026-06-26`, **`prediction_for_date: 2026-06-29`**.

### 5-1) 예측 결과 (내일 종가/수익률 기준)

모든 종목 권고 = **매도** (예상 수익률 전부 음수).

| 종목 | 권고 | 내일 예상 종가 | 예상 수익률 | 상승확률 | 예측 신뢰도 | 횡단면 순위 |
| --- | --- | --- | --- | --- | --- | --- |
| 카카오 (035720) | 매도 | 33,296원 | -5.274% | 26.7% | 64.8% | 4/5 |
| NAVER (035420) | 매도 | 195,699원 | -3.833% | 27.9% | 57.0% | 1/5 |
| 삼성전자 (005930) | 매도 | 338,119원 | -5.023% | 27.1% | 44.9% | 3/5 |
| SK하이닉스 (000660) | 매도 | 2,664,287원 | -4.266% | 27.0% | 34.5% | 2/5 |
| LG화학 (051910) | 매도 | 280,217원 | -6.750% | 26.8% | 25.6% | 5/5 |

- PM 요약: 5종목 모두 `비중축소`. 리스크 플래그 `LOW_UP_PROB`×3, `HIGH_UNCERTAINTY|LOW_UP_PROB`×2.

### 5-2) 모델/검증 지표 (`result/latest/pipeline_report.json`)

- Walk-forward: MAE 0.019, RMSE 0.026, corr 0.061, accuracy 0.513, ROC-AUC 0.51 (folds 5/5).
- 베이스라인 대비: `baseline_zero` accuracy 0.475 / `baseline_prev_return` 0.494 → 모델 0.513로 소폭 우위.
- 튜닝된 시그널 가중치: return 0.6 / up_prob 0.35 / uncertainty_penalty 0.15.
- 백테스트(홀드아웃 227일): cum_return -0.093, Sharpe -2.035, benchmark +1.272. **단, 5종목만 대상이라 `liquidity_blocked_days 216/227`, `avg_selected_count 0.053`로 사실상 표본이 거의 없어 백테스트 수치는 해석 의미가 작음**(소수 종목 검증 한계).
- 확률 보정: fit 완료, eval Brier 0.26 / ECE 0.098.
- 커버리지 게이트: external 1.0, investor 0.5 → `caution`.

### 5-3) 외부 컨텍스트 수집 결과

| 항목 | 결과 |
| --- | --- |
| 외부 시장 피처 | 9/9 성공 (^KS11, ^KQ11, ^GSPC, ^IXIC, NQ=F, ^SOX, ^VIX, KRW=X, ^TNX) |
| 투자자 수급(flow) | **5/5 성공** (pykrx, 최신 2026-06-26) |
| 공시(disclosure) | 0/5 (수집 실패) |
| 뉴스(news) 신규 수집 | 0건 — **Naver 뉴스 API가 전 종목 HTTP 429 Too Many Requests 반환** |
| raw_events | partial_failure (collected 144, 5종목 모두 429) |

### 5-4) OpenAI 실제 사용 여부 (중요)

OpenAI는 이 실행에서 두 경로로 호출되는데, 실제 동작이 갈렸습니다.

1. **이슈 요약(issue summary) — OpenAI 실제 호출됨 ✅**
   - `result/runtime/llm_cache/issue_summary/`에 이번 실행 시각(13:08:17 / 13:08:45 / 13:09:05)으로 **신규 캐시 3건** 생성.
   - 뉴스가 있던 3종목(카카오·NAVER·LG화학)의 "뉴스 요약/공시 요약"이 OpenAI로 생성됨. 나머지 2종목(삼성전자·SK하이닉스)은 수집 뉴스가 없어 "수집된 뉴스가 없어…" 템플릿 처리.

2. **뉴스임팩트(news impact) 판정 — 규칙 기반으로 폴백 ⚠️**
   - 콘솔 로그: `[NEWS IMPACT][gemma] 유효 결과 없음 → 규칙 기반 폴백`
     - 참고: `[gemma]` 라벨은 `src/reports/news_impact_context.py`에 **하드코딩**된 표기로, provider가 OpenAI여도 동일하게 출력됩니다(라벨이 곧 gemma 사용을 의미하지 않음).
   - 리포트: `news_impact_runtime = { requested_mode: "gemma", actual_mode: "rule_based", fallback_used: true, fallback_reason: "gemma_no_valid_rows" }`.
   - 원인: 당일 신규 뉴스 수집이 **Naver 429**로 막히고 공시도 0/5라, 임팩트 판정에 넣을 유효 뉴스/공시 행이 없어 LLM이 유효 점수 행을 만들지 못함 → 규칙 기반 점수로 폴백. (`result/runtime/llm_cache/news_impact/`에 이번 실행 시각의 신규 캐시 없음 = 신규 임팩트 LLM 호출이 사실상 발생하지 않음.)
   - 산출물의 "뉴스/공시 영향 점수/요약"은 규칙 기반 결과이며, 컬럼 "뉴스/공시 영향 참고"에 **`참고용·예측값 미반영`** 으로 표기됨(가드레일 준수: 예측 수익률에 반영 안 됨).

> 요약: **OpenAI 라우팅 자체는 정상**(llm-smoke ok, 이슈 요약 신규 호출 발생). 다만 뉴스임팩트 판정은 입력 뉴스 부재(429)로 규칙 기반 폴백되었음. 이는 설정 오류가 아니라 외부 뉴스 API 레이트리밋에 따른 정상 폴백 동작입니다.

## 6) Kakao 챗봇 — 앱 구성 검증 (서버 미기동)

`OPENAI_RUN.md`의 6번 단계(ngrok 공개 webhook 서버)는 **블로킹/외부 공개 프로세스**라 자동 실행 세션에서 무기한 띄우는 것이 부적절하여 기동하지 않았습니다. 대신 동일 설정으로 앱이 정상 구성되는지만 검증했습니다.

```
create_app(runtime_config=PipelineRuntimeConfig(
    llm_config='configs/news_impact.openai.example.json',
    fetch_investor_context=True, use_external=True))
```

- 결과: `APP_OK type= Flask`, 라우트 `['/health', '/kakao/webhook', '/static/<path:filename>']`, exit 0 (0.5s).
- **실제 외부 연결로 챗봇을 띄우려면** `OPENAI_RUN.md` 1)의 6번 블록(ngrok 포함) 또는 2)의 로컬 전용 블록을 사용자가 직접 실행하세요. (`.env`에 `NGROK_AUTH_TOKEN`, `OPENAI_API_KEY` 필요. 현재 둘 다 로드 확인됨.)

---

## 산출물 위치

- 리포트: `result/latest/pipeline_report.json` (run 사본: `result/runs/20260627T040721Z_bbd95775/pipeline_report.json`)
- 예측 CSV(`utf-8-sig`):
  - `result/result_simple.csv`, `result/result_detail.csv`
  - `result/result_news.csv`, `result/result_disclosure.csv` (캐시/기수집 뉴스 포함)
  - run 사본: `result/runs/20260627T040721Z_bbd95775/csv/`
- PM 리포트: `result/latest/pm_report.json`
- 모델: `result/runs/20260627T040721Z_bbd95775/model/model.pkl`
- LLM 캐시: `result/runtime/llm_cache/issue_summary/`(신규 3건), `result/runtime/llm_cache/news_impact/`
- 입력: `data/universe_openai_5.csv`(생성), `data/real_ohlcv.csv`(증분 갱신)

## 비용 메모

- 이번 실행의 OpenAI 과금은 **이슈 요약 신규 3건** 수준(나머지 2종목은 뉴스 없음, 일부는 캐시 적중). 뉴스임팩트 LLM은 입력 부재로 신규 호출 없음.
- 동일 입력은 `issue_summary` 캐시로 재과금을 피합니다. 종목 확장 시 비용은 (종목 수)×(뉴스/공시 입력 길이)에 비례.

## 재현/후속 메모

- 뉴스 수집 429를 피하려면 시간 간격을 두고 재실행하거나 종목 수를 더 줄이세요. 뉴스가 정상 수집되면 뉴스임팩트가 규칙 기반이 아닌 LLM(OpenAI) 경로로 동작할 수 있습니다.
- 로컬 Gemma로 되돌리려면 `--llm-config configs/news_impact.gemma.example.json`로만 바꾸면 됩니다(`docs/LOCAL_RUN.md`).
