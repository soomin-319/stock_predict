# LOCAL_RUN 실행 기록 (2026-06-27)

`docs/LOCAL_RUN.md` 가이드에 따라 로컬에서 Gemma 서버 + 예측 파이프라인(대상 **5종목**)을 실행한 과정과 결과를 기록한다.

- 실행자: Claude Code (Opus 4.8)
- 실행 PC: Windows 11, NVIDIA GeForce RTX 5070 Ti (16GB VRAM), Python 3.14.5
- 작업 브랜치: `docs/openai-llm-provider-option-plan` (실행 시점 HEAD `21a7dfb`)
- LLM provider: 로컬 Gemma (`configs/news_impact.gemma.example.json`, alias `gemma-4-26b-a4b`, `http://localhost:8001/v1`)
- 대상 5종목: 삼성전자(005930), SK하이닉스(000660), NAVER(035420), 카카오(035720), LG화학(051910)

> 주의: 본 산출물은 연구/운영 참고용이다. 매수/매도/보유 판단은 `predicted_return`(내일 예상 수익률) 기준만 사용한다. 뉴스/공시 LLM 출력은 표시용 컨텍스트로만 다루며 예측값에 반영하지 않는다.

## 요약 (한눈에)

| 단계 | 결과 | 소요(경과) |
| --- | --- | --- |
| 0. 사전 환경 점검 | ✓ 통과 | 수 초 |
| 1. 의존성 설치 (`pip install`) | ✓ exit 0 (모두 already satisfied) | **7초** |
| 2. Gemma 서버 기동 | ✓ 포트 8001 오픈, 모델 VRAM 전량 적재 | 포트 바인딩 **~3초**, 가중치 적재 포함 **수십 초** |
| 3. `llm-smoke` 연결 확인 | ✓ `status: ok` (exit 0) | 수 초 |
| 4. 5종목 예측 파이프라인 | ✓ `status: pass` (exit 0) | **596초 (9분 56초)** |
| 합계 (서버 기동 → 파이프라인 완료) | — | **약 12분 06초** (16:53:52 → 17:05:58 KST) |

핵심: `news_impact_runtime`은 `requested=gemma → actual=gemma`, `fallback_used=false` 로 **OpenAI 폴백 없이 전 구간 로컬 Gemma**로 수행되었다. 5/5 종목 예측 생성(누락 0). 다만 뉴스 수집 단계에서 네이버 뉴스 API가 **HTTP 429(Too Many Requests)** 로 전 종목 레이트리밋되어 뉴스/공시 컨텍스트는 일부만 채워졌다(아래 §4-2, §6 참조). 뉴스/공시는 표시용이라 예측·`status`에는 영향 없음.

## 0. 사전 환경 점검

| 항목 | 확인 결과 |
| --- | --- |
| `.env` 존재 | ✓ (값 미출력) |
| Python | 3.14.5 |
| `llama-server.exe` (winget) | ✓ 존재 |
| Gemma 모델 `models\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` | ✓ 존재 (12.66 GB) |
| `configs/news_impact.gemma.example.json` | ✓ 존재 |
| `requirements.txt` / `src/pipeline.py` | ✓ 존재 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16303 MiB (점검 시 사용 1879 MiB) |
| 8001 포트(Gemma) | 닫힘 → 신규 기동 필요 |

가이드의 한글 경로 버그 대응에 따라 모델은 **상대경로 + `-WorkingDirectory`** 로 기동했다(한글 사용자명 절대경로를 `llama-server`의 `-m` 인자로 넘기면 네이티브 exe가 ANSI 코드페이지로 받아 깨져 모델 로드 실패).

## 1. 의존성 설치

```powershell
python -m pip install -r requirements.txt   # exit 0
python -m pip install -e .                   # exit 0
```

- 두 명령 모두 **exit 0**. 전부 `already satisfied` 라 신규 다운로드 없음. **경과 약 7초.**
- `pip install -e .` 시 콘솔 스크립트(`stock-predict.exe` 등)가 `%APPDATA%\Python\Python314\Scripts`(PATH 미등록)에 설치된다는 경고가 떴으나 실행에는 무해.

## 2. Gemma 서버 기동

가이드의 한글 경로 버그 대응대로 모델을 **상대경로**로 지정하고 `-WorkingDirectory $PWD`로 기동했다.

```powershell
$LlamaServer = '...\WinGet\...\llama-server.exe'
$GemmaModel  = 'models\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf'   # 상대경로
Start-Process -FilePath $LlamaServer -ArgumentList @(
  '-m', $GemmaModel, '--host','127.0.0.1','--port','8001',
  '--alias','gemma-4-26b-a4b','-c','8192','-ngl','99','-fa','on',
  '--jinja','--reasoning','off'
) -WorkingDirectory $PWD -WindowStyle Hidden
```

- 기동 시작 **16:53:52** → TCP 8001 바인딩 **16:53:55(~3초)**. 단, 포트 바인딩 직후 GPU 사용량은 897 MiB로 아직 가중치 적재 중이었다.
- 잠시 후 검증 시 GPU 메모리 **15448 MiB / 16303 MiB** 로 안정 → 전 레이어(`-ngl 99`) GPU 오프로딩으로 모델이 VRAM에 전량 적재됨(OOM 없음). 프로세스 워킹셋 ~13.2 GB.
- `/v1/models` 프로브가 HTTP 200으로 `gemma-4-26b-a4b`(파라미터 25.2B, 파일 13.58 GB, `n_ctx 8192`) 메타를 반환해 정상 적재 확인.

> 가이드의 폴링 루프는 "포트 오픈"을 준비 완료 신호로 쓰지만, 실제로는 포트 바인딩이 가중치 적재보다 먼저 일어날 수 있다. 본 실행에서는 적재 완료를 GPU 메모리/`/v1/models` 응답으로 별도 확인했다.

## 3. Gemma 엔드포인트 연결 확인 (llm-smoke)

```powershell
python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json
```

결과 (exit `0`):

```json
{"base_url": "http://localhost:8001/v1", "model": "gemma-4-26b-a4b", "provider": "llama_cpp", "status": "ok"}
```

## 4. 5종목 예측 파이프라인 실행

`data/universe_gemma_5.csv`(헤더 `Symbol` + 5종목)를 기록한 뒤, 실데이터 자동 갱신 + 투자자/뉴스/공시 컨텍스트 + Gemma 뉴스임팩트로 실행했다. `.env`는 프로세스 환경변수로 로드(값 미출력)해 DART/네이버 키를 자식 프로세스에 전달했다.

```powershell
python src/pipeline.py `
  --auto-refresh-real --real-symbols $TargetSymbols `
  --universe-csv data/universe_gemma_5.csv `
  --fetch-investor-context --issue-summary-symbols $TargetSymbols `
  --llm-config configs/news_impact.gemma.example.json `
  --report-json pipeline_report.json
```

### 4-1. 실행 경과

- **16:56:02 시작 → 17:05:58 완료 (exit 0). 총 경과 596초 = 9분 56초.**
- `[1/12]~[11/12]`(설정/데이터 로드/정제/투자자 컨텍스트/피처/외부피처/walk-forward 검증/베이스라인/OOF/신호가중 튜닝/백테스트)는 시작 후 약 **2분 이내**에 모두 통과.
- 이전 10종목 실행에서 크래시했던 `[10/12] Tuning signal weights` 는 이번에 **정상 통과**. 직전 세션에서 적용한 캘리브레이션 정렬 수정(`src/validation/support.py:129`, `pd.Series(..., index=raw.index, ...)`)이 현재 브랜치에 반영되어 있어 회귀 없음(empirical 확인).
- 나머지 시간(약 8분)은 `[12/12] Training final model and creating latest predictions` 단계의 **5종목 Gemma 뉴스/이슈요약 LLM 호출**이 차지. (종목당 약 1.6분 — 본 실행은 뉴스 수집이 429로 일부 비어 직전 10종목 실행의 종목당 ~2.8분보다 빨랐다.)

### 4-2. 리포트 요약 (`result/latest/pipeline_report.json`)

- `run_id`: `20260627T075638Z_267779bb`, `git_commit`: `21a7dfb`, `generated_at`: `2026-06-27T07:56:38Z`(=16:56:38 KST, 실행 시작 기준)
- `status: pass`, `blocking_reasons: []`
- `universe_size_used: 5`, `feature_count: 111`
- `prediction_coverage`: requested 5 / row 5 / available 5 / **missing 0** / tradable 5 → 5/5 예측 생성
- `news_impact_runtime`: `requested_mode=gemma`, `actual_mode=gemma`, `fallback_used=false` → **OpenAI 폴백 없이 로컬 Gemma로 수행**
- `probability_calibration.fit`: `status=fitted`, tail_shrinkage `enabled=true (strength 0.5, support 0.108~0.918)` → 수정된 캘리브레이션 경로 정상 동작
- `data_fetch_coverage`: 5/5 성공(OHLCV 자동 갱신 정상, 폴백/재시도 0)
- `investor_context_coverage`: 투자자 수급 flow **5/5 성공(pykrx, latest_flow_date 2026-06-26)**; 공시(disclosure) 0/5 성공
- `coverage_gate`: `status=caution` (external_coverage 1.0, **investor_coverage 0.5**, min_value_traded 3.0e9) — 차단(block) 아님

#### 뉴스 수집 레이트리밋(주의)

- `investor_context_coverage.raw_events`: `status=partial_failure`, `collected=131`, `failed_symbols=[5종목 전체]`, `error_types=["HTTPError"]`.
- 원인: **네이버 뉴스 API `HTTP Error 429: Too Many Requests`** 가 전 종목 반복 발생. 그 결과 일부 종목은 뉴스/공시 요약이 비었다.
  - 뉴스 요약 채워짐: 카카오(035720), NAVER(035420)
  - 뉴스 없음("수집된 뉴스가 없어 …"): 삼성전자(005930), SK하이닉스(000660), LG화학(051910)
- 영향 범위: 뉴스/공시는 **표시용 컨텍스트**이므로 `predicted_return`·권고·`status`에 영향 없음. 재시도가 필요하면 네이버 API 호출 간격을 늘리거나 잠시 후 재실행 권장.

## 5. 산출물

`result/` 아래 생성/갱신 (CSV는 `utf-8-sig`):

| 파일 | 크기 | 내용 |
| --- | --- | --- |
| `result/result_simple.csv` | 2.9 KB | 종목별 예측 요약(권고/예상수익률/상승확률/신뢰도/뉴스·공시 요약) |
| `result/result_detail.csv` | 15.2 KB | 예측 상세 |
| `result/result_news.csv` | 159 KB | 뉴스 컨텍스트 |
| `result/result_disclosure.csv` | 3.7 KB | 공시 컨텍스트 |
| `result/latest/pipeline_report.json` | 25.2 KB | 최신 실행 리포트 (`--report-json pipeline_report.json` 경로에도 별도 저장) |
| `result/latest/manifest.json` | 11.5 KB | 매니페스트 |
| `result/pm_report.json` | 1.1 KB | PM 리포트 |

### 예측 결과 (5종목, `result_simple.csv` 기준)

- 입력 기준일 `input_as_of_date=2026-06-26`, 예측 대상일 `prediction_for_date=2026-06-29`(차익 거래일), 컨텍스트 기준일 2026-06-26.
- 권고는 `predicted_return`(내일 예상 수익률) 기준. 뉴스/공시 영향 점수 컬럼은 표시용(`참고용·예측값 미반영`)이며 계산에 반영하지 않는다.

| 종목코드 | 종목명 | 권고 | 내일 예상 종가 | 예상 수익률 | 상승확률 | 예측 신뢰도 | 횡단면 순위 | 랭킹 백분위 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 035720 | 카카오 | 매도 | 33,296원 | -5.274% | 26.7% | 64.8% | 4/5 | 40.0% |
| 035420 | NAVER | 매도 | 195,699원 | -3.833% | 27.9% | 57.0% | 1/5 | 100.0% |
| 005930 | 삼성전자 | 매도 | 338,119원 | -5.023% | 27.1% | 44.9% | 3/5 | 60.0% |
| 000660 | SK하이닉스 | 매도 | 2,664,287원 | -4.266% | 27.0% | 34.5% | 2/5 | 80.0% |
| 051910 | LG화학 | 매도 | 280,217원 | -6.750% | 26.8% | 25.6% | 5/5 | 20.0% |

> 5종목 모두 내일 예상 수익률이 음수로 산출되어 권고는 **매도 5**. 본 수치는 연구·참고용이며 투자 자문이 아니다.

- 뉴스/공시 영향 점수(표시용): 카카오 `-2.0점`, NAVER `-48.9점`(전반적 시장 급락 기사 반영). 나머지 3종목은 뉴스 미수집으로 `-`. 모두 `참고용·예측값 미반영`.

## 6. 시간 경과 (타임라인)

기준 시각: KST. 파이프라인 내부 단계 소요는 실행 로그 + 진행 중 관찰(시작 후 ~2분 시점에 이미 `[12/12]` 진입) 기준 추정.

| 시각 | 이벤트 | 소요(경과) |
| --- | --- | --- |
| (직전) | `pip install -r` / `-e .` | 약 7초 (exit 0) |
| 16:53:52 | Gemma `llama-server` 기동 시작 | — |
| 16:53:55 | TCP 8001 바인딩 | 약 3초 |
| 16:53:55 이후 | 가중치 VRAM 전량 적재 완료(15448 MiB), `/v1/models` 200, `llm-smoke` `status: ok` | 수십 초 |
| 16:56:02 | 파이프라인 시작 | — |
| ~16:58 | `[1/12]→[11/12]` 통과(`[10/12]` 정상), `[12/12]` 진입 | 약 2분 |
| 16:58 ~ 17:05:58 | `[12/12]` 5종목 Gemma 뉴스/이슈요약 + 아티팩트 저장 | 약 8분 |
| 17:05:58 | 파이프라인 **완료 (exit 0)** | 파이프라인 총 **596초(9분 56초)** |
| — | **서버 기동 → 파이프라인 완료 합계** | **약 12분 06초** |

> 관찰: 가격 피처·walk-forward 검증·백테스트 등 `[1/12]~[11/12]`은 약 2분 내외로 끝나고, **전체 소요의 대부분(약 8분/9분 56초)은 `[12/12]`의 5종목 Gemma LLM 호출**이 차지한다. 종목 수에 거의 선형 비례한다(직전 10종목 실행은 약 29분, 본 5종목 실행은 약 10분).

## 7. 종료 처리

- Gemma `llama-server`(포트 8001) 프로세스를 종료했다(`Get-Process llama-server | Stop-Process`).
- 본 실행 기록(이 문서)은 현재 브랜치에 커밋·푸시 후 PR로 정리한다.
- 참고: 스텝 7(Kakao 챗봇 + ngrok 외부 webhook)은 `server_thread.join()`으로 무한 대기하는 상주 서버라 본 자동 실행 범위에서 제외했다. 필요 시 `docs/LOCAL_RUN.md` §1의 마지막 블록 또는 §2(로컬 전용)를 수동 실행하면 된다.
