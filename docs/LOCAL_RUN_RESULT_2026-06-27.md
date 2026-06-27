# LOCAL_RUN 실행 기록 (2026-06-27)

`docs/LOCAL_RUN.md` 가이드에 따라 로컬에서 Gemma 서버 + 예측 파이프라인을 실행한 과정과 결과를 기록한다.

- 실행자: Claude Code (Opus 4.8)
- 실행 PC: Windows 11, RTX 5070 Ti (16GB VRAM), Python 3.14.5
- 작업 브랜치: `docs/openai-llm-provider-option-plan`
- LLM provider: 로컬 Gemma (`configs/news_impact.gemma.example.json`)

> 주의: 본 산출물은 연구/운영 참고용이다. 매수/매도/보유 판단은 `predicted_return` 기준만 사용한다. 뉴스/공시 LLM 출력은 표시용 컨텍스트로만 다룬다.

## 0. 사전 환경 점검

| 항목 | 확인 결과 |
| --- | --- |
| `.env` 존재 | ✓ |
| Python | 3.14.5 |
| `llama-server.exe` (winget) | ✓ 존재 |
| Gemma 모델 `models\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` | ✓ 존재 (12.66 GB) |
| `configs/news_impact.gemma.example.json` | ✓ 존재 |
| `requirements.txt` / `src/pipeline.py` | ✓ 존재 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16303 MiB (점검 시 사용 995 MiB) |
| 8001 포트(Gemma) | 닫힘 → 신규 기동 필요 |

가이드의 한글 경로 버그 대응에 따라 모델은 **상대경로 + `-WorkingDirectory`** 로 기동한다.

## 1. 의존성 설치

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

- 두 명령 모두 exit code `0`.
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

- 8001 포트가 약 수십 초 후 정상 오픈 (모델 GPU 로딩).
- GPU 메모리: **15053 MiB / 16303 MiB** 사용 → 전 레이어(`-ngl 99`) GPU 오프로딩 정상 동작 (OOM 없음).

## 3. Gemma 엔드포인트 연결 확인 (llm-smoke)

```powershell
python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json
```

결과 (exit `0`):

```json
{"base_url": "http://localhost:8001/v1", "model": "gemma-4-26b-a4b", "provider": "llama_cpp", "status": "ok"}
```

## 4. 10종목 예측 파이프라인 실행

대상 10종목(삼성전자 005930, SK하이닉스 000660, NAVER 035420, 카카오 035720, LG화학 051910, 현대차 005380, 기아 000270, 셀트리온 068270, LG에너지솔루션 373220, KB금융 105560)에 대해 실데이터 자동 갱신 + 투자자/뉴스/공시 컨텍스트 + Gemma 뉴스임팩트로 실행했다.

```powershell
python src/pipeline.py `
  --auto-refresh-real --real-symbols $TargetSymbols `
  --universe-csv data/universe_gemma_10.csv `
  --fetch-investor-context --issue-summary-symbols $TargetSymbols `
  --llm-config configs/news_impact.gemma.example.json `
  --report-json pipeline_report.json
```

### 4-1. 첫 실행 실패 → 코드 버그 발견 및 수정

첫 실행은 `[10/12] Tuning signal weights` 단계에서 다음 예외로 중단됐다 (exit 1):

```
pandas.errors.IndexingError: Unalignable boolean Series provided as indexer
  (index of the boolean Series and of the indexed object do not match).
  at src/validation/support.py:141  (_shrink_sparse_tails)
```

**근본 원인** (`UpProbabilityCalibrator.transform`, `src/validation/support.py`):

- `raw = pd.Series(probabilities, ...)` 는 입력이 `frame["up_probability"]`(Series)일 때 **원본 프레임의 인덱스**를 유지한다. walk-forward eval split의 인덱스는 비연속이다.
- `calibrated = pd.Series(model.predict(raw.values), ...)` 는 numpy 배열로 새로 만들어 **기본 RangeIndex(0..n-1)** 를 갖는다.
- `_shrink_sparse_tails`에서 `tail_mask`(=raw 인덱스)로 `shrunk`(=calibrated, RangeIndex)를 `.loc` 인덱싱하면 인덱스가 정렬되지 않아 크래시한다.
- `tune` 메트릭은 calibrator를 자기 자신으로 fit 해 support 범위 밖 확률이 없으므로 early-return(138행)으로 통과하지만, `eval`은 tune support 범위 밖 확률이 생겨 141행에서 터진다.
- 부수적으로 130행 `0.3*calibrated + 0.7*raw` 도 인덱스 불일치 시 조용히 NaN을 만들 수 있는 잠복 버그였다.

**수정** (`src/validation/support.py`, 최소 diff):

```python
# before
calibrated = pd.Series(self.model.predict(raw.values), dtype=float).clip(0.0, 1.0)
# after
calibrated = pd.Series(
    self.model.predict(raw.values), index=raw.index, dtype=float
).clip(0.0, 1.0)
```

`calibrated`에 `raw.index`를 부여해 `raw`/`calibrated`/`tail_mask`/`shrunk`가 모두 동일 인덱스를 공유하도록 정렬 문제를 근원에서 제거했다.

**회귀 방지 테스트** (`tests/test_probability_calibration_guard.py`):

- `test_transform_aligns_non_default_index_with_sparse_tails` — 비기본 인덱스 + sparse tail 입력에서 정렬·shrink 검증
- `test_calibration_report_handles_eval_split_with_non_default_index` — eval split 비연속 인덱스 경로 검증

수정 전 두 테스트는 동일한 `IndexingError`로 실패(RED) → 수정 후 통과(GREEN).

**전체 테스트**: `python -m pytest tests` → **516 passed** (회귀 없음).

### 4-2. 재실행 결과

수정 후 재실행은 이전에 크래시하던 `[10/12] Tuning signal weights`를 정상 통과했다.

- `[1/12] ~ [12/12]` 단계 진입까지 약 **78초** (12:12:00 → 12:13:18).
- 마지막 `[12/12] Training final model and creating latest predictions` 단계에서 10종목 Gemma 이슈요약을 수행 — 이 단계가 전체 실행 시간의 대부분을 차지한다(아래 타임라인 참조).
- **12:41:11 완료 (exit 0).** 총 소요 약 **29분** (대부분 `[12/12]` Gemma 이슈요약).
- 리포트 요약(`result/latest/pipeline_report.json`):
  - `status: pass`, `blocking_reasons: []`
  - `universe_size_used: 10`, `feature_count: 111`, 예측 커버리지 10/10 (누락 0)
  - `news_impact_runtime`: `requested_mode=gemma`, `actual_mode=gemma`, `fallback_used=false` → **OpenAI 폴백 없이 로컬 Gemma로 수행됨**
  - `probability_calibration.fit`: `status=fitted`, tail_shrinkage `enabled=true (strength 0.5, support 0.124~0.88)` → 수정한 캘리브레이션 경로가 정상 동작

## 5. 산출물

`result/` 아래 생성/갱신 (모두 12:41:10 기록, CSV는 `utf-8-sig`):

| 파일 | 크기 | 내용 |
| --- | --- | --- |
| `result/result_simple.csv` | 4.4 KB | 종목별 예측 요약(권고/예상수익률/상승확률/신뢰도/뉴스·공시 요약) |
| `result/result_detail.csv` | 23.2 KB | 예측 상세 |
| `result/result_news.csv` | 381 KB | 뉴스 컨텍스트 |
| `result/result_disclosure.csv` | 7.3 KB | 공시 컨텍스트 |
| `result/latest/pipeline_report.json` | 32.1 KB | 최신 실행 리포트 (`--report-json pipeline_report.json` 경로에도 별도 저장) |
| `result/latest/manifest.json`, `result/pm_report.json` | — | 매니페스트/PM 리포트 |

### 예측 결과 (10종목, `result_simple.csv` 기준)

- 입력 기준일 `input_as_of_date=2026-06-26`, 예측 대상일 `prediction_for_date=2026-06-29`(차익 거래일).
- 권고는 `predicted_return`(내일 예상 수익률) 기준. 뉴스/공시 영향 점수 컬럼은 모두 `-`(표시용 컨텍스트, 계산 미반영).

| 종목코드 | 종목명 | 권고 | 예상 종가 | 예상 수익률 | 상승확률 | 신뢰도 | 횡단면 순위 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 105560 | KB금융 | 매도 | 149,085원 | -2.495% | 15.4% | 70.9% | 4/5 |
| 068270 | 셀트리온 | 매도 | 167,531원 | -2.485% | 25.0% | 65.8% | 2/5 |
| 000270 | 기아 | 관망 | 137,946원 | -1.957% | 38.0% | 60.7% | 1/5 |
| 035720 | 카카오 | 매도 | 33,437원 | -4.874% | 25.0% | 54.9% | 4/5 |
| 035420 | NAVER | 매도 | 193,942원 | -4.697% | 27.3% | 51.0% | 1/5 |
| 005930 | 삼성전자 | 매도 | 339,319원 | -4.686% | 15.4% | 43.1% | 2/5 |
| 005380 | 현대차 | 매도 | 497,705원 | -2.981% | 44.7% | 39.3% | 3/5 |
| 373220 | LG에너지솔루션 | 매도 | 349,422원 | -4.660% | 38.0% | 36.2% | 5/5 |
| 000660 | SK하이닉스 | 매도 | 2,671,521원 | -4.006% | 15.4% | 29.0% | 3/5 |
| 051910 | LG화학 | 매도 | 282,168원 | -6.101% | 0.0% | 23.8% | 5/5 |

> 10종목 모두 내일 예상 수익률이 음수로 산출되어 권고는 매도 9 / 관망 1. 본 수치는 연구·참고용이며 투자 자문이 아니다.

## 6. 시간 경과 (타임라인)

기준 시각: KST. 단계별 소요는 백그라운드 작업 출력 파일/프로세스 시작시각 기준 추정.

| 시각 | 이벤트 | 소요(추정) |
| --- | --- | --- |
| 12:05:46 | Gemma `llama-server` 기동 시작 (모델 GPU 로딩) | — |
| ~12:06 | 8001 포트 오픈 = 모델 로딩 완료 | 약 30~60초 |
| ~12:06 | `llm-smoke` 연결 확인 (`status: ok`) | 수 초 |
| 12:07:33 | 파이프라인 **1차 실행** 시작 | — |
| 12:08:48 | 1차 실행 **실패** (`[10/12]` IndexingError) | 약 75초 |
| 12:08~12:12 | 버그 진단 → 실패 테스트 작성 → 수정 → 전체 516 테스트 통과 | 약 3~4분 |
| 12:12:00 | 파이프라인 **재실행** 시작 | — |
| 12:13:18 | `[1/12]→[12/12]` 단계 진입 (예측/검증/백테스트까지) | 약 78초 |
| 12:13:18 ~ 12:41:11 | `[12/12]` 10종목 Gemma 이슈요약 + 아티팩트 저장 | 약 28분 |
| 12:41:11 | 재실행 **완료 (exit 0)** | 재실행 총 약 29분 |

> 관찰: 가격 피처 생성·walk-forward 검증·백테스트 등 `[1/12]~[11/12]` 까지는 약 1분 내외로 끝나고, **전체 소요 시간의 대부분(약 28분/29분)은 `[12/12]`의 10종목 Gemma 뉴스/이슈요약 LLM 호출**이 차지한다. 종목 수에 거의 선형으로 비례하므로 종목 수를 줄이면 전체 시간도 비례해 줄어든다.

## 7. 후속 요청: 대상 10 → 5종목 축소

사용자 요청으로 기본 실행 대상을 10종목에서 **5종목**으로 줄였다.

- 대상 5종목(목록 순 앞 5개): 삼성전자(005930), SK하이닉스(000660), NAVER(035420), 카카오(035720), LG화학(051910).
- `docs/LOCAL_RUN.md` 의 `$TargetSymbols` 를 5개로 축소하고 universe 경로를 `data/universe_gemma_5.csv` 로 변경. 기존 `data/universe_gemma_10.csv` 는 제거.
- **5종목 재실행은 사용자 종료 지시에 따라 수행하지 않았다.** 따라서 현재 `result/` 산출물은 위 §5의 **10종목 실행 결과**가 그대로 유지된다. 5종목 기준 산출물이 필요하면 수정된 LOCAL_RUN.md의 명령을 그대로 실행하면 된다(예상 소요 약 15분 내외 — Gemma 단계가 종목 수에 비례).

## 8. 종료 처리

- Gemma `llama-server`(포트 8001) 프로세스를 종료했다(`Get-Process llama-server | Stop-Process`).
- 코드/문서 변경분은 현재 브랜치에 커밋·푸시 후 PR 생성(아래 핸드오프 참조).
