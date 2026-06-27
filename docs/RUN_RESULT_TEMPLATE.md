# RUN 결과 기록 템플릿 (재사용)

`docs/LOCAL_RUN.md` / `docs/OPENAI_RUN.md` 절차를 실제 실행한 뒤, 과정·결과·소요시간을 기록할 때 쓰는 공통 골격입니다. 두 provider 기록을 **같은 섹션 구성**으로 맞춰 두면 나란히 비교하기 쉽고, `docs/local-vs-openai-...comparison` 문서로 합치기도 편합니다.

## 사용법

1. 이 파일을 복사해 결과 문서를 만든다.
   - 이름: `LOCAL_RUN_RESULT_<YYYY-MM-DD>.md` 또는 `OPENAI_RUN_RESULT_<YYYY-MM-DD>.md`
   - 같은 날 재실행이면 이름이 겹치므로 `..._<YYYY-MM-DD>_run2.md`처럼 순번을 붙이거나 run_id 앞 6자리를 접미사로 쓴다.
   - 위치: 작성/검토 중에는 `docs/`, 이력만 남기는 시점에 `docs/archive/`로 이동.
2. `<...>`는 실제 값으로 채우고, `<!-- ... -->` 작성 가이드 주석은 지운다.
3. `<!-- LOCAL 전용 -->` / `<!-- OPENAI 전용 -->`로 표시된 블록은 해당 provider 문서에만 남긴다.
4. §1·§4·§5·§6의 **표 컬럼·순서는 두 문서에서 동일하게** 유지한다(비교 정렬용). 특히 §4의 `news_impact_runtime`(requested/actual/fallback)은 위치를 맞춘다.

> 가드레일: 본 산출물은 연구/운영 참고용이다. 매수/매도/보유 판단은 `predicted_return`(내일 예상 수익률) 기준만 사용한다. 뉴스/공시 LLM 출력은 표시용 컨텍스트이며 예측값에 반영하지 않는다.

---

# 아래부터 복사해서 사용 ↓

````markdown
# <LOCAL_RUN | OPENAI_RUN> 실행 기록 (<YYYY-MM-DD>)

`docs/<LOCAL_RUN.md | OPENAI_RUN.md>` 가이드 절차를 실제 실행한 과정·결과·소요시간 기록.

- 실행자: Claude Code (Opus 4.8)  <!-- 또는 본인 -->
- 실행 시작 / 종료: <13:05:32> → <13:14:10> (KST)
- 실행 환경: Windows 11, PowerShell 5.1, Python <3.14.5>, <CPU 코어 / GPU 모델·VRAM>
- 작업 브랜치: `<branch>` (실행 시점 HEAD `<short-sha>`)
- LLM provider: <로컬 Gemma `gemma-4-26b-a4b` @ http://localhost:8001/v1 | OpenAI `gpt-5-mini` @ https://api.openai.com/v1>
- config: `configs/news_impact.<gemma|openai>.example.json`
- run_id: `<...>`  /  git_commit: `<...>`

> 주의: 본 산출물은 연구/운영 참고용이다. 매수/매도/보유 판단은 `predicted_return`(내일 예상 수익률) 기준만 사용한다. 뉴스/공시 LLM 출력은 표시용 컨텍스트이며 예측값에 반영하지 않는다.

---

## 1. 요약 (한눈에)

| 단계 | 결과 | 소요(경과) |
| --- | --- | --- |
| 0. 사전 환경 점검 | <✓ 통과> | <수 초> |
| 1. 의존성 설치 | <✓ exit 0> | <…> |
| 2. LLM 서버 기동 <!-- LOCAL 전용; OPENAI는 "N/A (API)" --> | <✓ 포트 8001> | <…> |
| 3. `llm-smoke` 연결 확인 | <✓ status: ok> | <…> |
| 4. N종목 예측 파이프라인 | <✓ status: pass (exit 0)> | <…초> |
| 합계 | — | <약 …분> |

**핵심 한 줄:** <provider 실제 사용 여부(requested/actual/fallback) + 예측 N/N 생성 + 특이사항(예: 네이버 429) 한 문장>

## 2. 사전 환경 점검

| 항목 | 확인 결과 |
| --- | --- |
| `.env` 존재 / 키 로드 | <present; OPENAI/DART/NAVER loaded — 값 미출력, 길이만 확인> |
| Python | <3.14.5> |
| <provider별 점검: llama-server.exe·모델 gguf 존재 / OpenAI 키 유효> | <…> |
| 8001 포트 <!-- LOCAL 전용 --> | <닫힘 → 기동 필요> |

<!-- LOCAL 전용: 한글 경로 버그 대응(모델은 상대경로 + -WorkingDirectory로 기동)했음을 한 줄 메모 -->

## 3. 실행 단계

### 3-1. 의존성 설치

```powershell
python -m pip install -r requirements.txt   # exit 0
python -m pip install -e .                   # exit 0
```

<결과·경과·무해 경고 요약>

<!-- ### 3-2. LLM 서버 기동 (LOCAL 전용) -->
<!-- 기동 명령(모델 상대경로 + -WorkingDirectory), 포트 바인딩 시각/소요, GPU VRAM 적재 확인(-ngl 99), /v1/models 200 응답 -->

### 3-3. 연결 확인 (`llm-smoke`)

```powershell
python -m src.news_impact.run llm-smoke --config configs/news_impact.<gemma|openai>.example.json
```

```json
<{"base_url": "...", "model": "...", "provider": "...", "status": "ok"}>
```

### 3-4. N종목 예측 파이프라인

대상 N종목: <삼성전자(005930), SK하이닉스(000660), …>

```powershell
python src/pipeline.py `
  --auto-refresh-real --real-symbols $TargetSymbols `
  --universe-csv data/universe_<...>.csv `
  --fetch-investor-context --issue-summary-symbols $TargetSymbols `
  --llm-config configs/news_impact.<gemma|openai>.example.json `
  --report-json pipeline_report.json
```

- 결과: <exit 0 / status: pass>, <시작 → 종료 시각>, <…초>
- `input_as_of_date`: <…> / `prediction_for_date`: <…>

## 4. 리포트 요약 (`result/latest/pipeline_report.json`)

- status / blocking_reasons: <pass / []>
- universe_size_used / feature_count: <5 / 111>
- prediction_coverage: requested/row/available/**missing**/tradable = <5/5/5/0/5>
- **news_impact_runtime**: requested=`<gemma>` actual=`<gemma|rule_based>` fallback_used=`<true|false>` <fallback_reason>
- probability_calibration: <fit status, tail_shrinkage, eval Brier/ECE>
- walk-forward: <MAE/RMSE/corr/accuracy/ROC-AUC, folds>
- coverage_gate: <status, external/investor coverage>

## 5. 예측 결과 (N종목, `result_simple.csv`)

| 종목코드 | 종목명 | 권고 | 내일 예상 종가 | 예상 수익률 | 상승확률 | 예측 신뢰도 | 횡단면 순위 |
| --- | --- | --- | --- | --- | --- | --- |
| <035720> | <카카오> | <매도> | <33,296원> | <-5.274%> | <26.7%> | <64.8%> | <4/5> |

> <권고 분포 한 줄(예: 매수 N / 매도 N)>. 뉴스/공시 영향 점수는 표시용(`참고용·예측값 미반영`).

## 6. 외부 컨텍스트 수집

| 항목 | 결과 |
| --- | --- |
| 외부 시장 피처 | <9/9> |
| 투자자 수급(flow) | <5/5 (pykrx, 최신 …)> |
| 공시(disclosure) | <0/5> |
| 뉴스 신규 수집 | <0건 — Naver 429 / 정상 수집 N건> |

<429 등 특이사항·영향 범위(표시용이라 예측·status 무영향) 메모>

## 7. 산출물 위치

| 파일 | 내용 |
| --- | --- |
| `result/latest/pipeline_report.json` | 실행 리포트 (run 사본: `result/runs/<run_id>/pipeline_report.json`) |
| `result/result_simple.csv` 외 | 예측/상세/뉴스/공시 CSV (`utf-8-sig`) |
| `result/latest/pm_report.json` | PM 리포트 |

## 8. 타임라인 (벽시계, KST)

| 시각 | 경과 | 이벤트 |
| --- | --- | --- |
| <13:05:32> | +0:00 | 실행 시작 |
| <…> | <…> | <…> |
| <13:09:41> | <+…> | 파이프라인 종료 |

## 9. provider 메모

<!-- LOCAL 전용: 종료 처리 — llama-server(포트 8001) 프로세스 종료, 챗봇/ngrok 단계 제외 사유 -->
<!-- OPENAI 전용: 비용 메모 — 이슈요약/뉴스임팩트 신규 호출 건수, 캐시 적중, 종목 확장 시 비용 비례 -->

## 10. 재현 / 후속 메모

- <429 회피: 간격 두고 재실행 / 종목 축소>
- <provider 전환: `--llm-config`만 교체>
````
