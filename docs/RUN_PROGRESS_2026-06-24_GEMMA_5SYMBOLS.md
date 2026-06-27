# 실행 경과 기록: Gemma 로컬 5종목 런 (2026-06-24)

`docs/LOCAL_RUN.md` 절차를 **10종목이 아니라 5종목**으로 실행한 경과 기록입니다.
사용자 요청으로 작업을 **중단(중간 정지)**한 시점의 스냅샷입니다.

- 실행 시작: 2026-06-24 20:19:38 (KST)
- 중단 시각: 2026-06-24 21:12 경 (KST)
- 중단 사유: 사용자 요청("현재 작업 모두 중단하고 진행경과 기록")

> 참고: 본 산출물은 연구/운영 참고용입니다. 매수/매도/보유 판단은 `predicted_return` 기준만 사용하며, 뉴스/공시는 표시용 컨텍스트로 예측값·랭킹·추천·신호를 바꾸지 않습니다.

## 1) 대상 종목 (5개)

원본 문서 10종목 중 상위 5종목만 실행. `data/universe_gemma_5.csv` 생성.

| # | 종목코드 | 종목명 |
|---|----------|--------|
| 1 | `005930.KS` | 삼성전자 |
| 2 | `000660.KS` | SK하이닉스 |
| 3 | `035420.KS` | NAVER |
| 4 | `035720.KS` | 카카오 |
| 5 | `051910.KS` | LG화학 |

## 2) 단계별 진행 상태

| 단계 | 내용 | 상태 |
|------|------|------|
| 1 | `.env` 로드 (키 값 미출력) | ✅ 완료 |
| 2 | llama.cpp / Gemma 경로 확인 | ✅ 완료 |
| 3 | 의존성 설치 (`src`, yfinance/flask/pyngrok 기설치) | ✅ 생략 가능 확인 |
| 4 | Gemma `llama-server` 기동 (포트 8001, 모델 로딩) | ✅ 완료 |
| 5 | `llm-smoke` 연결 확인 | ✅ `status: ok` |
| 6 | 예측 파이프라인 12단계 | ✅ 12/12 완료 |
| 6b | Gemma 뉴스임팩트 판정 | ⏸ **77/304 (25%)에서 중단** |
| 6c | 최종 결과(`result/`) 기록 | ❌ 미기록 (6b 도중 중단) |
| 7 | Kakao 챗봇 + ngrok webhook | ⏭ 미실행 |

파이프라인 12단계(데이터 갱신 → 정제/유니버스 → 투자자 컨텍스트 → 피처 → 워크포워드 검증 → 베이스라인 → 시그널 가중 튜닝 → 백테스트 → 최종 모델/예측)는 모두 통과했고, 이후 Gemma 뉴스임팩트 판정 단계에서 중단했습니다. 최종 예측 결과 파일은 이 단계 종료 후에 기록되므로 **이번 5종목 런의 `result/` 최종 산출물은 생성되지 않았습니다.**

## 3) Gemma 뉴스임팩트 판정 진행률

- 수집 뉴스: **373건** (공시 0건)
- dedup 후 LLM 판정 대상: **304건** (1건당 Gemma 판정 1회)
- 완료(캐시 기준): **77/304 (25%)**
- 처리율: **~40초/건** (응답 1,500~1,800토큰 @ 약 38 토큰/초)
- 캐시 구간: 20:21:05 ~ 21:11:43 (50.6분)
- 잔여 추정: 227건 × 40초 ≈ **151분(약 2.5시간)**, 무중단 시 예상 완료 ~23:43

### 종목별 뉴스 분포 (dedup 전 373건 기준)

| 종목 | 코드 | 뉴스 수 |
|------|------|--------|
| 카카오 | 035720 | 131 |
| 삼성전자 | 005930 | 94 |
| SK하이닉스 | 000660 | 91 |
| NAVER | 035420 | 57 |
| LG화학 | 051910 | 0 |

> 관찰: LG화학은 수집 뉴스 0건. 5종목 전부 처리해도 뉴스임팩트는 사실상 4종목에 대해 발생.

## 4) 환경 / 트러블슈팅 메모

### llama-server 한글 경로 인자 문제 (중요)

- 증상: 모델 절대경로(`C:\Users\카운\...`)를 `-m` 인자로 넘기면 `llama-server.exe`가 즉시 종료.
  로그상 한글 사용자 폴더 `카운`이 ANSI mojibake 형태로 깨져 `gguf_init_from_file: ... No such file or directory` 발생.
- 원인: `llama-server.exe`가 명령행 인자를 ANSI(narrow)로 읽어, 시스템 코드페이지에 없는 한글 경로가 손상됨.
  실행 파일 경로(CreateProcess, 유니코드)는 정상이나 **인자로 넘기는 경로**가 문제.
- 해결: **작업 디렉터리를 모델 폴더로 두고 모델을 상대경로(순수 ASCII 파일명)로 전달**.
  ```powershell
  Start-Process -FilePath $LlamaServer `
    -WorkingDirectory 'C:\Users\카운\Desktop\stock_predict\models' `
    -ArgumentList @('-m','gemma-4-26B-A4B-it-UD-IQ4_XS.gguf','--host','127.0.0.1','--port','8001',
                    '--alias','gemma-4-26b-a4b','-c','8192','-ngl','99','-fa','on','--jinja')
  ```
  - 8.3 단축경로는 폴더명 `카운`이 2글자라 별칭이 생성되지 않아 해결 불가 → 상대경로 방식 사용.
  - Git Bash로 한글 경로 전달 시에도 동일하게 깨지므로 PowerShell + 상대경로 권장.
- 모델 로딩 자체는 정상: Vulkan / RTX 5070 Ti(15GB free)에 풀 오프로딩, `server is listening on http://127.0.0.1:8001`.

### ngrok (step 7)

- `.env`에 `NGROK_AUTH_TOKEN` **없음**.
- 단, 로컬 ngrok 설정 `C:\Users\카운\AppData\Local\ngrok\ngrok.yml`에 authtoken 저장되어 있음 → pyngrok이 저장 토큰을 사용하므로 step 7 터널은 동작 가능성 높음(미검증).
- 참고: `launch_colab_kakao_bot()`은 `tunnel_config`가 없어도 항상 `start_pyngrok_tunnel()`을 호출하므로, 문서 섹션 2("로컬 전용")도 ngrok을 시도함.

## 5) 중단 처리 결과

| 프로세스 | PID | 상태 |
|----------|-----|------|
| 파이프라인 python | 14732 | ✅ 종료 |
| Gemma llama-server | 8176 | ✅ 종료 |
| 포트 8001 | - | 닫힘 확인 |

- 부분 진행 캐시(완료 77건)는 임시 폴더에 존재:
  `C:\Users\카운\AppData\Local\Temp\news_impact_gemma_jupecix5\llm_cache`
  (TemporaryDirectory 기반이라 재실행 시 재사용되지 않고 새로 생성됨)

## 6) 재개 방법

1. Gemma 서버 재기동 — **반드시 위 4)의 상대경로 방식** 사용 (한글 경로 인자 금지).
2. 연결 확인: `python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json`
3. 5종목 파이프라인 재실행:
   ```powershell
   python src/pipeline.py --auto-refresh-real `
     --real-symbols 005930.KS 000660.KS 035420.KS 035720.KS 051910.KS `
     --universe-csv data/universe_gemma_5.csv `
     --fetch-investor-context `
     --issue-summary-symbols 005930.KS 000660.KS 035420.KS 035720.KS 051910.KS `
     --news-impact-llm-config configs/news_impact.gemma.example.json `
     --report-json pipeline_report.json
   ```
   - 뉴스임팩트는 매 실행마다 새 임시 fixture를 만들어 304건을 처음부터 판정(약 3.4시간 소요). 도중 중단된 캐시는 이어붙지 않음.
   - 빠른 완주가 필요하면 뉴스 수가 많은 점을 고려해 시간 여유를 두고 무중단 실행 권장.
4. 외부 webhook까지 필요하면 `docs/LOCAL_RUN.md` 섹션 1의 step 7 실행.

## 7) thinking 벤치마크 & 설정 변경 (reasoning off 채택)

판정당 출력 토큰이 ~1,500개로 과다했던 원인은 **모델 크기가 아니라 thinking(추론) 모드**였음.
저장되는 최종 JSON은 ~150토큰인데 그 앞에 ~1,400토큰의 추론을 생성/폐기하고 있었음.
동일 뉴스 6건으로 thinking 3개 설정을 실측 비교:

| 설정 | llama-server 플래그 | 평균 지연 | 출력 토큰 | 304건 추정 | 속도 |
|------|--------------------|----------|----------|-----------|------|
| A thinking-on | (기본) | 37.6초 | 1,311 | ~190분 | 1.0× |
| B budget-256 | `--reasoning-budget 256` | 15.8초 | 487 | ~80분 | 2.4× |
| **C off (채택)** | `--reasoning off` | **9.1초** | **215** | **~46분** | **4.1×** |

품질: direction·event_type은 A=B=C **6/6 완전 동일**, impact_score는 노이즈 수준 지터.
`reason`/`why_may_be_wrong` 텍스트도 동급이며 일부는 C(off)가 더 종목-특정적으로 정확했음.
뉴스/공시 점수는 표시용 컨텍스트라 `predicted_return`·랭킹·신호에 영향이 없으므로 점수 정밀도에 추론 토큰을 쓸 이유가 없음.

**결정: `--reasoning off` 채택.** `docs/LOCAL_RUN.md`의 서버 기동 인자에 반영함
(코드/`configs/news_impact.gemma.example.json` 변경 없음 — 서버 플래그만 추가).
