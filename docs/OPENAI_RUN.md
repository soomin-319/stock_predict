# 클라우드 LLM 실행: OpenAI API + Pipeline + Kakao/ngrok

이 문서는 로컬 `llama.cpp` Gemma 서버 대신 **OpenAI API**로 LLM을 호출해 전체 기능을 실행하는 PowerShell 명령어입니다. (기준일: 2026-06-27)

`LOCAL_RUN.md`의 OpenAI 버전입니다. 로컬 GPU/`llama-server`/8001 포트가 필요 없는 대신, 종목별 LLM 호출마다 OpenAI 사용 비용이 발생합니다.

- LLM provider: OpenAI (`https://api.openai.com/v1`)
- LLM model: `gpt-5-mini` (`configs/news_impact.openai.example.json` 기준, 변경 가능)
- LLM config: `configs/news_impact.openai.example.json`
- Kakao 챗봇: Flask local server + ngrok public webhook
- API 키: `.env`에서 프로세스 환경변수(`OPENAI_API_KEY`)로 로드. 값은 출력하지 않음.

> 주의: 이 프로젝트 출력은 연구/운영 참고용입니다. 매수/매도/보유 판단은 `predicted_return` 기준만 사용합니다. 뉴스/공시 LLM 출력은 명시 설정된 누수 방지 파이프라인 입력/점수화에만 사용할 수 있으며, 수동 서술만으로 모델 출력을 덮어쓰면 안 됩니다. **OpenAI 사용 시 종목별 호출 비용이 발생하므로 `--issue-summary-symbols`/`--real-symbols`로 대상 종목을 좁게 유지하세요.**

## LOCAL_RUN과의 차이

- `llama-server.exe`, GGUF 모델 파일, GPU/VRAM 오프로딩, 8001 포트가 필요 없습니다.
- LLM 호출은 모두 OpenAI 클라우드로 나가므로, 인터넷 연결과 유효한 `OPENAI_API_KEY`가 필수입니다.
- `--llm-config`만 `configs/news_impact.openai.example.json`으로 바꾸면 뉴스임팩트 판정과 종목 이슈 요약(issue summary)이 모두 OpenAI로 라우팅됩니다.
- 비용/레이트리밋이 변수이므로 대상 종목 수와 `issue-summary` 캐시 활용이 중요합니다.

## LLM provider 설정

- OpenAI: `.env` 또는 프로세스 환경변수에 `OPENAI_API_KEY` 설정 후 `--llm-config configs/news_impact.openai.example.json`
- API 키는 argv에 넣지 마세요. `PipelineRuntimeConfig.build_subprocess_env`가 `OPENAI_API_KEY`를 자식 프로세스 env로 전달합니다.
- `configs/news_impact.openai.example.json` 내용:

  ```json
  {
    "llm_provider": "openai",
    "llm_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-5-mini",
    "temperature": 0.1,
    "max_retries": 2,
    "json_schema_required": true,
    "timeout_seconds": 60
  }
  ```

- 모델을 바꾸려면 위 JSON의 `llm_model`을 수정하거나, 이 파일을 복사해 별도 config를 만들어 `--llm-config`로 지정하세요. (config에 `llm_model`이 있으면 `LLM_MODEL`/`OPENAI_MODEL` 환경변수보다 우선합니다.)
- 하위 호환 플래그도 유지됩니다: `--news-impact-llm-config`, `--openai-api-key`, `--openai-model`. 새 실행은 `--llm-config` 하나를 권장합니다.

## 1) 전체 기능 실행 명령어: ngrok 외부 연결 포함

PowerShell에서 아래 전체 블록을 실행하세요.

```powershell
cd C:\Users\카운\Desktop\stock_predict

# 1. .env 로드. 키 값 출력 금지. (.env에 OPENAI_API_KEY=sk-... 가 있어야 함)
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
  $k, $v = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim().Trim('"').Trim("'"), 'Process')
}

# 2. OPENAI_API_KEY 로드 확인. (값 자체는 출력하지 않음)
if (-not $env:OPENAI_API_KEY) { throw 'OPENAI_API_KEY가 비어 있습니다. .env를 확인하세요.' }

# 3. 의존성 설치 (openai 패키지 포함)
python -m pip install -r requirements.txt
python -m pip install -e .

# 4. OpenAI 연결/모델 접근 확인
# - /v1/models 를 조회해 키 유효성과 설정 모델(gpt-5-mini) 접근 권한을 확인합니다.
python -m src.news_impact.run llm-smoke --config configs/news_impact.openai.example.json
if ($LASTEXITCODE -ne 0) { throw 'OpenAI llm-smoke 실패. OPENAI_API_KEY 또는 모델 접근 권한을 확인하세요.' }

# 5. 5종목 예측 파이프라인 실행
# - 아래 5종목만 실데이터 갱신/분석/OpenAI 뉴스임팩트 실행
# - 투자자/공시/뉴스 컨텍스트 활성화
# - OpenAI 뉴스임팩트 + 이슈 요약 연결
$TargetSymbols = @(
  '005930.KS',  # 삼성전자
  '000660.KS',  # SK하이닉스
  '035420.KS',  # NAVER
  '035720.KS',  # 카카오
  '051910.KS'   # LG화학
)
$TargetUniversePath = 'data/universe_openai_5.csv'
@('Symbol') + $TargetSymbols | Set-Content -Encoding utf8 $TargetUniversePath

python src/pipeline.py `
  --auto-refresh-real `
  --real-symbols $TargetSymbols `
  --universe-csv $TargetUniversePath `
  --fetch-investor-context `
  --issue-summary-symbols $TargetSymbols `
  --llm-config configs/news_impact.openai.example.json `
  --report-json pipeline_report.json
if ($LASTEXITCODE -ne 0) { throw 'pipeline 실행 실패. 위 로그를 확인하세요.' }

# 6. Kakao 챗봇 + ngrok 외부 webhook 실행
# .env에 NGROK_AUTH_TOKEN, OPENAI_API_KEY 필요.
python -c "import os; from src.chatbot.kakao_colab_bot import launch_colab_kakao_bot, PipelineRuntimeConfig, PyngrokTunnelConfig; cfg=PipelineRuntimeConfig(llm_config='configs/news_impact.openai.example.json', fetch_investor_context=True, use_external=True); app=launch_colab_kakao_bot(runtime_config=cfg, tunnel_config=PyngrokTunnelConfig(port=8000, auth_token=os.getenv('NGROK_AUTH_TOKEN')), host='0.0.0.0'); print('Public URL:', app['public_url']); print('Kakao Webhook URL:', app['webhook_url']); print('Health URL:', app['health_url']); app['server_thread'].join()"
```

## 2) 로컬 챗봇만 실행하고 싶을 때: ngrok 없음

`launch_colab_kakao_bot`은 `tunnel_config` 없이 호출해도 항상 ngrok 터널을 엽니다. 외부 webhook 없이 순수 로컬(Flask)로만 띄우려면 `create_app`을 직접 써서 `app.run`으로 기동합니다.

```powershell
cd C:\Users\카운\Desktop\stock_predict

Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
  $k, $v = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim().Trim('"').Trim("'"), 'Process')
}
if (-not $env:OPENAI_API_KEY) { throw 'OPENAI_API_KEY가 비어 있습니다. .env를 확인하세요.' }

python -c "from src.chatbot.kakao_colab_bot import create_app, PipelineRuntimeConfig; cfg=PipelineRuntimeConfig(llm_config='configs/news_impact.openai.example.json', fetch_investor_context=True, use_external=True); app=create_app(runtime_config=cfg); print('Local Webhook URL: http://localhost:8000/kakao/webhook'); print('Health URL: http://localhost:8000/health'); app.run(host='0.0.0.0', port=8000)"
```

## 3) 산출물 위치

주요 산출물은 `result/` 아래에 생성됩니다.

- `result/result_detail.csv`, `result/result_simple.csv` (예측 상세/요약, `utf-8-sig`)
- `result/result_news.csv`, `result/result_disclosure.csv` (뉴스/공시 컨텍스트 — 외부 활성 시)
- `result/latest/pipeline_report.json` (최신 실행 리포트; `--report-json`으로 지정한 경로에도 별도 저장됨)
- `result/runtime/llm_cache/issue_summary/` (이슈 요약 LLM 응답 캐시 — 동일 입력 재호출 비용 절감)
- 챗봇 런타임 파일: `result/runtime/`

## 4) 종료 방법

PowerShell에서 실행 중인 챗봇은 `Ctrl + C`로 종료합니다.

OpenAI 모드에서는 로컬 LLM 서버 프로세스가 없으므로 별도로 끌 것이 없습니다. (`llama-server`를 띄우지 않았습니다.)

## 5) 문제 해결

### `OpenAI llm-smoke 실패`

확인할 것:

```powershell
# 키가 로드됐는지 (값은 출력하지 말 것 — 길이만 확인)
if ($env:OPENAI_API_KEY) { "OPENAI_API_KEY length: $($env:OPENAI_API_KEY.Length)" } else { 'OPENAI_API_KEY missing' }

# OpenAI 접근 확인 (네트워크/프록시/방화벽)
Test-NetConnection -ComputerName api.openai.com -Port 443
```

- `OPENAI_API_KEY missing`이면 `.env`에 `OPENAI_API_KEY=sk-...` 줄이 있는지, 1) 단계 로드 블록을 실행했는지 확인하세요.
- 키가 있는데도 실패하면 401(키 무효/만료) 또는 모델 접근 권한 문제일 수 있습니다. 아래 항목을 확인하세요.

### 인증 실패 (401 Unauthorized)

- 키가 만료/폐기됐거나 다른 조직의 키일 수 있습니다. OpenAI 대시보드에서 키를 재발급하고 `.env`를 갱신하세요.
- 키 앞뒤 공백, 따옴표 누락 여부를 확인하세요. (`.env` 로드 시 따옴표는 제거됩니다.)

### 모델 접근 불가 (`Configured model alias ... not found` / 404 / model_not_found)

- 계정에 `gpt-5-mini` 접근 권한이 없을 수 있습니다. `configs/news_impact.openai.example.json`의 `llm_model`을 계정에서 사용 가능한 모델로 바꾸세요.
- `llm-smoke`는 `/v1/models` 목록에 설정 모델이 있는지 검사합니다. 목록에 없으면 위처럼 모델명을 교체하세요.

### 레이트리밋 / 쿼터 초과 (429)

- 짧은 시간에 많은 종목을 호출하면 429가 날 수 있습니다. `--real-symbols`/`--issue-summary-symbols`로 종목 수를 줄이세요.
- 결제/쿼터(`insufficient_quota`) 문제면 OpenAI 대시보드에서 사용 한도/결제 수단을 확인하세요.
- 파이프라인은 `max_retries`(기본 2회)와 이슈 요약 캐시로 일시적 오류/중복 호출을 일부 흡수합니다.

### 비용이 걱정될 때

- 호출 비용은 (대상 종목 수) × (종목별 뉴스/공시 입력 길이)에 비례합니다. 우선 1~5종목으로 검증한 뒤 확장하세요.
- 동일 입력은 `result/runtime/llm_cache/issue_summary/` 캐시로 재호출 비용을 줄입니다. 캐시를 비우면 다음 실행에서 다시 과금됩니다.
- 더 저렴한 모델로 바꾸려면 `llm_model`을 조정하세요. (출력 품질과의 트레이드오프를 확인)

### 네트워크/프록시 환경

- 사내 프록시/방화벽 환경이면 `api.openai.com:443` 아웃바운드가 막혀 있을 수 있습니다. 프록시 환경변수(`HTTPS_PROXY`)나 방화벽 허용을 확인하세요.

### ngrok URL이 안 나옴

`.env`에 아래 키가 있는지 확인하세요. 값은 출력하지 마세요.

```text
NGROK_AUTH_TOKEN=...
```

### 챗봇은 뜨는데 OpenAI가 안 쓰이는 경우

반드시 `PipelineRuntimeConfig(llm_config='configs/news_impact.openai.example.json')` 방식으로 실행해야 합니다. 기본 `stock-predict-kakao` 명령만 쓰면 LLM config가 직접 주입되지 않을 수 있습니다. 또한 챗봇 프로세스 환경에 `OPENAI_API_KEY`가 로드돼 있어야 합니다.

### 로컬 Gemma로 되돌리려면

`--llm-config`만 다시 바꾸면 됩니다. 로컬 실행 절차는 `docs/LOCAL_RUN.md`를 참고하세요.

```powershell
python src/pipeline.py `
  --auto-refresh-real `
  --real-symbols $TargetSymbols `
  --universe-csv $TargetUniversePath `
  --fetch-investor-context `
  --issue-summary-symbols $TargetSymbols `
  --llm-config configs/news_impact.gemma.example.json `
  --report-json pipeline_report.json
```

키는 실제 실행 환경의 `.env`/비밀 저장소에서 로드하고, 명령줄 기록에 남기지 마세요.
