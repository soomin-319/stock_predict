# Full Local Run: llama.cpp Gemma + Pipeline + Kakao/ngrok

이 문서는 현재 PC에서 감지된 경로를 기준으로, 전체 기능을 한 번에 실행하는 PowerShell 명령어입니다.

- Gemma 서버: `llama.cpp` `llama-server.exe`
- Gemma endpoint: `http://localhost:8001/v1`
- Gemma model alias: `gemma-4-26b-a4b`
- Gemma model file: `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` (IQ4_XS, ~12.7GB — 16GB VRAM 풀 오프로딩용)
- Kakao 챗봇: Flask local server + ngrok public webhook
- API 키: `.env`에서 프로세스 환경변수로 로드. 값은 출력하지 않음.

> 주의: 이 프로젝트 출력은 연구/운영 참고용입니다. 매수/매도/보유 판단은 `predicted_return` 기준만 사용합니다. 뉴스/공시는 표시용 컨텍스트이며 예측값, 랭킹, 추천, 신호를 바꾸면 안 됩니다.

## 1) 전체 기능 실행 명령어: ngrok 외부 연결 포함

PowerShell에서 아래 전체 블록을 실행하세요.

```powershell
cd C:\Users\카운\Desktop\stock_predict

# 1. .env 로드. 키 값 출력 금지.
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
  $k, $v = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim().Trim('"').Trim("'"), 'Process')
}

# 2. 현재 PC 기준 llama.cpp/Gemma 경로
$LlamaServer = 'C:\Users\카운\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe'
$GemmaModel  = 'C:\Users\카운\Desktop\stock_predict\models\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf'

# 3. 의존성 설치
python -m pip install -r requirements.txt
python -m pip install -e .

# 4. 기존 8001 Gemma 서버가 있으면 그대로 사용. 없으면 llama.cpp 서버 실행.
$portOpen = Test-NetConnection -ComputerName 127.0.0.1 -Port 8001 -InformationLevel Quiet
if (-not $portOpen) {
  # IQ4_XS(~12.7GB)를 16GB VRAM에 통째로 올림: 전 레이어 GPU 오프로딩(-ngl) + flash-attn(-fa) + gemma 템플릿(--jinja).
  # --reasoning off: 뉴스임팩트 판정은 짧은 텍스트→정해진 스키마 분류라 thinking 토큰의 효용이 거의 없음.
  #   thinking을 끄면 판정당 출력 ~1,300토큰→~215토큰으로 줄어 약 4배 빨라짐(품질 저하 미관측). 뉴스/공시 점수는 표시용이라 영향 없음.
  Start-Process -FilePath $LlamaServer -ArgumentList @(
    '-m', $GemmaModel,
    '--host', '127.0.0.1',
    '--port', '8001',
    '--alias', 'gemma-4-26b-a4b',
    '-c', '8192',
    '-ngl', '99',
    '-fa', 'on',
    '--jinja',
    '--reasoning', 'off'
  ) -WindowStyle Hidden
  Start-Sleep -Seconds 15
}

# 5. Gemma OpenAI-compatible endpoint 연결 확인
python -m src.news_impact.run llm-smoke --config configs/news_impact.gemma.example.json
if ($LASTEXITCODE -ne 0) { throw 'Gemma llm-smoke 실패. llama-server/model 경로 또는 8001 포트를 확인하세요.' }

# 6. 10종목 예측 파이프라인 실행
# - 아래 10종목만 실데이터 갱신/분석/Gemma 뉴스임팩트 실행
# - 투자자/공시/뉴스 컨텍스트 활성화
# - Gemma 뉴스임팩트 연결
$TargetSymbols = @(
  '005930.KS',  # 삼성전자
  '000660.KS',  # SK하이닉스
  '035420.KS',  # NAVER
  '035720.KS',  # 카카오
  '051910.KS',  # LG화학
  '005380.KS',  # 현대차
  '000270.KS',  # 기아
  '068270.KS',  # 셀트리온
  '373220.KS',  # LG에너지솔루션
  '105560.KS'   # KB금융
)
$TargetUniversePath = 'data/universe_gemma_10.csv'
@('Symbol') + $TargetSymbols | Set-Content -Encoding utf8 $TargetUniversePath

python src/pipeline.py `
  --auto-refresh-real `
  --real-symbols $TargetSymbols `
  --universe-csv $TargetUniversePath `
  --fetch-investor-context `
  --issue-summary-symbols $TargetSymbols `
  --news-impact-llm-config configs/news_impact.gemma.example.json `
  --report-json pipeline_report.json
if ($LASTEXITCODE -ne 0) { throw 'pipeline 실행 실패. 위 로그를 확인하세요.' }

# 7. Kakao 챗봇 + ngrok 외부 webhook 실행
# .env에 NGROK_AUTH_TOKEN 필요.
python -c "import os; from src.chatbot.kakao_colab_bot import launch_colab_kakao_bot, PipelineRuntimeConfig, PyngrokTunnelConfig; cfg=PipelineRuntimeConfig(news_impact_llm_config='configs/news_impact.gemma.example.json', fetch_investor_context=True, use_external=True); app=launch_colab_kakao_bot(runtime_config=cfg, tunnel_config=PyngrokTunnelConfig(port=8000, auth_token=os.getenv('NGROK_AUTH_TOKEN')), host='0.0.0.0'); print('Public URL:', app['public_url']); print('Kakao Webhook URL:', app['webhook_url']); print('Health URL:', app['health_url']); app['server_thread'].join()"
```

## 2) 로컬 챗봇만 실행하고 싶을 때: ngrok 없음

```powershell
cd C:\Users\카운\Desktop\stock_predict

Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
  $k, $v = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim().Trim('"').Trim("'"), 'Process')
}

python -c "from src.chatbot.kakao_colab_bot import launch_colab_kakao_bot, PipelineRuntimeConfig; cfg=PipelineRuntimeConfig(news_impact_llm_config='configs/news_impact.gemma.example.json', fetch_investor_context=True, use_external=True); app=launch_colab_kakao_bot(runtime_config=cfg, host='0.0.0.0'); print('Local Webhook URL:', app['webhook_url']); print('Health URL:', app['health_url']); app['server_thread'].join()"
```

## 3) 산출물 위치

주요 산출물은 `result/` 아래에 생성됩니다.

- `result/result_detail.csv`
- `result/result_simple.csv`
- `result/pipeline_report.json` 또는 `result/latest/...`
- 챗봇 런타임 파일: `result/runtime/`

## 4) 종료 방법

PowerShell에서 실행 중인 챗봇은 `Ctrl + C`로 종료합니다.

Gemma 서버를 끄려면:

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process
```

## 5) 문제 해결

### `Gemma llm-smoke 실패`

확인할 것:

```powershell
Test-Path $LlamaServer
Test-Path $GemmaModel
Test-NetConnection -ComputerName 127.0.0.1 -Port 8001
```

### VRAM 부족(OOM) / GPU에 다 안 올라감

`gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`(~12.7GB)는 16GB VRAM에 통째로 올라가도록 받은 모델입니다. 그래도 OOM이 나면(디스플레이 점유 VRAM 등):

```powershell
# 1) 컨텍스트를 줄임 (8192 -> 4096): 위 Start-Process 인자에서 '-c','8192' 를 '-c','4096' 으로
# 2) 그래도 부족하면 일부 레이어만 GPU에: '-ngl','99' 를 '-ngl','40' 등으로 낮춤(나머지는 CPU)
# 3) 모니터를 메인보드 내장 출력에 연결하면 GPU에서 ~1.5GB가 추가로 확보됨
```

### ngrok URL이 안 나옴

`.env`에 아래 키가 있는지 확인하세요. 값은 출력하지 마세요.

```text
NGROK_AUTH_TOKEN=...
```

### 챗봇은 뜨는데 Gemma가 안 쓰이는 경우

반드시 `PipelineRuntimeConfig(news_impact_llm_config='configs/news_impact.gemma.example.json')` 방식으로 실행해야 합니다. 기본 `stock-predict-kakao` 명령만 쓰면 Gemma config가 직접 주입되지 않을 수 있습니다.
