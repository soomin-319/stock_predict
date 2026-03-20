from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.data.fetch_real_data import normalize_user_symbols


_STOCK_CODE_PATTERN = re.compile(r"\b\d{6}(?:\.(?:KS|KQ))?\b", re.IGNORECASE)


@dataclass(slots=True)
class PredictionJobState:
    symbol: str
    display_code: str
    command: list[str]
    log_path: str
    submitted_at: str
    status: str = "running"
    pid: int | None = None
    exit_code: int | None = None
    completed_at: str | None = None


@dataclass(slots=True)
class PipelineRuntimeConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    python_executable: str = sys.executable
    input_csv: str = "data/real_ohlcv.csv"
    report_json: str = "pipeline_report_with_context.json"
    figure_dir: str = "figures_with_context"
    dart_api_key: str | None = None
    dart_corp_map_csv: str | None = "data/dart_corp_map.csv"
    fetch_investor_context: bool = True
    disable_news_context: bool = False
    extra_args: tuple[str, ...] = ()

    def build_command(self, symbol: str) -> list[str]:
        cmd = [
            self.python_executable,
            "src/pipeline.py",
            "--input",
            self.input_csv,
            "--add-symbols",
            symbol,
        ]
        if self.fetch_investor_context:
            cmd.append("--fetch-investor-context")
        if self.disable_news_context:
            cmd.append("--disable-news-context")
        if self.dart_api_key:
            cmd.extend(["--dart-api-key", self.dart_api_key])
        if self.dart_corp_map_csv:
            cmd.extend(["--dart-corp-map-csv", self.dart_corp_map_csv])
        if self.report_json:
            cmd.extend(["--report-json", self.report_json])
        if self.figure_dir:
            cmd.extend(["--figure-dir", self.figure_dir])
        cmd.extend(self.extra_args)
        return [str(part) for part in cmd]


class KakaoColabPredictionBot:
    def __init__(
        self,
        runtime_config: PipelineRuntimeConfig | None = None,
        result_simple_path: str | Path | None = None,
        state_path: str | Path | None = None,
        process_runner: Callable[..., Any] | None = None,
    ):
        self.runtime_config = runtime_config or PipelineRuntimeConfig()
        self.project_root = Path(self.runtime_config.project_root)
        self.result_simple_path = self.project_root / (result_simple_path or "result/result_simple.csv")
        self.state_path = self.project_root / (state_path or "result/chatbot_jobs.json")
        self.log_dir = self.project_root / "result" / "chatbot_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.process_runner = process_runner or subprocess.Popen
        self._active_processes: dict[str, Any] = {}
        self._job_registry = self._load_job_registry()

    def handle_utterance(self, utterance: str) -> dict[str, Any]:
        symbol_input = self._extract_stock_code(utterance)
        if not symbol_input:
            return self._simple_text_response(
                "종목코드를 찾지 못했습니다. 예: 005930 또는 000660.KS 형태로 입력해주세요."
            )

        symbol = self._normalize_symbol(symbol_input)
        if not symbol:
            return self._simple_text_response("입력한 종목코드를 해석하지 못했습니다. 다시 확인해주세요.")

        self._refresh_job_states()

        cached_row = self._find_cached_prediction(symbol)
        if cached_row is not None:
            return self._simple_text_response(self._format_prediction_message(cached_row))

        job_state = self._job_registry.get(symbol)
        if job_state and job_state.get("status") == "running":
            display_code = job_state.get("display_code", self._display_code(symbol))
            return self._simple_text_response(
                f"{display_code} 예측이 아직 진행 중입니다. 잠시 후 같은 종목코드를 다시 입력해주세요."
            )

        if job_state and job_state.get("status") == "failed":
            self._start_prediction_job(symbol)
            return self._simple_text_response(
                f"{self._display_code(symbol)} 예측을 재시도합니다. 완료 후 같은 종목코드를 다시 입력하면 최신 예측 결과를 안내해드릴게요."
            )

        self._start_prediction_job(symbol)
        return self._simple_text_response(
            f"{self._display_code(symbol)} 예측을 시작합니다. 완료 후 같은 종목코드를 다시 입력하면 최신 예측 결과를 안내해드릴게요."
        )

    def _extract_stock_code(self, utterance: str) -> str | None:
        text = str(utterance or "").strip()
        if not text:
            return None
        match = _STOCK_CODE_PATTERN.search(text)
        if match:
            return match.group(0)
        first_token = text.split()[0]
        return first_token if first_token else None

    def _normalize_symbol(self, stock_code: str) -> str | None:
        normalized = normalize_user_symbols([stock_code])
        if not normalized:
            return None
        return normalized[0]

    def _display_code(self, symbol: str) -> str:
        return str(symbol).split(".")[0]

    def _find_cached_prediction(self, symbol: str) -> pd.Series | None:
        if not self.result_simple_path.exists():
            return None
        try:
            simple_df = pd.read_csv(self.result_simple_path)
        except Exception:
            return None

        if simple_df.empty or "종목코드" not in simple_df.columns:
            return None

        target_code = self._display_code(symbol)
        matched = simple_df[simple_df["종목코드"].astype(str).str.zfill(6) == target_code.zfill(6)]
        if matched.empty:
            return None
        return matched.iloc[0]

    def _format_prediction_message(self, row: pd.Series) -> str:
        code = str(row.get("종목코드", "-"))
        name = str(row.get("종목명", "-"))
        recommendation = str(row.get("권고", "-"))
        predicted_return = self._format_percent(row.get("내일 예상 수익률(%)"))
        predicted_close = self._format_price(row.get("내일 예상 종가"))
        confidence = self._format_confidence(row.get("예측 신뢰도"))
        reason = str(row.get("예측 이유", "예측 이유 정보가 없습니다."))
        return (
            f"[{code} {name}]\n"
            f"권고: {recommendation}\n"
            f"내일 예측 수익률: {predicted_return}\n"
            f"내일 예측 종가: {predicted_close}\n"
            f"신뢰도: {confidence}\n"
            f"사유: {reason}"
        )

    def _format_percent(self, value: Any) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        return f"{float(numeric):.3f}%"

    def _format_price(self, value: Any) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        return f"{float(numeric):,.0f}원"

    def _format_confidence(self, value: Any) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        if numeric >= 0.67:
            label = "높음"
        elif numeric >= 0.34:
            label = "보통"
        else:
            label = "낮음"
        return f"{float(numeric):.3f} ({label})"

    def _start_prediction_job(self, symbol: str):
        command = self.runtime_config.build_command(symbol)
        submitted_at = datetime.now(timezone.utc).isoformat()
        log_path = self.log_dir / f"{self._display_code(symbol)}_{submitted_at.replace(':', '').replace('+00:00', 'Z')}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = self.process_runner(
            command,
            cwd=self.project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._active_processes[symbol] = {"process": process, "log_handle": log_handle}
        self._job_registry[symbol] = asdict(
            PredictionJobState(
                symbol=symbol,
                display_code=self._display_code(symbol),
                command=command,
                log_path=str(log_path.relative_to(self.project_root)),
                submitted_at=submitted_at,
                status="running",
                pid=getattr(process, "pid", None),
            )
        )
        self._save_job_registry()

    def _refresh_job_states(self):
        for symbol, runtime in list(self._active_processes.items()):
            process = runtime["process"]
            exit_code = process.poll()
            if exit_code is None:
                continue
            runtime["log_handle"].close()
            status = "completed" if exit_code == 0 else "failed"
            job_state = self._job_registry.get(symbol, {})
            job_state.update(
                {
                    "status": status,
                    "exit_code": int(exit_code),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._job_registry[symbol] = job_state
            del self._active_processes[symbol]
        self._save_job_registry()

    def _load_job_registry(self) -> dict[str, dict[str, Any]]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    def _save_job_registry(self):
        self.state_path.write_text(json.dumps(self._job_registry, ensure_ascii=False, indent=2), encoding="utf-8")

    def _simple_text_response(self, text: str) -> dict[str, Any]:
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": text,
                        }
                    }
                ]
            },
        }


def create_app(bot: KakaoColabPredictionBot | None = None, runtime_config: PipelineRuntimeConfig | None = None):
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    service = bot or KakaoColabPredictionBot(runtime_config=runtime_config)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/kakao/webhook")
    def kakao_webhook():
        payload = request.get_json(silent=True) or {}
        utterance = ((payload.get("userRequest") or {}).get("utterance") or "").strip()
        return jsonify(service.handle_utterance(utterance))

    return app


def main():
    parser = argparse.ArgumentParser(description="KakaoTalk chatbot webhook for the stock prediction pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input", default="data/real_ohlcv.csv")
    parser.add_argument("--report-json", default="pipeline_report_with_context.json")
    parser.add_argument("--figure-dir", default="figures_with_context")
    parser.add_argument("--dart-api-key", default=None)
    parser.add_argument("--dart-corp-map-csv", default="data/dart_corp_map.csv")
    parser.add_argument("--disable-news-context", action="store_true")
    args = parser.parse_args()

    runtime_config = PipelineRuntimeConfig(
        input_csv=args.input,
        report_json=args.report_json,
        figure_dir=args.figure_dir,
        dart_api_key=args.dart_api_key,
        dart_corp_map_csv=args.dart_corp_map_csv,
        disable_news_context=args.disable_news_context,
    )
    app = create_app(runtime_config=runtime_config)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
