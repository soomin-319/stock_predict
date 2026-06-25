"""Runtime configuration dataclasses for the Kakao/Colab bot.

Extracted from ``kakao_colab_bot`` so the large bot module can focus on request
handling. ``kakao_colab_bot`` re-imports these names, so the public surface
(``src.chatbot.PipelineRuntimeConfig`` / ``PyngrokTunnelConfig``) is preserved.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineRuntimeConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    python_executable: str = sys.executable
    input_csv: str = "result/session/session_ohlcv.csv"
    report_json: str = "pipeline_report_with_context.json"
    dart_api_key: str | None = None
    dart_corp_map_csv: str | None = "data/dart_corp_map.csv"
    fetch_investor_context: bool = True
    enable_investor_disclosure: bool = True
    openai_api_key: str | None = None
    openai_model: str | None = None
    naver_client_id: str | None = None
    naver_client_secret: str | None = None
    use_external: bool = False
    bootstrap_default_symbols: bool = False
    bootstrap_on_launch: bool = False
    async_issue_summary_on_demand: bool = True
    real_start: str = "2018-01-01"
    prewarm_default_predictions: bool = False
    runtime_dir: str = "result/runtime"
    llm_config: str | None = None
    news_impact_llm_config: str | None = None
    published_dir: str = "published/latest"
    extra_args: tuple[str, ...] = ()
    kakao_webhook_secret: str | None = None
    max_concurrent_prediction_jobs: int = 2
    refresh_cooldown_seconds: int = 60
    allowed_webhook_cidrs: tuple[str, ...] = ()

    def build_command(
        self,
        symbol: str,
        add_symbols: list[str] | None = None,
        issue_summary_symbols: list[str] | None = None,
        enable_news_impact_llm: bool = False,
    ) -> list[str]:
        normalized_add_symbols = [str(s) for s in (add_symbols or [symbol]) if str(s).strip()]
        normalized_issue_symbols = [str(s) for s in (issue_summary_symbols or [symbol]) if str(s).strip()]
        cmd = [
            self.python_executable,
            "src/pipeline.py",
            "--input",
            self.input_csv,
            "--add-symbols",
            *normalized_add_symbols,
            "--issue-summary-symbols",
            *normalized_issue_symbols,
        ]
        if self.fetch_investor_context:
            cmd.append("--fetch-investor-context")
            if not self.enable_investor_disclosure:
                cmd.append("--disable-disclosure-context")
            if self.dart_corp_map_csv:
                cmd.extend(["--dart-corp-map-csv", self.dart_corp_map_csv])
        if self.openai_model:
            cmd.extend(["--openai-model", self.openai_model])
        if enable_news_impact_llm and self.llm_config:
            cmd.extend(["--llm-config", self.llm_config])
        elif enable_news_impact_llm and self.news_impact_llm_config:
            cmd.extend(["--news-impact-llm-config", self.news_impact_llm_config])
        if not self.use_external:
            cmd.append("--disable-external")
        if self.report_json:
            cmd.extend(["--report-json", self.report_json])
        cmd.extend(self.extra_args)
        return [str(part) for part in cmd]

    def build_subprocess_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Return environment variables for subprocess so secrets stay out of argv.

        Secrets passed via CLI arguments are visible in `ps` output and shell
        history. Passing them through the subprocess environment avoids that
        exposure while still letting the child pipeline pick them up via
        `os.getenv`.
        """
        env = dict(base_env) if base_env is not None else dict(os.environ)
        secret_map = {
            "DART_API_KEY": self.dart_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
            "NAVER_CLIENT_ID": self.naver_client_id,
            "NAVER_CLIENT_SECRET": self.naver_client_secret,
        }
        for key, value in secret_map.items():
            if value:
                env[key] = value
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def secret_values(self) -> tuple[str, ...]:
        return tuple(
            str(value)
            for value in (
                self.dart_api_key,
                self.openai_api_key,
                self.naver_client_id,
                self.naver_client_secret,
            )
            if value
        )


@dataclass(slots=True)
class PyngrokTunnelConfig:
    port: int = 8000
    auth_token: str | None = None
    domain: str | None = None
    bind_tls: bool = True
    ngrok_path: str | None = None
