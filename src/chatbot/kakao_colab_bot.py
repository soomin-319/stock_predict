from __future__ import annotations

import argparse
import json
import re
import hashlib
import subprocess
import sys
import threading
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.data.krx_universe import find_symbol_candidates_by_name
from src.data.fetch_real_data import normalize_user_symbols


_STOCK_CODE_PATTERN = re.compile(r"\b\d{6}(?:\.(?:KS|KQ))?\b", re.IGNORECASE)
_HELP_KEYWORDS = {"도움말", "help", "사용법", "시작", "안내"}
_STATUS_KEYWORDS = {"결과", "상태", "진행상황", "조회", "확인"}
_REFRESH_KEYWORDS = {"최신화", "새로고침", "재실행", "다시예측", "다시 예측"}


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
class UserSessionState:
    user_id: str
    last_symbol: str | None = None
    last_display_code: str | None = None
    last_intent: str = "idle"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
    enable_investor_flow: bool = True
    enable_investor_disclosure: bool = True
    enable_investor_news: bool = True
    news_scoring_mode: str = "auto"
    openai_api_key: str | None = None
    openai_model: str | None = None
    use_external: bool = False
    bootstrap_default_symbols: bool = True
    real_start: str = "2018-01-01"
    prewarm_default_predictions: bool = True
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
            if not self.enable_investor_flow:
                cmd.append("--disable-investor-flow")
            if not self.enable_investor_disclosure:
                cmd.append("--disable-disclosure-context")
            if not self.enable_investor_news:
                cmd.append("--disable-news-context")
            if self.news_scoring_mode:
                cmd.extend(["--news-scoring-mode", self.news_scoring_mode])
            if self.openai_api_key:
                cmd.extend(["--openai-api-key", self.openai_api_key])
            if self.openai_model:
                cmd.extend(["--openai-model", self.openai_model])
        if not self.use_external:
            cmd.append("--disable-external")
        if self.report_json:
            cmd.extend(["--report-json", self.report_json])
        if self.figure_dir:
            cmd.extend(["--figure-dir", self.figure_dir])
        cmd.extend(self.extra_args)
        return [str(part) for part in cmd]


@dataclass(slots=True)
class PyngrokTunnelConfig:
    port: int = 8000
    auth_token: str | None = None
    domain: str | None = None
    bind_tls: bool = True


class KakaoColabPredictionBot:
    def __init__(
        self,
        runtime_config: PipelineRuntimeConfig | None = None,
        result_simple_path: str | Path | None = None,
        state_path: str | Path | None = None,
        session_path: str | Path | None = None,
        process_runner: Callable[..., Any] | None = None,
    ):
        self.runtime_config = runtime_config or PipelineRuntimeConfig()
        self.project_root = Path(self.runtime_config.project_root)
        self.result_simple_path = self.project_root / (result_simple_path or "result/result_simple.csv")
        self.state_path = self.project_root / (state_path or "result/chatbot_jobs.json")
        self.session_path = self.project_root / (session_path or "result/chatbot_sessions.json")
        self.prewarm_meta_path = self.project_root / "result" / "prewarm_cache_meta.json"
        self.log_dir = self.project_root / "result" / "chatbot_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.process_runner = process_runner or subprocess.Popen
        self._active_processes: dict[str, Any] = {}
        self._job_registry = self._load_registry(self.state_path)
        self._session_registry = self._load_registry(self.session_path)
        self._legacy_formatter_patched = False

    def handle_kakao_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        utterance = ((payload.get("userRequest") or {}).get("utterance") or "").strip()
        user_id = self._extract_user_id(payload)
        return self.handle_utterance(utterance, user_id=user_id)

    def handle_utterance(self, utterance: str, user_id: str | None = None) -> dict[str, Any]:
        text = str(utterance or "").strip()
        self._refresh_job_states()

        if not text or self._is_help_request(text):
            return self._guide_response()

        if self._is_status_request(text):
            symbol = self._symbol_from_session(user_id)
            if not symbol:
                return self._guide_response("먼저 종목코드를 입력해주세요. 예: 005930")
            return self._handle_symbol_request(symbol, user_id=user_id, force_refresh=False, from_session=True)

        if self._is_refresh_request(text):
            symbol = self._symbol_from_session(user_id)
            if not symbol:
                return self._guide_response("최신화할 종목이 없습니다. 먼저 종목코드를 입력해주세요.")
            return self._handle_symbol_request(symbol, user_id=user_id, force_refresh=True, from_session=True)

        symbol_input = self._extract_stock_code(text)
        if not symbol_input:
            return self._handle_name_lookup_request(text, user_id=user_id)

        symbol = self._normalize_symbol(symbol_input)
        if not symbol:
            return self._guide_response("입력한 종목코드를 해석하지 못했습니다. 다시 확인해주세요.")

        return self._handle_symbol_request(symbol, user_id=user_id, force_refresh=False, from_session=False)

    def _handle_symbol_request(
        self,
        symbol: str,
        user_id: str | None,
        force_refresh: bool,
        from_session: bool,
    ) -> dict[str, Any]:
        display_code = self._display_code(symbol)
        self._update_session(user_id, symbol=symbol, intent="tracking")

        job_state = self._job_registry.get(symbol)
        if job_state and job_state.get("status") == "running":
            return self._build_response(
                f"{display_code} 예측이 현재 진행 중입니다. 잠시 후 '결과' 또는 '{display_code}'를 다시 입력해주세요.",
                quick_replies=[
                    ("결과 확인", "결과"),
                    ("최신화", "최신화"),
                    ("도움말", "도움말"),
                ],
            )

        cached_row = None if force_refresh else self._find_cached_prediction(symbol)
        if cached_row is not None:
            try:
                message = self._format_prediction_message(cached_row)
            except Exception as exc:
                if self._maybe_patch_legacy_rationale_bug(exc):
                    message = self._safe_format_prediction_message(cached_row)
                else:
                    self._console_log(
                        f"{display_code} 응답 메시지 포맷 오류({type(exc).__name__}): {exc}. 원문 사유로 대체합니다."
                    )
                    raw_reason = str(cached_row.get("예측 이유", "예측 이유 정보가 없습니다."))
                    message = (
                        f"[{display_code} {str(cached_row.get('종목명', '-'))}]\n"
                        f"사유: {raw_reason}"
                    )
            return self._build_response(
                message,
                quick_replies=[
                    ("최신화", "최신화"),
                    ("결과 확인", "결과"),
                    ("다른 종목", "다른 종목 코드를 입력하세요"),
                ],
            )

        if job_state and job_state.get("status") == "failed":
            return self._start_job_response(symbol, retry=True)

        if force_refresh and from_session:
            return self._start_job_response(symbol, retry=True)

        return self._start_job_response(symbol, retry=False)

    def _start_job_response(self, symbol: str, retry: bool) -> dict[str, Any]:
        self._start_prediction_job(symbol)
        display_code = self._display_code(symbol)
        if retry:
            text = (
                f"{display_code} 최신 예측을 다시 시작합니다. 잠시 후 '결과'를 입력하면 완료 여부를 확인할 수 있어요."
            )
        else:
            text = f"{display_code} 예측을 시작합니다. 잠시 후 '결과'를 입력하면 최신 예측 결과를 안내해드릴게요."
        return self._build_response(
            text,
            quick_replies=[
                ("결과 확인", "결과"),
                ("최신화", "최신화"),
                ("도움말", "도움말"),
            ],
        )

    def _handle_name_lookup_request(self, utterance: str, user_id: str | None = None) -> dict[str, Any]:
        candidates = self._find_name_candidates(utterance)
        if not candidates:
            return self._guide_response(
                "종목코드나 종목명을 찾지 못했습니다. 예: 005930 또는 삼성전자처럼 입력해주세요."
            )

        top = candidates[0]
        exact_enough = float(top.get("score", 0.0) or 0.0) >= 0.96
        same_score_candidates = [c for c in candidates if float(c.get("score", 0.0) or 0.0) >= 0.96]
        if exact_enough and len(same_score_candidates) == 1:
            symbol = self._normalize_symbol(str(top["ticker"]))
            if symbol:
                return self._handle_symbol_request(symbol, user_id=user_id, force_refresh=False, from_session=False)

        lines = ["입력하신 이름과 비슷한 종목입니다. 아래 후보 중 하나를 골라주세요:"]
        for idx, candidate in enumerate(candidates, start=1):
            lines.append(
                f"{idx}) {candidate['name']} ({candidate['ticker']}, {candidate['market']})"
            )

        quick_replies = [
            (f"{candidate['name']}({candidate['ticker']})", str(candidate["ticker"]))
            for candidate in candidates[:5]
        ]
        quick_replies.append(("도움말", "도움말"))
        return self._build_response("\n".join(lines), quick_replies=quick_replies)

    def _guide_response(self, prefix: str | None = None) -> dict[str, Any]:
        lines = []
        if prefix:
            lines.append(prefix)
        lines.extend(
            [
                "사용 방법:",
                "1) 종목코드 입력: 005930",
                "2) 종목명 입력: 삼성전자",
                "3) 예측 진행 중이면 '결과' 입력",
                "4) 최신값으로 다시 돌리고 싶으면 '최신화' 입력",
            ]
        )
        return self._build_response(
            "\n".join(lines),
            quick_replies=[
                ("예시 005930", "005930"),
                ("결과 확인", "결과"),
                ("최신화", "최신화"),
            ],
        )

    def _extract_stock_code(self, utterance: str) -> str | None:
        text = str(utterance or "").strip()
        if not text:
            return None
        match = _STOCK_CODE_PATTERN.search(text)
        if match:
            return match.group(0)
        first_token = text.split()[0]
        if re.search(r"\d", first_token):
            return first_token
        return None

    def _find_name_candidates(self, query: str) -> list[dict[str, Any]]:
        normalized_query = self._normalize_name(query)
        if not normalized_query:
            return []

        candidates = find_symbol_candidates_by_name(query, limit=None)
        if candidates:
            return candidates

        cached_df = self._load_cached_result_simple()
        if cached_df.empty or "종목명" not in cached_df.columns or "종목코드" not in cached_df.columns:
            return []

        rows: list[dict[str, Any]] = []
        for row in cached_df.itertuples(index=False):
            name = str(getattr(row, "종목명", "") or "")
            ticker = str(getattr(row, "종목코드", "") or "")
            market = self._infer_market_from_symbol(ticker)
            name_norm = self._normalize_name(name)
            if not name_norm or not ticker:
                continue
            if normalized_query == name_norm:
                score = 1.0
            elif normalized_query in name_norm:
                score = 0.9
            else:
                score = SequenceMatcher(None, normalized_query, name_norm).ratio()
            if score < 0.45:
                continue
            rows.append({"symbol": ticker, "ticker": ticker, "name": name, "market": market, "score": float(score)})
        rows.sort(key=lambda item: (-float(item["score"]), str(item["name"]), str(item["ticker"])))
        return rows

    def _normalize_name(self, text: str) -> str:
        return "".join(str(text).strip().lower().split())

    def _normalize_symbol(self, stock_code: str) -> str | None:
        normalized = normalize_user_symbols([stock_code])
        if not normalized:
            return None
        return normalized[0]

    def _display_code(self, symbol: str) -> str:
        return str(symbol).split(".")[0]

    def _extract_user_id(self, payload: dict[str, Any]) -> str:
        user_request = payload.get("userRequest") or {}
        user = user_request.get("user") or {}
        return str(
            user.get("id")
            or user.get("userKey")
            or (user.get("properties") or {}).get("plusfriendUserKey")
            or "anonymous"
        )

    def _find_cached_prediction(self, symbol: str) -> pd.Series | None:
        simple_df = self._load_cached_result_simple()
        if simple_df.empty or "종목코드" not in simple_df.columns:
            return None

        target_code = self._display_code(symbol)
        matched = simple_df[simple_df["종목코드"].astype(str).str.zfill(6) == target_code.zfill(6)]
        if matched.empty:
            return None
        return matched.iloc[0]

    def _load_cached_result_simple(self) -> pd.DataFrame:
        if not self.result_simple_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(self.result_simple_path, dtype={"종목코드": str}, encoding="utf-8-sig")
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as exc:
            self._console_log(f"예측 캐시 CSV 로드 실패: {self.result_simple_path} ({exc})")
            return pd.DataFrame()

    def _infer_market_from_symbol(self, symbol: str) -> str:
        s = str(symbol).upper()
        if s.endswith(".KQ"):
            return "KOSDAQ"
        return "KOSPI"

    def _format_prediction_message(self, row: pd.Series) -> str:
        return self._safe_format_prediction_message(row)

    def _safe_format_prediction_message(self, row: pd.Series) -> str:
        code = str(row.get("종목코드", "-"))
        name = str(row.get("종목명", "-"))
        recommendation = str(row.get("권고", "-"))
        predicted_return = self._format_percent(row.get("내일 예상 수익률(%)"))
        up_probability = self._format_percent(row.get("상승확률(%)"))
        predicted_close = self._format_price(row.get("내일 예상 종가"))
        confidence = self._format_confidence(row.get("예측 신뢰도"))
        reason = self._format_reason_for_display(str(row.get("예측 이유", "예측 이유 정보가 없습니다.")))
        return (
            f"[{code} {name}]\n"
            f"권고: {recommendation}\n"
            f"상승확률: {up_probability}\n"
            f"내일 예측 수익률: {predicted_return}\n"
            f"내일 예측 종가: {predicted_close}\n"
            f"신뢰도: {confidence}\n"
            f"{rationale_block}\n"
            f"원문 사유: {reason}"
        )

    def _maybe_patch_legacy_rationale_bug(self, exc: Exception) -> bool:
        is_legacy_name_error = isinstance(exc, NameError) and "rationale_block" in str(exc)
        if not is_legacy_name_error:
            return False

        if not self._legacy_formatter_patched:
            self._console_log(
                "레거시 포맷터 오류 감지(NameError: rationale_block). "
                "안전 포맷터로 즉시 대체합니다. (Colab 런타임 재시작 권장)"
            )
            self._legacy_formatter_patched = True
        self._format_prediction_message = self._safe_format_prediction_message
        return True

    def _format_percent(self, value: Any) -> str:
        if isinstance(value, str) and value.strip().endswith("%"):
            return value.strip()
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        return f"{float(numeric):.3f}%"

    def _format_price(self, value: Any) -> str:
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            cleaned = re.sub(r"[^\d.\-]", "", cleaned)
        else:
            cleaned = value
        numeric = pd.to_numeric(pd.Series([cleaned]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        return f"{float(numeric):,.0f}원"

    def _format_confidence(self, value: Any) -> str:
        raw_value = value.strip() if isinstance(value, str) else value
        if isinstance(raw_value, str) and raw_value.endswith("%"):
            numeric = pd.to_numeric(pd.Series([raw_value[:-1]]), errors="coerce").iloc[0]
            display = raw_value
            if not pd.isna(numeric):
                numeric = float(numeric) / 100.0
        else:
            numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            display = None
        if pd.isna(numeric):
            return "-"
        if numeric >= 0.67:
            label = "높음"
        elif numeric >= 0.34:
            label = "보통"
        else:
            label = "낮음"
        if display is None:
            display = f"{float(numeric) * 100.0:.1f}%"
        return f"{display} ({label})"

    def _console_log(self, message: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[KAKAO BOT {timestamp}] {message}", flush=True)

    def _stream_process_output(self, symbol: str, process: Any, log_handle):
        stdout = getattr(process, "stdout", None)
        if stdout is None:
            return

        for raw_line in stdout:
            log_handle.write(raw_line)
            log_handle.flush()

    def _finalize_process(self, symbol: str, exit_code: int):
        runtime = self._active_processes.get(symbol)
        if runtime is None:
            return

        log_thread = runtime.get("log_thread")
        if log_thread is not None:
            log_thread.join(timeout=1.0)
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
        self._save_registry(self.state_path, self._job_registry)

        display_code = self._display_code(symbol)
        if status == "completed":
            cached_row = self._find_cached_prediction(symbol)
            if cached_row is not None:
                try:
                    message = self._format_prediction_message(cached_row)
                except Exception as exc:
                    if self._maybe_patch_legacy_rationale_bug(exc):
                        message = self._safe_format_prediction_message(cached_row)
                    else:
                        self._console_log(
                            f"{display_code} 예측 완료 메시지 포맷 오류({type(exc).__name__}): {exc}. 원문 사유로 대체합니다."
                        )
                        raw_reason = str(cached_row.get("예측 이유", "예측 이유 정보가 없습니다."))
                        message = f"[{display_code}] 예측 완료\n사유: {raw_reason}"
                self._console_log(f"{display_code} 예측 완료\n{message}")
            else:
                self._console_log(f"{display_code} 예측 작업 completed (exit_code=0). 결과 CSV를 확인해주세요.")
        else:
            self._console_log(f"{display_code} 예측 작업 failed (exit_code={int(exit_code)}). 로그를 확인해주세요.")

        del self._active_processes[symbol]

    def _monitor_process_completion(self, symbol: str, process: Any):
        wait = getattr(process, "wait", None)
        if wait is None:
            return

        try:
            exit_code = wait()
        except Exception:
            exit_code = -1
        self._finalize_process(symbol, int(exit_code))

    def _start_prediction_job(self, symbol: str):
        command = self.runtime_config.build_command(symbol)
        display_code = self._display_code(symbol)
        submitted_at = datetime.now(timezone.utc).isoformat()
        log_path = self.log_dir / f"{display_code}_{submitted_at.replace(':', '').replace('+00:00', 'Z')}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        self._console_log(f"{display_code} 예측 작업 시작: {' '.join(command)}")
        process = self.process_runner(
            command,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        log_thread = None
        if getattr(process, "stdout", None) is not None:
            log_thread = threading.Thread(
                target=self._stream_process_output,
                args=(symbol, process, log_handle),
                daemon=True,
            )
            log_thread.start()
        else:
            self._console_log(f"{display_code} 로그 스트림을 사용할 수 없어 파일 로그만 남깁니다: {log_path}")

        monitor_thread = None
        if getattr(process, "wait", None) is not None:
            monitor_thread = threading.Thread(
                target=self._monitor_process_completion,
                args=(symbol, process),
                daemon=True,
            )
            monitor_thread.start()

        self._active_processes[symbol] = {
            "process": process,
            "log_handle": log_handle,
            "log_thread": log_thread,
            "monitor_thread": monitor_thread,
        }
        self._job_registry[symbol] = asdict(
            PredictionJobState(
                symbol=symbol,
                display_code=display_code,
                command=command,
                log_path=str(log_path.relative_to(self.project_root)),
                submitted_at=submitted_at,
                status="running",
                pid=getattr(process, "pid", None),
            )
        )
        self._save_registry(self.state_path, self._job_registry)

    def _refresh_job_states(self):
        for symbol, runtime in list(self._active_processes.items()):
            process = runtime["process"]
            exit_code = process.poll()
            if exit_code is None:
                continue
            self._finalize_process(symbol, int(exit_code))

    def _update_session(self, user_id: str | None, symbol: str | None, intent: str):
        if not user_id:
            return
        self._session_registry[user_id] = asdict(
            UserSessionState(
                user_id=user_id,
                last_symbol=symbol,
                last_display_code=self._display_code(symbol) if symbol else None,
                last_intent=intent,
            )
        )
        self._save_registry(self.session_path, self._session_registry)

    def _symbol_from_session(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        session = self._session_registry.get(user_id, {})
        symbol = session.get("last_symbol")
        return str(symbol) if symbol else None

    def _is_help_request(self, text: str) -> bool:
        return text.strip().lower() in _HELP_KEYWORDS

    def _is_status_request(self, text: str) -> bool:
        return text.strip().lower() in _STATUS_KEYWORDS

    def _is_refresh_request(self, text: str) -> bool:
        return text.strip().lower() in _REFRESH_KEYWORDS

    def _load_registry(self, path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    def _save_registry(self, path: Path, data: dict[str, dict[str, Any]]):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_response(self, text: str, quick_replies: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        response = {
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
        if quick_replies:
            response["template"]["quickReplies"] = [
                {
                    "action": "message",
                    "label": label,
                    "messageText": message_text,
                }
                for label, message_text in quick_replies
            ]
        return response


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
        try:
            return jsonify(service.handle_kakao_payload(payload))
        except Exception as exc:
            service._console_log(f"웹훅 처리 오류({type(exc).__name__}): {exc}")
            return jsonify(
                service._build_response(
                    "예측 메시지 처리 중 오류가 발생했습니다. 같은 종목코드를 다시 입력해주세요.",
                    quick_replies=[("도움말", "도움말"), ("다시 시도", "결과")],
                )
            )

    return app


def _runtime_cache_signature(cfg: PipelineRuntimeConfig, project_root: Path) -> dict[str, Any]:
    input_path = project_root / cfg.input_csv
    universe_path = project_root / "data" / "default_universe_kospi50_kosdaq50.csv"

    def _stat_payload(path: Path) -> dict[str, int | None]:
        if not path.exists():
            return {"mtime_ns": None, "size": None}
        stat = path.stat()
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}

    return {
        "input_csv": str(cfg.input_csv),
        "input_stat": _stat_payload(input_path),
        "default_universe_stat": _stat_payload(universe_path),
        "report_json": str(cfg.report_json),
        "figure_dir": str(cfg.figure_dir),
        "fetch_investor_context": bool(cfg.fetch_investor_context),
        "use_external": bool(cfg.use_external),
        "bootstrap_default_symbols": bool(cfg.bootstrap_default_symbols),
        "real_start": str(cfg.real_start),
        "dart_api_key_enabled": bool(cfg.dart_api_key),
        "dart_corp_map_csv": str(cfg.dart_corp_map_csv or ""),
    }


def _cache_signature_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_prewarm_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_prewarm_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def start_pyngrok_tunnel(tunnel_config: PyngrokTunnelConfig | None = None) -> str:
    # pyngrok is imported here so the rest of the chatbot module can still be
    # imported without opening a tunnel until the Colab launcher is actually used.
    from pyngrok import ngrok

    config = tunnel_config or PyngrokTunnelConfig()
    if config.auth_token:
        ngrok.set_auth_token(config.auth_token)

    connect_kwargs: dict[str, Any] = {
        "addr": config.port,
        "proto": "http",
        "bind_tls": config.bind_tls,
    }
    if config.domain:
        connect_kwargs["domain"] = config.domain

    listener = ngrok.connect(**connect_kwargs)
    return str(listener.public_url).rstrip("/")


def prewarm_prediction_cache(runtime_config: PipelineRuntimeConfig | None = None, force: bool = False) -> dict[str, str]:
    cfg = runtime_config or PipelineRuntimeConfig()
    project_root = Path(cfg.project_root)
    result_simple_path = project_root / "result" / "result_simple.csv"
    meta_path = project_root / "result" / "prewarm_cache_meta.json"
    signature = _runtime_cache_signature(cfg, project_root)
    signature_hash = _cache_signature_hash(signature)

    if not force and result_simple_path.exists():
        try:
            cached = pd.read_csv(result_simple_path, dtype={"종목코드": str}, encoding="utf-8-sig")
            cached_meta = _load_prewarm_meta(meta_path)
            if not cached.empty and cached_meta.get("signature_hash") == signature_hash:
                print(f"[KAKAO BOT] 기존 예측 캐시를 재사용합니다: {result_simple_path}")
                return {"result_simple_csv": str(result_simple_path)}
        except Exception as exc:
            print(f"[KAKAO BOT] 기존 예측 캐시 확인 실패. 캐시를 다시 생성합니다: {exc}")

    print("[KAKAO BOT] 기본 심볼 예측 캐시를 미리 생성합니다...")
    from colab.stock_predict_colab import run_colab_pipeline

    outputs = run_colab_pipeline(
        input_csv=cfg.input_csv,
        universe_csv=None,
        report_json=cfg.report_json,
        figure_dir=cfg.figure_dir,
        use_external=cfg.use_external,
        use_investor_context=cfg.fetch_investor_context,
        dart_api_key=cfg.dart_api_key,
        dart_corp_map_csv=cfg.dart_corp_map_csv,
        bootstrap_default_symbols=cfg.bootstrap_default_symbols,
        real_start=cfg.real_start,
    )
    _write_prewarm_meta(meta_path, {"signature": signature, "signature_hash": signature_hash})
    print(f"[KAKAO BOT] 기본 심볼 예측 캐시 준비 완료: {outputs.get('result_simple_csv', '')}")
    return outputs


def launch_colab_kakao_bot(
    runtime_config: PipelineRuntimeConfig | None = None,
    tunnel_config: PyngrokTunnelConfig | None = None,
    host: str = "0.0.0.0",
    prewarm_cache: bool | None = None,
    force_prewarm: bool = False,
):
    cfg = runtime_config or PipelineRuntimeConfig()
    should_prewarm = cfg.prewarm_default_predictions if prewarm_cache is None else prewarm_cache
    if should_prewarm:
        prewarm_prediction_cache(cfg, force=force_prewarm)

    app = create_app(runtime_config=cfg)
    port = (tunnel_config or PyngrokTunnelConfig()).port

    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    public_url = start_pyngrok_tunnel(tunnel_config)
    return {
        "app": app,
        "server_thread": server_thread,
        "public_url": public_url,
        "webhook_url": f"{public_url}/kakao/webhook",
        "health_url": f"{public_url}/health",
    }


def main():
    parser = argparse.ArgumentParser(description="KakaoTalk chatbot webhook for the stock prediction pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input", default="data/real_ohlcv.csv")
    parser.add_argument("--report-json", default="pipeline_report_with_context.json")
    parser.add_argument("--figure-dir", default="figures_with_context")
    parser.add_argument("--dart-api-key", default=None)
    parser.add_argument("--dart-corp-map-csv", default="data/dart_corp_map.csv")
    parser.add_argument("--use-pyngrok", action="store_true")
    parser.add_argument("--ngrok-auth-token", default=None)
    parser.add_argument("--ngrok-domain", default=None)
    args = parser.parse_args()

    runtime_config = PipelineRuntimeConfig(
        input_csv=args.input,
        report_json=args.report_json,
        figure_dir=args.figure_dir,
        dart_api_key=args.dart_api_key,
        dart_corp_map_csv=args.dart_corp_map_csv,
    )
    if args.use_pyngrok:
        launched = launch_colab_kakao_bot(
            runtime_config=runtime_config,
            tunnel_config=PyngrokTunnelConfig(
                port=args.port,
                auth_token=args.ngrok_auth_token,
                domain=args.ngrok_domain,
            ),
            host=args.host,
        )
        print(f"Public URL: {launched['public_url']}")
        print(f"Webhook URL: {launched['webhook_url']}")
        launched["server_thread"].join()
        return

    app = create_app(runtime_config=runtime_config)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
