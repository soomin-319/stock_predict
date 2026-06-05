from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.news_impact.schema import DisclosureItem, NewsItem
from src.utils.atomic_files import atomic_write_text


class DataCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write_news_search(
        self,
        query: str,
        search_date: date,
        source: str,
        items: list[NewsItem] | tuple[NewsItem, ...],
    ) -> Path:
        path = self._news_path(query=query, search_date=search_date, source=source)
        payload = {
            "cache_type": "news_search",
            "source": source,
            "query": query,
            "search_date": search_date.isoformat(),
            "cached_at": _now_iso(),
            "items": [item.to_dict() for item in items],
        }
        return self._write_json(path, payload)

    def read_news_search(self, query: str, search_date: date, source: str) -> list[dict[str, Any]]:
        payload = self._read_json(self._news_path(query=query, search_date=search_date, source=source))
        return list(payload["items"])

    def write_disclosure(self, item: DisclosureItem) -> Path:
        return self._write_json(self._disclosure_path(item.receipt_no), item.to_dict())

    def read_disclosure(self, receipt_no: str) -> dict[str, Any]:
        return self._read_json(self._disclosure_path(receipt_no))

    def write_market_price(
        self,
        vendor: str,
        ticker: str,
        trading_day: date,
        payload: dict[str, Any],
    ) -> Path:
        cache_payload = {
            "cache_type": "market_price",
            "vendor": vendor,
            "ticker": ticker,
            "trading_day": trading_day.isoformat(),
            "cached_at": _now_iso(),
            **payload,
        }
        return self._write_json(self._market_path(vendor, ticker, trading_day), cache_payload)

    def read_market_price(self, vendor: str, ticker: str, trading_day: date) -> dict[str, Any]:
        return self._read_json(self._market_path(vendor, ticker, trading_day))

    def write_snapshot_manifest(self, snapshot_id: str) -> Path:
        files = []
        for path in sorted(self.root.rglob("*.json")):
            if path.name == f"{snapshot_id}.manifest.json":
                continue
            files.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": _sha256(path),
                }
            )
        manifest_path = self.root / "snapshots" / f"{snapshot_id}.manifest.json"
        return self._write_json(
            manifest_path,
            {
                "snapshot_id": snapshot_id,
                "created_at": _now_iso(),
                "files": files,
            },
        )

    def _news_path(self, query: str, search_date: date, source: str) -> Path:
        key = _stable_key(query)
        return self.root / "news" / source / search_date.isoformat() / f"{key}.json"

    def _disclosure_path(self, receipt_no: str) -> Path:
        return self.root / "dart" / f"{receipt_no}.json"

    def _market_path(self, vendor: str, ticker: str, trading_day: date) -> Path:
        return self.root / "market" / vendor / ticker / f"{trading_day.isoformat()}.json"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))


def _stable_key(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return digest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
