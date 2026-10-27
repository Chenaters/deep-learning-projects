"""Dependency-light HTTP scoring service for container and Kubernetes practice."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib

from .config import DEFAULT_ARTIFACT_DIR
from .modeling import reconstruction_error
from .schema import SchemaError, validate_record

MAX_REQUEST_BYTES = 64 * 1024


class AnomalyScorer:
    def __init__(self, model_path: Path):
        self.bundle: dict[str, Any] = joblib.load(model_path)

    def score(self, record: dict[str, object]) -> dict[str, object]:
        frame = validate_record(record)
        score = float(
            reconstruction_error(
                self.bundle["scaler"], self.bundle["model"], frame
            )[0]
        )
        threshold = float(self.bundle["threshold"])
        return {
            "model_version": self.bundle["model_version"],
            "anomaly_score": round(score, 6),
            "threshold": round(threshold, 6),
            "predicted_anomaly": score >= threshold,
            "review_required": score >= threshold,
        }


def make_handler(scorer: AnomalyScorer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CroweAnomalyLab/0.1"

        def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path == "/health":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "model_version": scorer.bundle["model_version"],
                    },
                )
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path != "/score":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                    raise SchemaError("Request body size is invalid.")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise SchemaError("Request body must be one JSON object.")
                self._write_json(HTTPStatus.OK, scorer.score(payload))
            except (json.JSONDecodeError, SchemaError, TypeError, ValueError) as exc:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_request", "message": str(exc)},
                )

        def log_message(self, format_string: str, *args: object) -> None:
            # Record method/status only; never log sensor payloads.
            super().log_message(format_string, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_ARTIFACT_DIR / "model.joblib"
    )
    args = parser.parse_args()

    scorer = AnomalyScorer(args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(scorer))
    print(f"Serving model {scorer.bundle['model_version']} on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

