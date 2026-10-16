"""CLI probe for packaged JointFM inference services."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HEALTH_ROUTE = "/healthz"
PREDICT_ROUTE = "/predict"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_SCHEMA_VERSION = "v6"
DEFAULT_QUERY_MODE = "forecast"
DEFAULT_TIME_INDEX_MODE = "absolute_datetime"
DEFAULT_TIME_COLUMN = "timestamp"
DEFAULT_SAMPLE_COUNT = 4
DEFAULT_QUANTILES = (0.1, 0.9)
SUPPORTED_RETURN_MODES = ("mean", "samples", "quantiles", "log_prob")
DEFAULT_COLUMNS = (
    {"name": "known_feature", "modality": "numeric"},
    {"name": "target", "modality": "numeric", "role": "target"},
)
DEFAULT_HISTORY_ROWS = (
    {
        "timestamp": "2026-01-01T00:00:00Z",
        "known_feature": 1.0,
        "target": 10.0,
    },
    {
        "timestamp": "2026-01-02T00:00:00Z",
        "known_feature": 1.2,
        "target": 11.0,
    },
    {
        "timestamp": "2026-01-03T00:00:00Z",
        "known_feature": 1.1,
        "target": 10.5,
    },
    {
        "timestamp": "2026-01-04T00:00:00Z",
        "known_feature": 1.3,
        "target": 11.2,
    },
)
DEFAULT_QUERY_TIMES = (
    "2026-01-05T00:00:00Z",
    "2026-01-06T00:00:00Z",
)
DEFAULT_REQUESTED_COLUMNS = ("target",)
DEFAULT_LOG_PROB_REQUESTED_COLUMNS = ("known_feature", "target")
DEFAULT_QUERY_ROWS = (
    {
        "timestamp": "2026-01-05T00:00:00Z",
        "known_feature": 1.4,
        "target": 11.8,
    },
    {
        "timestamp": "2026-01-06T00:00:00Z",
        "known_feature": 1.5,
        "target": 12.1,
    },
)

JsonRequestCallable = Callable[
    [str, str, Mapping[str, Any] | None, Mapping[str, str], float],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class ProbeModeSummary:
    """Summary of one successful inference-mode probe."""

    return_mode: str
    output_key: str
    horizon_count: int


@dataclass(frozen=True)
class ServiceProbeSummary:
    """Summary of one successful service probe."""

    service_url: str
    health_url: str
    predict_url: str
    schema_version: str
    model_version: str
    checkpoint_version: str
    head: str
    supported_return_modes: tuple[str, ...]
    checked_modes: tuple[ProbeModeSummary, ...]


def normalize_service_url(service_url: str) -> str:
    """Return a normalized service base URL without a trailing slash."""

    normalized = service_url.strip().rstrip("/")
    if not normalized:
        raise RuntimeError("service URL must not be empty")
    return normalized


def build_probe_payload(
    *,
    schema_version: str,
    model_version: str,
    return_mode: str,
) -> dict[str, Any]:
    """Build one canonical forecast payload for the packaged transport."""

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "model_version": model_version,
        "query_mode": DEFAULT_QUERY_MODE,
        "return_mode": return_mode,
        "time_index_mode": DEFAULT_TIME_INDEX_MODE,
        "time_column": DEFAULT_TIME_COLUMN,
        "columns": [dict(column) for column in DEFAULT_COLUMNS],
        "history_rows": [dict(row) for row in DEFAULT_HISTORY_ROWS],
        "query_times": list(DEFAULT_QUERY_TIMES),
        "requested_columns": list(DEFAULT_REQUESTED_COLUMNS),
    }
    if return_mode == "log_prob":
        payload["requested_columns"] = list(DEFAULT_LOG_PROB_REQUESTED_COLUMNS)
        payload["query_rows"] = [dict(row) for row in DEFAULT_QUERY_ROWS]
    if return_mode in {"samples", "quantiles"}:
        payload["n_samples"] = DEFAULT_SAMPLE_COUNT
    if return_mode == "quantiles":
        payload["quantiles"] = list(DEFAULT_QUANTILES)
    return payload


def build_health_url(service_url: str) -> str:
    """Return the health URL for one service base URL."""

    return f"{normalize_service_url(service_url)}{HEALTH_ROUTE}"


def build_predict_url(service_url: str) -> str:
    """Return the predict URL for one service base URL."""

    return f"{normalize_service_url(service_url)}{PREDICT_ROUTE}"


def _request_json(
    url: str,
    method: str,
    payload: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Perform one JSON request and return a JSON object response."""

    request_headers = dict(headers)
    if payload is not None and "Content-Type" not in request_headers:
        request_headers["Content-Type"] = "application/json"
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {url} failed with HTTP {error.code}: {body[:500]}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"{method} {url} failed: {error.reason}") from error

    if not isinstance(response_payload, Mapping):
        raise RuntimeError(f"{method} {url} returned a non-object JSON payload")
    return response_payload


def _require_string(value: Any, *, field: str) -> str:
    """Require one non-empty string field."""

    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise RuntimeError(f"{field} must not be empty")
    return normalized


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    """Require one JSON-array-like field."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RuntimeError(f"{field} must be a sequence")
    return value


def _health_metadata(
    payload: Mapping[str, Any],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    """Return validated probe metadata from a health payload."""

    status = _require_string(payload.get("status"), field="status")
    if status != "ok":
        raise RuntimeError(f"health status must be 'ok', got {status!r}")

    schema_version = _require_string(
        payload.get("schema_version", DEFAULT_SCHEMA_VERSION),
        field="schema_version",
    )
    model_version = _require_string(payload.get("model_version"), field="model_version")
    checkpoint_version = _require_string(
        payload.get("checkpoint_version"),
        field="checkpoint_version",
    )
    head = _require_string(payload.get("head"), field="head")

    raw_supported_modes = _require_sequence(
        payload.get("supported_return_modes"),
        field="supported_return_modes",
    )
    supported_modes = tuple(
        mode for mode in SUPPORTED_RETURN_MODES if mode in raw_supported_modes
    )
    if not supported_modes:
        raise RuntimeError("service does not advertise any supported return modes")
    return schema_version, model_version, checkpoint_version, head, supported_modes


def _validate_probe_response(
    payload: Mapping[str, Any],
    *,
    expected_model_version: str,
    return_mode: str,
) -> ProbeModeSummary:
    """Validate one predict response for a specific return mode."""

    actual_model_version = _require_string(
        payload.get("model_version"), field="model_version"
    )
    if actual_model_version != expected_model_version:
        raise RuntimeError(
            "predict response model_version mismatch: "
            f"expected {expected_model_version!r}, got {actual_model_version!r}"
        )

    actual_return_mode = _require_string(
        payload.get("return_mode"), field="return_mode"
    )
    if actual_return_mode != return_mode:
        raise RuntimeError(
            f"predict response return_mode mismatch: expected {return_mode!r}, got {actual_return_mode!r}"
        )

    query_mode = _require_string(payload.get("query_mode"), field="query_mode")
    if query_mode != DEFAULT_QUERY_MODE:
        raise RuntimeError(
            f"predict response query_mode mismatch: expected {DEFAULT_QUERY_MODE!r}, got {query_mode!r}"
        )

    errors = _require_sequence(payload.get("errors"), field="errors")
    if errors:
        raise RuntimeError(f"predict response returned errors: {list(errors)!r}")

    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("predict response outputs must be an object")

    query_times = _require_sequence(
        outputs.get("query_times"), field="outputs.query_times"
    )
    if len(query_times) != len(DEFAULT_QUERY_TIMES):
        raise RuntimeError(
            "predict response query_times length mismatch: "
            f"expected {len(DEFAULT_QUERY_TIMES)}, got {len(query_times)}"
        )

    expected_requested_columns = (
        DEFAULT_LOG_PROB_REQUESTED_COLUMNS
        if return_mode == "log_prob"
        else DEFAULT_REQUESTED_COLUMNS
    )
    requested_columns = _require_sequence(
        outputs.get("requested_columns"),
        field="outputs.requested_columns",
    )
    if list(requested_columns) != list(expected_requested_columns):
        raise RuntimeError(
            "predict response requested_columns mismatch: "
            f"expected {list(expected_requested_columns)!r}, got {list(requested_columns)!r}"
        )

    output_key = return_mode
    if output_key == "quantiles":
        quantiles = _require_sequence(
            outputs.get(output_key), field="outputs.quantiles"
        )
        if len(quantiles) != len(DEFAULT_QUANTILES):
            raise RuntimeError(
                "predict response quantile count mismatch: "
                f"expected {len(DEFAULT_QUANTILES)}, got {len(quantiles)}"
            )
    elif output_key == "log_prob":
        score = outputs.get(output_key)
        if not isinstance(score, Mapping):
            raise RuntimeError("predict response outputs.log_prob must be an object")
        values = _require_sequence(score.get("values"), field="outputs.log_prob.values")
        nll_values = _require_sequence(
            score.get("nll_values"),
            field="outputs.log_prob.nll_values",
        )
        if len(values) != len(DEFAULT_QUERY_TIMES):
            raise RuntimeError(
                "predict response log_prob values length mismatch: "
                f"expected {len(DEFAULT_QUERY_TIMES)}, got {len(values)}"
            )
        if len(nll_values) != len(DEFAULT_QUERY_TIMES):
            raise RuntimeError(
                "predict response log_prob nll_values length mismatch: "
                f"expected {len(DEFAULT_QUERY_TIMES)}, got {len(nll_values)}"
            )
    else:
        values = outputs.get(output_key)
        if values is None:
            raise RuntimeError(f"predict response missing outputs.{output_key}")
        _require_sequence(values, field=f"outputs.{output_key}")

    return ProbeModeSummary(
        return_mode=return_mode,
        output_key=output_key,
        horizon_count=len(query_times),
    )


def probe_service(
    service_url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    requested_return_modes: Sequence[str] | None = None,
    json_request: JsonRequestCallable = _request_json,
) -> ServiceProbeSummary:
    """Probe one service for health and supported inference return modes."""

    normalized_service_url = normalize_service_url(service_url)
    request_headers = dict(headers or {})
    health_url = build_health_url(normalized_service_url)
    predict_url = build_predict_url(normalized_service_url)

    health_payload = json_request(
        health_url,
        "GET",
        None,
        request_headers,
        timeout_seconds,
    )
    (
        schema_version,
        model_version,
        checkpoint_version,
        head,
        supported_return_modes,
    ) = _health_metadata(health_payload)

    if requested_return_modes is None:
        modes_to_check = supported_return_modes
    else:
        modes_to_check = tuple(requested_return_modes)
        unsupported_modes = [
            mode for mode in modes_to_check if mode not in supported_return_modes
        ]
        if unsupported_modes:
            raise RuntimeError(
                "service does not advertise requested return modes: "
                f"{unsupported_modes!r}; advertised {list(supported_return_modes)!r}"
            )

    if not modes_to_check:
        raise RuntimeError("no return modes were selected for probing")

    checked_modes: list[ProbeModeSummary] = []
    for return_mode in modes_to_check:
        payload = build_probe_payload(
            schema_version=schema_version,
            model_version=model_version,
            return_mode=return_mode,
        )
        response_payload = json_request(
            predict_url,
            "POST",
            payload,
            request_headers,
            timeout_seconds,
        )
        checked_modes.append(
            _validate_probe_response(
                response_payload,
                expected_model_version=model_version,
                return_mode=return_mode,
            )
        )

    return ServiceProbeSummary(
        service_url=normalized_service_url,
        health_url=health_url,
        predict_url=predict_url,
        schema_version=schema_version,
        model_version=model_version,
        checkpoint_version=checkpoint_version,
        head=head,
        supported_return_modes=supported_return_modes,
        checked_modes=tuple(checked_modes),
    )


def _parse_header_values(values: Sequence[str]) -> dict[str, str]:
    """Parse repeated CLI header values of the form `Name: Value`."""

    headers: dict[str, str] = {}
    for value in values:
        name, separator, header_value = value.partition(":")
        if separator == "":
            raise RuntimeError(
                f"invalid header {value!r}; expected the form 'Name: Value'"
            )
        normalized_name = name.strip()
        normalized_value = header_value.strip()
        if not normalized_name or not normalized_value:
            raise RuntimeError(
                f"invalid header {value!r}; header name and value are required"
            )
        headers[normalized_name] = normalized_value
    return headers


def _build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Probe one or more JointFM inference service instances by checking "
            "health and exercising supported return modes."
        )
    )
    parser.add_argument(
        "--service-url",
        action="append",
        required=True,
        help="Base service URL, for example http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Optional HTTP header in the form 'Name: Value'; may be repeated",
    )
    parser.add_argument(
        "--return-mode",
        action="append",
        choices=SUPPORTED_RETURN_MODES,
        default=None,
        help="Restrict the probe to one or more advertised return modes",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the successful probe summaries as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the probe CLI."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    headers = _parse_header_values(arguments.header)

    summaries: list[ServiceProbeSummary] = []
    failures: list[str] = []
    for service_url in arguments.service_url:
        try:
            summaries.append(
                probe_service(
                    service_url,
                    headers=headers,
                    timeout_seconds=arguments.timeout_seconds,
                    requested_return_modes=arguments.return_mode,
                )
            )
        except RuntimeError as error:
            failures.append(f"{normalize_service_url(service_url)}: {error}")

    if arguments.json:
        print(
            json.dumps(
                {
                    "services": [asdict(summary) for summary in summaries],
                    "failures": failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for summary in summaries:
            checked_modes = ", ".join(
                mode.return_mode for mode in summary.checked_modes
            )
            print(
                f"{summary.service_url}: ok; head={summary.head}; "
                f"checkpoint={summary.checkpoint_version}; modes={checked_modes}"
            )
        for failure in failures:
            print(failure, file=sys.stderr)

    if failures:
        raise SystemExit(1)
