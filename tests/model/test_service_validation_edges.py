# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused validation tests for the packaged inference HTTP service."""

from __future__ import annotations

import json
import math
import threading
import time
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import jointfm_inference.model.service.server as service_server
from jointfm_inference.data.adapters import ColumnSpec
from jointfm_inference.model.checkpoint import JOINTFM_1_GENERATION
from jointfm_inference.model.service import (
    DEFAULT_SCHEMA_VERSION,
    InferenceServiceConfig,
    JointFMHTTPService,
    JointFMRequestHandler,
    ServiceError,
    create_service_from_env,
    encode_response,
    serve,
)
from jointfm_inference.model.service.column_specs import (
    COLUMN_DESCRIPTOR_FIELDS,
    _parse_column_specs,
    _resolve_requested_columns,
)
from jointfm_inference.model.service.http_handler import _handler_class, _service
from jointfm_inference.model.service.scoring import (
    _resolve_sample_count,
    _sampling_bound_tensors,
)
from jointfm_inference.model.service.serialization import _serialize_log_prob_tensor
from jointfm_inference.model.service.time_conversion import _serialize_query_times
from jointfm_inference.model.service.validation import (
    _optional_bool,
    _optional_float,
    _optional_int,
    _optional_string,
    _positive_int_from_env,
    _require_mapping,
    _require_quantiles,
    _require_sequence,
    _require_string,
    _required_env,
)
from tests.support.models import tiny_model, write_checkpoint
from jointfm_inference.model.service.config import (
    CHECKPOINT_PATH_ENV,
    CHECKPOINT_VERSION_ENV,
    MAX_CONCURRENT_REQUESTS_ENV,
    PRECISION_ENV,
)
from tests.support.service import (
    SERVICE_PRECISION,
    clear_service_environment,
    make_service_config,
    make_session,
)


def _column_specs() -> list[ColumnSpec]:
    return [
        ColumnSpec(name="known_feature"),
        ColumnSpec(name="target", role="target"),
    ]


def test_resolve_requested_columns_defaults_to_every_column_and_rejects_duplicates() -> (
    None
):
    """Omitting the projection answers the whole declared position, role-agnostically."""
    column_specs = _column_specs()

    assert _resolve_requested_columns(column_specs, None) == [
        "known_feature",
        "target",
    ]
    assert _resolve_requested_columns(column_specs, [0, "target"]) == [
        "known_feature",
        "target",
    ]

    with pytest.raises(ServiceError, match="duplicates"):
        _resolve_requested_columns(column_specs, ["target", 1])


def test_resolve_requested_columns_honors_an_arbitrary_caller_order() -> None:
    """The projection is the caller's list verbatim, not a filtered declared order."""
    column_specs = _column_specs()

    assert _resolve_requested_columns(column_specs, ["target", "known_feature"]) == [
        "target",
        "known_feature",
    ]
    assert _resolve_requested_columns(column_specs, [1, 0]) == [
        "target",
        "known_feature",
    ]

    with pytest.raises(ServiceError, match="out of bounds"):
        _resolve_requested_columns(column_specs, [2])

    with pytest.raises(ServiceError, match="unknown column"):
        _resolve_requested_columns(column_specs, ["missing"])

    with pytest.raises(ServiceError, match="string name or integer"):
        _resolve_requested_columns(column_specs, [True])

    feature_only = [ColumnSpec(name="feature")]
    assert _resolve_requested_columns(feature_only, None) == ["feature"]


def test_resolve_sample_count_falls_back_to_max_and_honors_client_override() -> None:
    """Without ``n_samples`` the resolver returns the operator-configured ``max_sample_count``.

    Every return mode shares one rule: an omitted ``n_samples`` uses the
    deployment's sized budget, a smaller one is honored exactly, and a larger
    one is refused.
    """
    config = make_service_config(max_sample_count=32)

    assert (
        _resolve_sample_count(
            request_payload={},
            config=config,
        )
        == 32
    )
    assert (
        _resolve_sample_count(
            request_payload={"n_samples": 12},
            config=config,
        )
        == 12
    )

    with pytest.raises(ServiceError, match="must be an integer"):
        _resolve_sample_count(
            request_payload={"n_samples": True},
            config=config,
        )

    with pytest.raises(ServiceError, match="must be positive"):
        _resolve_sample_count(
            request_payload={"n_samples": 0},
            config=config,
        )

    with pytest.raises(ServiceError, match="container cap"):
        _resolve_sample_count(
            request_payload={"n_samples": 64},
            config=config,
        )


def test_sampling_bound_tensors_fill_unbounded_columns_with_infinities() -> None:
    specs = [
        ColumnSpec(name="a", lower_bound=0.0),
        ColumnSpec(name="b", upper_bound=5.0),
        ColumnSpec(name="c"),
    ]

    lower, upper = _sampling_bound_tensors(specs, device=torch.device("cpu"))

    assert lower is not None and upper is not None
    assert lower.tolist() == [0.0, float("-inf"), float("-inf")]
    assert upper.tolist() == [float("inf"), 5.0, float("inf")]

    unbounded_lower, unbounded_upper = _sampling_bound_tensors(
        [ColumnSpec(name="c")],
        device=torch.device("cpu"),
    )
    assert unbounded_lower is None
    assert unbounded_upper is None


def test_serialize_log_prob_tensor_validates_shape_and_reports_nll() -> None:
    scores = torch.tensor([[math.log(0.5), math.log(0.25)]])

    payload = _serialize_log_prob_tensor(scores)

    assert payload["total"] == pytest.approx(math.log(0.125))
    assert payload["nll_total"] == pytest.approx(-math.log(0.125))
    assert payload["nll_mean"] == pytest.approx(-payload["mean"])

    with pytest.raises(RuntimeError, match="one batch with horizon scores"):
        _serialize_log_prob_tensor(torch.zeros(3))

    with pytest.raises(RuntimeError, match="one batch with horizon scores"):
        _serialize_log_prob_tensor(torch.zeros(2, 3))


def test_required_field_parsers_reject_wrong_types() -> None:
    assert _require_mapping({"a": 1}, field="request") == {"a": 1}
    with pytest.raises(ServiceError, match="must be a JSON object"):
        _require_mapping(["not", "a", "mapping"], field="request")

    assert _require_string("name", field="column") == "name"
    with pytest.raises(ServiceError, match="non-empty string"):
        _require_string("", field="column")
    with pytest.raises(ServiceError, match="non-empty string"):
        _require_string(3, field="column")

    assert _optional_string(None, "column", "fallback") == "fallback"
    assert _optional_string("given", "column") == "given"

    assert _optional_int(None, field="count") is None
    assert _optional_int(3, field="count") == 3
    with pytest.raises(ServiceError, match="must be an integer"):
        _optional_int(True, field="count")
    with pytest.raises(ServiceError, match="must be an integer"):
        _optional_int(1.5, field="count")


def test_quantile_and_serialization_helpers_cover_validation_edges() -> None:
    assert _require_quantiles([0.1, 0.9]) == [0.1, 0.9]
    assert _serialize_query_times([1, 2], time_index_mode="ordinal") == [1, 2]
    assert _serialize_query_times([1, 2], time_index_mode="continuous_float") == [
        1.0,
        2.0,
    ]
    assert _serialize_query_times(
        ["2026-01-01T00:00:00Z"],
        time_index_mode="absolute_datetime",
    ) == ["2026-01-01T00:00:00+00:00"]

    with pytest.raises(ServiceError, match="strictly between 0 and 1"):
        _require_quantiles([0.0])

    with pytest.raises(ServiceError, match="must be numeric"):
        _require_quantiles([True])

    with pytest.raises(ServiceError, match="JSON array"):
        _require_sequence("not-an-array", field="rows")

    with pytest.raises(ServiceError, match="must not be empty"):
        _require_sequence([], field="rows")


def test_service_env_and_optional_field_parsers_validate_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment and optional JSON parsers should reject ambiguous types."""
    monkeypatch.delenv("JOINTFM_INFERENCE_REQUIRED_TEST", raising=False)
    with pytest.raises(ServiceError, match="Required environment variable"):
        _required_env("JOINTFM_INFERENCE_REQUIRED_TEST")

    monkeypatch.setenv("JOINTFM_INFERENCE_REQUIRED_TEST", "value")
    assert _required_env("JOINTFM_INFERENCE_REQUIRED_TEST") == "value"

    monkeypatch.setenv("JOINTFM_INFERENCE_PORT_TEST", "not-int")
    with pytest.raises(ServiceError, match="must be an integer"):
        _positive_int_from_env("JOINTFM_INFERENCE_PORT_TEST", 8080)

    monkeypatch.setenv("JOINTFM_INFERENCE_PORT_TEST", "0")
    with pytest.raises(ServiceError, match="must be positive"):
        _positive_int_from_env("JOINTFM_INFERENCE_PORT_TEST", 8080)

    monkeypatch.delenv("JOINTFM_INFERENCE_PORT_TEST", raising=False)
    assert _positive_int_from_env("JOINTFM_INFERENCE_PORT_TEST", 8080) == 8080

    assert _optional_bool(None, field="flag", default=True) is True
    assert _optional_bool(False, field="flag", default=True) is False
    with pytest.raises(ServiceError, match="must be a boolean"):
        _optional_bool(1, field="flag", default=False)

    assert _optional_float(4, field="scale") == 4.0
    with pytest.raises(ServiceError, match="must be numeric"):
        _optional_float(True, field="scale")


def test_parse_column_specs_reads_name_role_and_bounds() -> None:
    columns = _parse_column_specs(
        [
            {"name": "driver", "modality": "numeric"},
            {
                "name": "target",
                "modality": "numeric",
                "role": "target",
                "lower_bound": 0,
                "upper_bound": 10.5,
            },
        ]
    )

    assert columns == [
        ColumnSpec(name="driver", role="feature"),
        ColumnSpec(name="target", role="target", lower_bound=0.0, upper_bound=10.5),
    ]


# One non-default value per descriptor field the service accepts only at its default.
_NON_DEFAULT_DESCRIPTOR_VALUES = {
    "nullable": True,
    "vocabulary_size": 5,
    "level_count": 4,
    "mapping": {"a": 0, "b": 1},
    "time_value_kind": "absolute_datetime",
    "time_value_scale_seconds": 60,
    "time_value_use_local_normalized_time": True,
    "time_value_calendar_id": "business-days",
    "time_value_timezone": "UTC",
    "traits": {"seasonal": "present"},
}


def test_every_default_only_descriptor_field_has_a_refusal_case() -> None:
    """A descriptor field added later must be covered by the refusal cases below."""
    served = {"name", "modality", "role", "lower_bound", "upper_bound"}

    assert set(_NON_DEFAULT_DESCRIPTOR_VALUES) == COLUMN_DESCRIPTOR_FIELDS - served


@pytest.mark.parametrize(
    ("key", "value"), sorted(_NON_DEFAULT_DESCRIPTOR_VALUES.items())
)
def test_parse_column_specs_refuses_a_non_default_descriptor_field(
    key: str, value: Any
) -> None:
    column = {"name": "target", "modality": "numeric", key: value}

    with pytest.raises(ServiceError, match="is not served") as exc_info:
        _parse_column_specs([{"name": "driver", "modality": "numeric"}, column])

    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.field == f"columns[1].{key}"


def test_parse_column_specs_accepts_descriptor_fields_at_their_defaults() -> None:
    columns = _parse_column_specs(
        [
            {
                "name": "target",
                "modality": "numeric",
                "nullable": False,
                "traits": {},
                "time_value_use_local_normalized_time": False,
                "time_value_calendar_id": "pandas-default",
                "vocabulary_size": None,
            }
        ]
    )

    assert columns == [ColumnSpec(name="target")]


@pytest.mark.parametrize(
    ("column", "field", "message"),
    [
        pytest.param(
            {"name": "c", "modality": "categorical"},
            "columns[0].modality",
            "is not served",
            id="modality",
        ),
        pytest.param(
            {"name": "c", "modality": "numeric", "role": "static"},
            "columns[0].role",
            "is not served",
            id="role",
        ),
        pytest.param(
            {"name": "c", "modality": "numeric", "unit": "kg"},
            "columns[0]",
            "unknown fields",
            id="unknown-key",
        ),
        pytest.param(
            {"modality": "numeric"}, "columns[0].name", "non-empty", id="name"
        ),
        pytest.param(
            {"name": "c"}, "columns[0].modality", "non-empty", id="no-modality"
        ),
        pytest.param(
            {"name": "c", "modality": "numeric", "lower_bound": 2, "upper_bound": 1},
            "columns[0]",
            "must not exceed",
            id="crossed-bounds",
        ),
        pytest.param("c", "columns[0]", "JSON object", id="not-a-mapping"),
    ],
)
def test_parse_column_specs_refuses_malformed_descriptors(
    column: Any, field: str, message: str
) -> None:
    with pytest.raises(ServiceError, match=message) as exc_info:
        _parse_column_specs([column])

    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.field == field


class _FakeHTTPHandler(JointFMRequestHandler):
    """Minimal request-handler double that records response writes."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.rfile = BytesIO()
        self.wfile = BytesIO()
        self.path = "/"
        self.responses: list[int] = []
        self.sent_headers: list[tuple[str, str]] = []
        self.ended = False

    def send_response(self, code: int, message: str | None = None) -> None:
        del message
        cast(list[int], self.responses).append(code)

    def send_header(self, keyword: str, value: str) -> None:
        self.sent_headers.append((keyword, value))

    def end_headers(self) -> None:
        self.ended = True


def test_request_handler_reads_json_and_reports_decode_errors() -> None:
    """The HTTP handler should validate body length and JSON decoding."""
    handler = _FakeHTTPHandler()
    raw_body = b'{"ok": true}'
    handler.headers["Content-Length"] = str(len(raw_body))
    handler.rfile = BytesIO(raw_body)

    assert handler._read_json_body() == {"ok": True}

    handler = _FakeHTTPHandler()
    with pytest.raises(ServiceError, match="Content-Length header"):
        handler._read_json_body()

    handler.headers["Content-Length"] = "bad"
    with pytest.raises(ServiceError, match="must be an integer"):
        handler._read_json_body()

    handler.headers["Content-Length"] = "1"
    handler.rfile = BytesIO(b"{")
    with pytest.raises(ServiceError, match="not valid JSON"):
        handler._read_json_body()


def _response_body(handler: _FakeHTTPHandler) -> dict[str, Any]:
    """Decode the JSON body one fake handler wrote."""
    return json.loads(cast(BytesIO, handler.wfile).getvalue())


def test_request_handler_sends_json_errors_and_dispatches_endpoints() -> None:
    """Handler methods should emit structured JSON for health, predict, and errors."""
    service = cast(
        JointFMHTTPService,
        SimpleNamespace(
            health_payload=lambda: {"status": "ok"},
            predict_response_bytes=lambda payload: encode_response(
                {"echo": payload["value"]}
            ),
        ),
    )
    handler = _FakeHTTPHandler()
    handler.service = service

    handler.path = "/healthz"
    handler.do_GET()
    assert handler.responses == [HTTPStatus.OK]
    assert _response_body(handler) == {"status": "ok"}

    handler = _FakeHTTPHandler()
    handler.service = service
    handler.path = "/predict"
    raw_body = b'{"value": 3}'
    handler.headers["Content-Length"] = str(len(raw_body))
    handler.rfile = BytesIO(raw_body)
    handler.do_POST()
    assert handler.responses == [HTTPStatus.OK]
    assert _response_body(handler) == {"echo": 3}

    handler = _FakeHTTPHandler()
    handler.path = "/missing"
    handler.do_GET()
    assert handler.responses == [HTTPStatus.NOT_FOUND]
    assert _response_body(handler)["errors"][0]["code"] == "VALIDATION_ERROR"


def test_request_handler_rejects_unknown_post_endpoint() -> None:
    """POST to anything but /predict must return a structured 404 error."""
    handler = _FakeHTTPHandler()
    handler.path = "/unknown"

    handler.do_POST()

    assert handler.responses == [HTTPStatus.NOT_FOUND]
    [error] = _response_body(handler)["errors"]
    assert error["message"] == "Unknown endpoint: /unknown"
    assert error["code"] == "VALIDATION_ERROR"


def test_request_handler_suppresses_access_logging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The handler must not emit stdlib access logs to stderr."""
    handler = _FakeHTTPHandler()

    handler.log_message("%s", "GET /healthz 200")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_request_handler_dispatches_service_errors() -> None:
    """ServiceError exceptions should preserve API code, status, and field."""

    def fail_predict(_payload):
        raise ServiceError(
            "VALIDATION_ERROR",
            "bad request",
            status=HTTPStatus.BAD_REQUEST,
            field="value",
        )

    handler = _FakeHTTPHandler()
    handler.service = cast(
        JointFMHTTPService, SimpleNamespace(predict_response_bytes=fail_predict)
    )
    handler.path = "/predict"
    raw_body = b'{"value": 3}'
    handler.headers["Content-Length"] = str(len(raw_body))
    handler.rfile = BytesIO(raw_body)

    handler.do_POST()

    assert handler.responses == [HTTPStatus.BAD_REQUEST]
    assert _response_body(handler)["errors"][0]["field"] == "value"


def test_handler_class_binds_the_service_and_an_unbound_handler_fails() -> None:
    bound = _handler_class(cast(JointFMHTTPService, SimpleNamespace()))
    handler = _FakeHTTPHandler()
    handler.service = bound.service
    assert _service(handler) is bound.service

    unbound = _FakeHTTPHandler()
    unbound.service = None
    with pytest.raises(RuntimeError, match="without a service"):
        _service(unbound)


def test_service_startup_loads_the_configured_checkpoint_and_serves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Startup refuses a missing checkpoint, serves a present one, and always closes the server."""
    config = make_service_config(
        checkpoint_path=str(tmp_path / "missing.pt"), port=8081
    )

    def _config_from_env(cls: type[InferenceServiceConfig]) -> InferenceServiceConfig:
        del cls
        return config

    monkeypatch.setattr(
        InferenceServiceConfig, "from_env", classmethod(_config_from_env)
    )
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        create_service_from_env()

    checkpoint_path = tmp_path / "jointfm.pt"
    write_checkpoint(checkpoint_path)
    config.checkpoint_path = str(checkpoint_path)
    created = create_service_from_env()

    health = created.health_payload()
    assert health["generation"] == JOINTFM_1_GENERATION
    assert health["precision"] == SERVICE_PRECISION
    assert health["checkpoint_path"] == str(checkpoint_path)

    events: list[object] = []

    class FakeServer:
        def __init__(self, address, handler_class) -> None:
            events.append(address)
            events.append(handler_class.service)

        def serve_forever(self) -> None:
            events.append("served")

        def server_close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(service_server, "ThreadingHTTPServer", FakeServer)
    serve(created)

    assert events == [("127.0.0.1", 8081), created, "served", "closed"]


def _service_with_concurrency_cap(
    max_concurrent: int, config: InferenceServiceConfig | None = None
) -> JointFMHTTPService:
    """Mount a tiny network behind a service whose forecast path the tests stub."""
    if config is None:
        config = make_service_config(max_concurrent_requests=max_concurrent)
    return JointFMHTTPService(config=config, session=make_session(tiny_model()))


def test_health_short_circuit_does_not_consume_concurrency_slot() -> None:
    """Health probes must not block on the prediction-side cap.

    With ``max_concurrent_requests=1`` and a long-running forecast in flight,
    a concurrent ``health`` request must still return immediately.
    """
    service = _service_with_concurrency_cap(max_concurrent=1)

    enter_event = threading.Event()
    release_event = threading.Event()

    def _stub_run_forecast_request(payload):
        enter_event.set()
        release_event.wait(timeout=5.0)
        return {"served": True}

    service._run_forecast_request = _stub_run_forecast_request  # ty: ignore[invalid-assignment]

    forecast_thread = threading.Thread(
        target=service.predict,
        args=(
            {
                "request_type": "predict",
                "schema_version": DEFAULT_SCHEMA_VERSION,
                "model_version": "ignored",
            },
        ),
    )
    forecast_thread.start()
    assert enter_event.wait(timeout=5.0)

    # Forecast is holding the only slot; health must still complete promptly.
    health_payload = service.predict({"request_type": "health"})
    assert health_payload["status"] == "ok"
    assert health_payload["max_concurrent_requests"] == 1

    release_event.set()
    forecast_thread.join(timeout=5.0)
    assert not forecast_thread.is_alive()


def _run_predict_concurrent_peak(
    service: JointFMHTTPService,
    *,
    callers: int,
) -> int:
    """Race ``callers`` forecast requests through ``predict`` and return the peak in-flight count.

    Each thread blocks inside a stubbed ``_run_forecast_request`` until the
    shared release event is set, so the peak observed under the bounded
    semaphore is the effective concurrency cap of the service.
    """
    in_flight = 0
    peak_in_flight = 0
    in_flight_lock = threading.Lock()
    enter_event = threading.Event()
    release_event = threading.Event()

    def _stub_run_forecast_request(payload):
        nonlocal in_flight, peak_in_flight
        with in_flight_lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        enter_event.set()
        release_event.wait(timeout=5.0)
        with in_flight_lock:
            in_flight -= 1
        return {"served": True}

    service._run_forecast_request = _stub_run_forecast_request  # ty: ignore[invalid-assignment]

    def _caller() -> None:
        service.predict(
            {
                "request_type": "predict",
                "schema_version": DEFAULT_SCHEMA_VERSION,
                "model_version": "ignored",
            }
        )

    threads = [threading.Thread(target=_caller) for _ in range(callers)]
    for thread in threads:
        thread.start()
    assert enter_event.wait(timeout=5.0)
    # Give the scheduler time to admit any remaining slots before we observe peak.
    time.sleep(0.05)
    release_event.set()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    return peak_in_flight


def test_slot_is_released_after_forecast_raises_and_after_success() -> None:
    """Both success and failure paths must release the bounded semaphore.

    A bare ``.acquire()`` without ``try/finally`` in place of
    ``with self._concurrency_semaphore:`` would leak a slot on every raised
    exception, gradually deadlocking the container. With ``cap=1``
    we run one failing call followed by one succeeding call; the second
    call can only complete if the first call's slot was released despite
    the exception.
    """
    service = _service_with_concurrency_cap(max_concurrent=1)

    invocations: list[bool] = []

    def _stub_run_forecast_request(payload):
        if not invocations:
            invocations.append(False)
            raise RuntimeError("stubbed forecast failure")
        invocations.append(True)
        return {"served": True}

    service._run_forecast_request = _stub_run_forecast_request  # ty: ignore[invalid-assignment]

    payload = {
        "request_type": "predict",
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "model_version": "ignored",
    }

    with pytest.raises(RuntimeError, match="stubbed forecast failure"):
        service.predict(payload)

    # A second call from a fresh thread must complete promptly. If the
    # semaphore had leaked, the join below would time out and we'd catch
    # the deadlock as an ``is_alive`` assertion failure.
    finished = threading.Event()

    def _second_call() -> None:
        service.predict(payload)
        finished.set()

    second = threading.Thread(target=_second_call)
    second.start()
    assert finished.wait(timeout=1.0), "second predict() blocked; slot was leaked"
    second.join(timeout=1.0)
    assert not second.is_alive()
    assert invocations == [False, True]


def test_max_concurrent_requests_env_var_caps_observed_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: ``JOINTFM_INFERENCE_MAX_CONCURRENT_REQUESTS`` -> from_env -> semaphore.

    Constructs the service exactly as the container does at startup, then
    observes the runtime cap by racing more callers than the cap and reading
    the peak in-flight count: ``predict`` must admit exactly ``cap`` forecasts
    at once, no fewer and no more. The single-slot case, the production
    default, is covered by the blocking test below.
    """
    clear_service_environment(monkeypatch)
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, "/models/jointfm.pt")
    monkeypatch.setenv(CHECKPOINT_VERSION_ENV, "example-1")
    monkeypatch.setenv(MAX_CONCURRENT_REQUESTS_ENV, "3")
    monkeypatch.setenv(PRECISION_ENV, SERVICE_PRECISION)

    config = InferenceServiceConfig.from_env()
    assert config.max_concurrent_requests == 3

    service = _service_with_concurrency_cap(3, config)

    peak = _run_predict_concurrent_peak(service, callers=5)

    assert peak == 3, f"observed peak {peak}, expected env-configured cap of 3"


def test_extra_caller_blocks_until_an_in_flight_slot_is_released() -> None:
    """A request that overflows the cap must block until a slot frees up.

    The contract is "block until the caller's timeout gives up";
    we exercise that by holding the only slot, starting one more caller,
    asserting it does not progress, then releasing the holder and
    asserting the waiter immediately proceeds.
    """
    service = _service_with_concurrency_cap(max_concurrent=1)

    holder_entered = threading.Event()
    holder_release = threading.Event()
    waiter_entered = threading.Event()
    forecast_count = [0]
    forecast_count_lock = threading.Lock()

    def _stub_run_forecast_request(payload):
        with forecast_count_lock:
            forecast_count[0] += 1
            invocation_index = forecast_count[0]
        if invocation_index == 1:
            holder_entered.set()
            holder_release.wait(timeout=5.0)
        else:
            waiter_entered.set()
        return {"served": True}

    service._run_forecast_request = _stub_run_forecast_request  # ty: ignore[invalid-assignment]

    payload = {
        "request_type": "predict",
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "model_version": "ignored",
    }

    holder = threading.Thread(target=service.predict, args=(payload,))
    waiter = threading.Thread(target=service.predict, args=(payload,))

    holder.start()
    assert holder_entered.wait(timeout=5.0)

    waiter.start()
    # The waiter must remain blocked on the semaphore until the holder releases.
    assert not waiter_entered.wait(timeout=0.2), (
        "waiter entered _run_forecast_request before the holder released its slot"
    )

    holder_release.set()
    assert waiter_entered.wait(timeout=5.0), (
        "waiter never progressed after the holder released its slot"
    )
    holder.join(timeout=5.0)
    waiter.join(timeout=5.0)
    assert not holder.is_alive()
    assert not waiter.is_alive()


def test_server_main_creates_service_and_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    sentinel = object()

    def fake_create_service_from_env() -> object:
        events.append("created")
        return sentinel

    def fake_serve(service: object) -> None:
        events.append(service)

    monkeypatch.setattr(
        service_server, "create_service_from_env", fake_create_service_from_env
    )
    monkeypatch.setattr(service_server, "serve", fake_serve)
    service_server.main()
    assert events == ["created", sentinel]


@pytest.mark.parametrize("field", ["nullable", "time_value_use_local_normalized_time"])
def test_a_default_only_boolean_field_refuses_an_integer(field: str) -> None:
    with pytest.raises(ServiceError) as raised:
        _parse_column_specs([{"name": "c", "modality": "numeric", field: 0}])
    assert raised.value.field == f"columns[0].{field}"
