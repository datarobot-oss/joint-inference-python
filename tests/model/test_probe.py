"""Tests for the packaged JointFM inference probe CLI."""

from __future__ import annotations

import io
import json
from typing import Any, Mapping, cast
from urllib.error import HTTPError, URLError

import pytest

from jointfm_inference import cli
from jointfm_inference.model import probe
from tests.support.condition_mode import random_weight_service


def _health_payload(
    *, supported_return_modes: list[str] | None = None
) -> dict[str, Any]:
    return {
        "status": "ok",
        "schema_version": probe.DEFAULT_SCHEMA_VERSION,
        "model_version": "jointfm-inference:0.2.0+ckpt.example-1",
        "checkpoint_version": "example-1",
        "head": "studentt",
        "supported_return_modes": supported_return_modes
        or [
            "mean",
            "samples",
            "quantiles",
            "log_prob",
        ],
    }


def _predict_payload(return_mode: str) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "query_times": [
            "2026-01-05T00:00:00+00:00",
            "2026-01-06T00:00:00+00:00",
        ],
        "requested_columns": ["target"],
        "mean": None,
        "samples": None,
        "quantiles": None,
        "log_prob": None,
    }
    if return_mode == "mean":
        outputs["mean"] = [[10.0], [11.0]]
    elif return_mode == "samples":
        outputs["samples"] = [
            [[10.0], [11.0]],
            [[10.5], [11.5]],
        ]
    elif return_mode == "quantiles":
        outputs["quantiles"] = [
            {"quantile": 0.1, "values": [[9.0], [10.0]]},
            {"quantile": 0.9, "values": [[11.0], [12.0]]},
        ]
    else:
        outputs["requested_columns"] = ["known_feature", "target"]
        outputs["log_prob"] = {
            "values": [-1.25, -1.5],
            "nll_values": [1.25, 1.5],
            "total": -2.75,
            "mean": -1.375,
            "nll_total": 2.75,
            "nll_mean": 1.375,
        }

    return {
        "errors": [],
        "query_mode": "forecast",
        "return_mode": return_mode,
        "model_version": "jointfm-inference:0.2.0+ckpt.example-1",
        "outputs": outputs,
    }


def test_probe_service_exercises_advertised_return_modes() -> None:
    calls: list[
        tuple[str, str, Mapping[str, Any] | None, Mapping[str, str], int | float]
    ] = []

    def fake_request(
        url: str,
        method: str,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_seconds: int | float,
    ) -> Mapping[str, Any]:
        calls.append((url, method, payload, headers, timeout_seconds))
        if method == "GET":
            return _health_payload()
        assert payload is not None
        return _predict_payload(payload["return_mode"])

    summary = probe.probe_service(
        "http://127.0.0.1:8080/",
        headers={"Authorization": "Token test-token"},
        timeout_seconds=12.0,
        json_request=fake_request,
    )

    assert summary.service_url == "http://127.0.0.1:8080"
    assert summary.health_url == "http://127.0.0.1:8080/healthz"
    assert summary.predict_url == "http://127.0.0.1:8080/predict"
    assert summary.supported_return_modes == (
        "mean",
        "samples",
        "quantiles",
        "log_prob",
    )
    assert [mode.return_mode for mode in summary.checked_modes] == [
        "mean",
        "samples",
        "quantiles",
        "log_prob",
    ]

    assert calls[0] == (
        "http://127.0.0.1:8080/healthz",
        "GET",
        None,
        {"Authorization": "Token test-token"},
        12.0,
    )
    assert calls[1][2] == probe.build_probe_payload(
        schema_version=probe.DEFAULT_SCHEMA_VERSION,
        model_version="jointfm-inference:0.2.0+ckpt.example-1",
        return_mode="mean",
    )
    sample_payload = calls[2][2]
    assert sample_payload is not None
    assert sample_payload["n_samples"] == probe.DEFAULT_SAMPLE_COUNT
    quantile_payload = calls[3][2]
    assert quantile_payload is not None
    assert quantile_payload["quantiles"] == list(probe.DEFAULT_QUANTILES)
    log_prob_payload = calls[4][2]
    assert log_prob_payload is not None
    assert log_prob_payload["requested_columns"] == [
        "known_feature",
        "target",
    ]
    assert log_prob_payload["query_rows"] == [
        dict(row) for row in probe.DEFAULT_QUERY_ROWS
    ]


def test_probe_service_rejects_requested_modes_missing_from_health() -> None:
    def fake_request(
        url: str,
        method: str,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_seconds: int | float,
    ) -> Mapping[str, Any]:
        _ = url, method, payload, headers, timeout_seconds
        return _health_payload(supported_return_modes=["mean"])

    with pytest.raises(RuntimeError, match="does not advertise requested return modes"):
        probe.probe_service(
            "http://127.0.0.1:8080",
            requested_return_modes=("samples",),
            json_request=fake_request,
        )


def test_main_prints_cli_summary(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = probe.ServiceProbeSummary(
        service_url="http://127.0.0.1:8080",
        health_url="http://127.0.0.1:8080/healthz",
        predict_url="http://127.0.0.1:8080/predict",
        schema_version=probe.DEFAULT_SCHEMA_VERSION,
        model_version="jointfm-inference:0.2.0+ckpt.example-1",
        checkpoint_version="example-1",
        head="studentt",
        supported_return_modes=("mean", "samples"),
        checked_modes=(
            probe.ProbeModeSummary(
                return_mode="mean",
                output_key="mean",
                horizon_count=2,
            ),
        ),
    )

    monkeypatch.setattr(probe, "probe_service", lambda *args, **kwargs: summary)

    cli.main(["probe", "--service-url", "http://127.0.0.1:8080"])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert (
        "http://127.0.0.1:8080: ok; head=studentt; checkpoint=example-1; modes=mean"
        in captured.out
    )


EXPECTED_MODEL_VERSION = "jointfm-inference:0.2.0+ckpt.example-1"


def _validate(payload: Mapping[str, Any], return_mode: str = "mean") -> object:
    return probe._validate_probe_response(
        payload,
        expected_model_version=EXPECTED_MODEL_VERSION,
        return_mode=return_mode,
    )


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_normalize_service_url_rejects_empty() -> None:
    with pytest.raises(RuntimeError, match="must not be empty"):
        probe.normalize_service_url("  / ")


def test_request_json_posts_payload_and_parses_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["content_type"] = request.get_header("Content-type")
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _FakeResponse(b'{"status": "ok"}')

    monkeypatch.setattr(probe, "urlopen", fake_urlopen)
    payload = probe._request_json(
        "http://127.0.0.1:8080/predict", "POST", {"a": 1}, {}, 2.5
    )
    assert payload == {"status": "ok"}
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    assert json.loads(captured["body"].decode("utf-8")) == {"a": 1}
    assert captured["timeout"] == 2.5


def test_request_json_wraps_http_and_url_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_http_error(request: Any, timeout: float) -> _FakeResponse:
        raise HTTPError(
            request.full_url, 503, "unavailable", cast(Any, None), io.BytesIO(b"boom")
        )

    monkeypatch.setattr(probe, "urlopen", raise_http_error)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        probe._request_json("http://x/healthz", "GET", None, {}, 1.0)

    def raise_url_error(request: Any, timeout: float) -> _FakeResponse:
        raise URLError("connection refused")

    monkeypatch.setattr(probe, "urlopen", raise_url_error)
    with pytest.raises(RuntimeError, match="connection refused"):
        probe._request_json("http://x/healthz", "GET", None, {}, 1.0)


def test_request_json_rejects_non_object_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def return_array(request: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"[1]")

    monkeypatch.setattr(probe, "urlopen", return_array)
    with pytest.raises(RuntimeError, match="non-object JSON payload"):
        probe._request_json("http://x/healthz", "GET", None, {}, 1.0)


def test_require_helpers_reject_invalid_fields() -> None:
    with pytest.raises(RuntimeError, match="must be a string"):
        probe._require_string(7, field="status")
    with pytest.raises(RuntimeError, match="must not be empty"):
        probe._require_string("  ", field="status")
    with pytest.raises(RuntimeError, match="must be a sequence"):
        probe._require_sequence("abc", field="errors")


def test_health_metadata_rejects_bad_status_and_empty_modes() -> None:
    with pytest.raises(RuntimeError, match="health status must be 'ok'"):
        probe._health_metadata({**_health_payload(), "status": "down"})
    with pytest.raises(RuntimeError, match="does not advertise any supported"):
        probe._health_metadata(
            {**_health_payload(), "supported_return_modes": ["bogus"]}
        )


def test_validate_probe_response_rejects_metadata_mismatches() -> None:
    with pytest.raises(RuntimeError, match="model_version mismatch"):
        _validate({**_predict_payload("mean"), "model_version": "stale"})
    with pytest.raises(RuntimeError, match="return_mode mismatch"):
        _validate(_predict_payload("samples"), return_mode="mean")
    with pytest.raises(RuntimeError, match="query_mode mismatch"):
        _validate({**_predict_payload("mean"), "query_mode": "bogus"})
    with pytest.raises(RuntimeError, match="returned errors"):
        _validate({**_predict_payload("mean"), "errors": [{"code": "X"}]})
    with pytest.raises(RuntimeError, match="outputs must be an object"):
        _validate({**_predict_payload("mean"), "outputs": []})


def test_validate_probe_response_rejects_output_shape_mismatches() -> None:
    short_times = _predict_payload("mean")
    short_times["outputs"]["query_times"] = ["2026-01-05T00:00:00+00:00"]
    with pytest.raises(RuntimeError, match="query_times length mismatch"):
        _validate(short_times)

    wrong_columns = _predict_payload("mean")
    wrong_columns["outputs"]["requested_columns"] = ["known_feature"]
    with pytest.raises(RuntimeError, match="requested_columns mismatch"):
        _validate(wrong_columns)

    missing_mean = _predict_payload("mean")
    missing_mean["outputs"]["mean"] = None
    with pytest.raises(RuntimeError, match=r"missing outputs\.mean"):
        _validate(missing_mean)

    short_quantiles = _predict_payload("quantiles")
    short_quantiles["outputs"]["quantiles"] = short_quantiles["outputs"]["quantiles"][
        :1
    ]
    with pytest.raises(RuntimeError, match="quantile count mismatch"):
        _validate(short_quantiles, return_mode="quantiles")


def test_validate_probe_response_rejects_malformed_log_prob() -> None:
    not_object = _predict_payload("log_prob")
    not_object["outputs"]["log_prob"] = [1.0]
    with pytest.raises(RuntimeError, match="log_prob must be an object"):
        _validate(not_object, return_mode="log_prob")

    short_values = _predict_payload("log_prob")
    short_values["outputs"]["log_prob"]["values"] = [-1.0]
    with pytest.raises(RuntimeError, match="log_prob values length mismatch"):
        _validate(short_values, return_mode="log_prob")

    short_nll = _predict_payload("log_prob")
    short_nll["outputs"]["log_prob"]["nll_values"] = [1.0]
    with pytest.raises(RuntimeError, match="nll_values length mismatch"):
        _validate(short_nll, return_mode="log_prob")


def test_probe_service_requires_selected_return_modes() -> None:
    def health_only(
        url: str,
        method: str,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        return _health_payload()

    with pytest.raises(RuntimeError, match="no return modes were selected"):
        probe.probe_service(
            "http://127.0.0.1:8080",
            requested_return_modes=(),
            json_request=health_only,
        )


def test_parse_header_values_parses_and_rejects_invalid() -> None:
    headers = probe._parse_header_values(["Authorization: Token abc", "X-Extra:  v "])
    assert headers == {"Authorization": "Token abc", "X-Extra": "v"}
    with pytest.raises(RuntimeError, match="expected the form"):
        probe._parse_header_values(["no-colon-header"])
    with pytest.raises(RuntimeError, match="name and value are required"):
        probe._parse_header_values([": value"])


def test_main_reports_probe_failures_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_probe_service(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("health check failed")

    monkeypatch.setattr(probe, "probe_service", failing_probe_service)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["probe", "--service-url", "http://127.0.0.1:8080/"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "http://127.0.0.1:8080: health check failed" in captured.err


def test_main_json_output_includes_summaries_and_failures(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = probe.ServiceProbeSummary(
        service_url="http://good:8080",
        health_url="http://good:8080/healthz",
        predict_url="http://good:8080/predict",
        schema_version=probe.DEFAULT_SCHEMA_VERSION,
        model_version=EXPECTED_MODEL_VERSION,
        checkpoint_version="example-1",
        head="studentt",
        supported_return_modes=("mean",),
        checked_modes=(
            probe.ProbeModeSummary(
                return_mode="mean", output_key="mean", horizon_count=2
            ),
        ),
    )

    def probe_one(service_url: str, **kwargs: Any) -> Any:
        if service_url == "http://good:8080":
            return summary
        raise RuntimeError("bad service")

    monkeypatch.setattr(probe, "probe_service", probe_one)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "probe",
                "--service-url",
                "http://good:8080",
                "--service-url",
                "http://bad:8080",
                "--json",
            ]
        )
    payload = json.loads(capsys.readouterr().out)
    assert payload["failures"] == ["http://bad:8080: bad service"]
    assert payload["services"][0]["service_url"] == "http://good:8080"


def test_probe_default_schema_version_tracks_the_service() -> None:
    """The probe's copy of the wire version must equal the service's.

    ``probe`` is deliberately dependency-free — it imports only the standard
    library so it can run inside the container — so it restates the wire
    version instead of importing it. That duplication is the failure mode this
    guards: bumping the service constant alone leaves the probe defaulting to a
    version the service rejects, and nothing else in the suite would notice.
    """
    from jointfm_inference.model.service.constants import DEFAULT_SCHEMA_VERSION

    assert probe.DEFAULT_SCHEMA_VERSION == DEFAULT_SCHEMA_VERSION


def test_the_probe_payloads_are_served_by_a_mounted_network() -> None:
    """Every request the probe builds must pass the service's own validation.

    The probe restates the request contract instead of importing it, so a
    field the service stops accepting would otherwise surface only against a
    deployment.
    """
    service = random_weight_service()

    def in_process_request(
        url: str,
        method: str,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        del headers, timeout_seconds
        if method == "GET":
            assert url.endswith(probe.HEALTH_ROUTE)
            return service.health_payload()
        assert url.endswith(probe.PREDICT_ROUTE)
        assert payload is not None
        return service.predict(payload)

    summary = probe.probe_service(
        "http://127.0.0.1:8080", json_request=in_process_request
    )

    assert summary.head == "studentt"
    assert {mode.return_mode for mode in summary.checked_modes} == set(
        probe.SUPPORTED_RETURN_MODES
    )
    assert {mode.horizon_count for mode in summary.checked_modes} == {
        len(probe.DEFAULT_QUERY_TIMES)
    }
