"""The packaged HTTP service over a mounted JointFM 1 network.

Most tests mount ``_FixedParameterJointFM``, a JointFM 1 network whose forward
emits one known Student-t distribution, so a response can be checked against
a closed form; one test serves a real checkpoint file end to end.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from scipy import stats

from jointfm_inference.model.checkpoint import (
    CHECKPOINT_CAPACITY_ENVELOPE_KEY,
    CHECKPOINT_INFERENCE_METADATA_KEY,
    CHECKPOINT_TIME_INDEX_SUPPORT_KEY,
)
from jointfm_inference.model.inference import JointFMInferenceSession
from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.model import JointFM
from jointfm_inference.model.service import (
    DEFAULT_SCHEMA_VERSION,
    JointFMHTTPService,
    ServiceError,
)
from tests.support.models import (
    TINY_CAPACITY_ENVELOPE,
    TINY_CONFIG,
    checkpoint_mapping,
    tiny_config,
    write_checkpoint,
)
from tests.support.service import (
    MODEL_VERSION,
    SERVICE_PRECISION,
    make_service_config,
    make_session,
    mount_service,
)

# Large enough for Monte Carlo quantiles to sit close to their closed form.
_MAX_SAMPLE_COUNT = 4096
_STUDENT_T_DF = 5.0


class _FixedParameterJointFM(JointFM):
    """A JointFM 1 network whose forward emits one known Student-t distribution.

    Column ``d`` at horizon index ``h`` is centered on ``column_locations[d] + h``
    with an isotropic ``scale``, so a response encodes the horizon index the
    service asked for. A zero scale makes every draw equal its location.
    """

    def __init__(
        self,
        cfg: JointFMConfig = TINY_CONFIG,
        *,
        column_locations: Sequence[float] = (100.0, 200.0),
        scale: float = 0.0,
    ) -> None:
        super().__init__(cfg)
        self._column_locations = torch.tensor(column_locations)
        self._scale = scale
        self.last_past_time_indices: torch.Tensor | None = None
        self.last_horizons: torch.Tensor | None = None

    def forward(
        self,
        past: torch.Tensor,
        horizons: torch.Tensor,
        past_time_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.last_past_time_indices = past_time_indices
        self.last_horizons = horizons
        horizon_count, column_count = horizons.shape[0], past.shape[-1]
        locations = self._column_locations[:column_count] + horizons.float().unsqueeze(
            -1
        )
        return {
            "logits": torch.zeros(1, horizon_count, 1),
            "means": locations.reshape(1, horizon_count, 1, column_count),
            "df": torch.full((1, horizon_count, 1), _STUDENT_T_DF),
            "chol": (self._scale * torch.eye(column_count))
            .expand(1, horizon_count, 1, column_count, column_count)
            .clone(),
        }


def _service_and_model(
    model: _FixedParameterJointFM | None = None,
    *,
    checkpoint: dict[str, Any] | None = None,
) -> tuple[JointFMHTTPService, _FixedParameterJointFM]:
    model = _FixedParameterJointFM() if model is None else model
    service = mount_service(
        model, checkpoint=checkpoint, max_sample_count=_MAX_SAMPLE_COUNT
    )
    return service, model


def _service(model: _FixedParameterJointFM | None = None) -> JointFMHTTPService:
    return _service_and_model(model)[0]


def _inference_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return checkpoint[CHECKPOINT_INFERENCE_METADATA_KEY]


def _request_payload() -> dict[str, Any]:
    return {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "query_mode": "forecast",
        "return_mode": "mean",
        "time_index_mode": "absolute_datetime",
        "time_column": "timestamp",
        "columns": [
            {"name": "known_feature", "modality": "numeric"},
            {"name": "target", "modality": "numeric", "role": "target"},
        ],
        "history_rows": [
            {
                "timestamp": f"2026-01-0{day}T00:00:00Z",
                "known_feature": 1.0 + day / 10,
                "target": 10.0 + day,
            }
            for day in range(1, 5)
        ],
        "query_times": ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"],
        "requested_columns": ["target"],
        "seed": 7,
    }


def _ordinal_payload(
    history_steps: Sequence[int], query_times: Sequence[Any]
) -> dict[str, Any]:
    payload = _request_payload()
    payload["time_index_mode"] = "ordinal"
    payload["time_column"] = "step"
    payload["history_rows"] = [
        {"step": step, "known_feature": 1.0 + step / 10, "target": 10.0 + step}
        for step in history_steps
    ]
    payload["query_times"] = list(query_times)
    return payload


# History times on grid positions 0, 1, 3, 4 and query times at horizon
# indices 0 and 2, in each served time-index mode.
_GAPPED_TIMES = {
    "ordinal": ("step", [1, 2, 4, 5], [6, 8], [6, 8]),
    "continuous_float": ("time", [0.0, 0.5, 1.5, 2.0], [2.5, 3.5], [2.5, 3.5]),
    "absolute_datetime": (
        "timestamp",
        [f"2026-01-0{day}T00:00:00Z" for day in (1, 2, 4, 5)],
        ["2026-01-06T00:00:00Z", "2026-01-08T00:00:00Z"],
        ["2026-01-06T00:00:00+00:00", "2026-01-08T00:00:00+00:00"],
    ),
}
_TIME_INDEX_MODES = pytest.mark.parametrize("time_index_mode", sorted(_GAPPED_TIMES))


def _gapped_payload(time_index_mode: str) -> dict[str, Any]:
    time_column, history_times, query_times, _ = _GAPPED_TIMES[time_index_mode]
    payload = _request_payload()
    payload["time_index_mode"] = time_index_mode
    payload["time_column"] = time_column
    payload["history_rows"] = [
        {time_column: time, "known_feature": 1.0 + index / 10, "target": 10.0 + index}
        for index, time in enumerate(history_times)
    ]
    payload["query_times"] = query_times
    return payload


def _refusal(service: JointFMHTTPService, payload: dict[str, Any]) -> ServiceError:
    with pytest.raises(ServiceError) as exc_info:
        service.predict(payload)
    return exc_info.value


def test_service_returns_requested_mean_forecast_only() -> None:
    response = _service().predict(_request_payload())

    assert response["head"] == "studentt"
    assert response["query_mode"] == "forecast"
    assert response["return_mode"] == "mean"
    assert response["outputs"]["requested_columns"] == ["target"]
    assert response["outputs"]["query_times"] == [
        "2026-01-05T00:00:00+00:00",
        "2026-01-06T00:00:00+00:00",
    ]
    assert response["outputs"]["mean"] == [[200.0], [201.0]]
    assert response["outputs"]["samples"] is None
    assert response["outputs"]["quantiles"] is None
    assert response["outputs"]["log_prob"] is None
    assert response["plausibility"] is None
    assert response["diagnostics"] == {"history_rows": 4, "horizon_count": 2, "seed": 7}


def test_service_serves_a_checkpoint_file_end_to_end(tmp_path: Path) -> None:
    """A real network loaded from disk answers every return mode with finite values."""
    checkpoint_path = tmp_path / "jointfm.pt"
    write_checkpoint(checkpoint_path)
    session = JointFMInferenceSession.from_checkpoint(
        checkpoint_path, precision=SERVICE_PRECISION
    )
    service = JointFMHTTPService(
        config=make_service_config(max_sample_count=_MAX_SAMPLE_COUNT),
        session=session,
    )
    payload = _request_payload()
    payload["n_samples"] = 8

    samples_payload = {**payload, "return_mode": "samples"}
    samples = np.asarray(service.predict(samples_payload)["outputs"]["samples"])
    quantiles_payload = {**payload, "return_mode": "quantiles", "quantiles": [0.5]}
    [median] = service.predict(quantiles_payload)["outputs"]["quantiles"]
    log_prob_payload = {
        **payload,
        "return_mode": "log_prob",
        "requested_columns": ["known_feature", "target"],
        "query_rows": [
            {"timestamp": time, "known_feature": 1.5, "target": 15.0}
            for time in payload["query_times"]
        ],
    }
    score = service.predict(log_prob_payload)["outputs"]["log_prob"]

    assert samples.shape == (8, 2, 1)
    assert np.isfinite(samples).all()
    assert np.isfinite(np.asarray(median["values"])).all()
    assert len(score["values"]) == 2
    assert np.isfinite(score["total"])


def test_service_accepts_missing_model_version_field() -> None:
    """Callers may omit model_version; the response still names the served one."""
    request = _request_payload()
    del request["model_version"]

    response = _service().predict(request)

    assert response["model_version"] == MODEL_VERSION


def test_service_rejects_explicit_mismatched_model_version() -> None:
    """A client pinned to another deployment learns of the redeploy.

    Answering would hand it numbers from a model it did not ask for.
    """
    request = _request_payload()
    request["model_version"] = "jointfm-inference:0.2.0+ckpt.stale"

    error = _refusal(_service(), request)

    assert error.code == "MODEL_VERSION_MISMATCH"
    assert error.field == "model_version"


def test_service_applies_one_sided_column_bounds_to_samples() -> None:
    """A bound redraws only the column that declares it."""
    service = _service(_FixedParameterJointFM(column_locations=(0.0, 0.0), scale=1.0))
    request = _request_payload()
    request["return_mode"] = "samples"
    request["n_samples"] = 64
    request["requested_columns"] = ["known_feature", "target"]
    request["columns"][1]["lower_bound"] = 0.0

    samples = np.asarray(service.predict(request)["outputs"]["samples"])

    assert samples.shape == (64, 2, 2)
    assert samples[..., 1].min() >= 0.0
    assert samples[..., 0].min() < 0.0


def test_service_rejects_time_column_declared_in_columns() -> None:
    request = _request_payload()
    request["columns"].insert(0, {"name": "timestamp", "modality": "numeric"})

    with pytest.raises(
        ServiceError, match="must not also appear in the columns list"
    ) as exc_info:
        _service().predict(request)

    assert exc_info.value.field == "time_column"


def test_service_returns_quantiles_for_requested_columns() -> None:
    """Monte Carlo quantiles sit on the emitted Student-t marginal's closed form."""
    service = _service(_FixedParameterJointFM(scale=1.0))
    request = _request_payload()
    request["return_mode"] = "quantiles"
    request["n_samples"] = _MAX_SAMPLE_COUNT
    request["quantiles"] = [0.1, 0.9]

    response = service.predict(request)

    assert response["outputs"]["mean"] is None
    assert response["outputs"]["samples"] is None
    quantiles = response["outputs"]["quantiles"]
    assert [entry["quantile"] for entry in quantiles] == [0.1, 0.9]
    for entry in quantiles:
        expected = [
            location + stats.t.ppf(entry["quantile"], _STUDENT_T_DF)
            for location in (200.0, 201.0)
        ]
        assert [row[0] for row in entry["values"]] == pytest.approx(expected, abs=0.15)


def test_service_returns_log_prob_for_query_rows() -> None:
    """Each horizon's score is the emitted joint Student-t density of its query row."""
    service = _service(_FixedParameterJointFM(column_locations=(1.0, 10.0), scale=1.0))
    request = _request_payload()
    request["return_mode"] = "log_prob"
    request["requested_columns"] = ["known_feature", "target"]
    observed = [[1.4, 14.0], [1.5, 15.0]]
    request["query_rows"] = [
        {"timestamp": time, "known_feature": feature, "target": target}
        for time, (feature, target) in zip(request["query_times"], observed)
    ]

    response = service.predict(request)

    assert response["return_mode"] == "log_prob"
    assert response["outputs"]["mean"] is None
    assert response["outputs"]["samples"] is None
    assert response["outputs"]["quantiles"] is None
    expected = [
        stats.multivariate_t(
            loc=[1.0 + horizon, 10.0 + horizon], shape=np.eye(2), df=_STUDENT_T_DF
        ).logpdf(row)
        for horizon, row in enumerate(observed)
    ]
    score = response["outputs"]["log_prob"]
    assert score["values"] == pytest.approx(expected, rel=1e-5)
    assert score["nll_values"] == pytest.approx(
        [-value for value in expected], rel=1e-5
    )
    assert score["total"] == pytest.approx(sum(expected), rel=1e-5)
    assert score["mean"] == pytest.approx(sum(expected) / 2, rel=1e-5)
    assert score["nll_total"] == pytest.approx(-sum(expected), rel=1e-5)
    assert score["nll_mean"] == pytest.approx(-sum(expected) / 2, rel=1e-5)


def test_service_rejects_unknown_request_fields() -> None:
    """A misspelled option fails visibly instead of being ignored."""
    request = _request_payload()
    request["quantile"] = [0.5]
    request["unknown_option"] = 1

    error = _refusal(_service(), request)

    assert error.code == "VALIDATION_ERROR"
    assert error.field == "quantile"


@_TIME_INDEX_MODES
def test_service_preserves_gaps_on_the_time_grid(time_index_mode: str) -> None:
    service, model = _service_and_model()

    response = service.predict(_gapped_payload(time_index_mode))

    assert response["outputs"]["query_times"] == _GAPPED_TIMES[time_index_mode][3]
    assert response["outputs"]["mean"] == [[200.0], [202.0]]
    assert model.last_past_time_indices is not None
    assert model.last_horizons is not None
    assert torch.equal(model.last_past_time_indices, torch.tensor([0, 1, 3, 4]))
    assert torch.equal(model.last_horizons, torch.tensor([0, 2]))


def test_service_rejects_non_grid_continuous_float() -> None:
    request = _gapped_payload("continuous_float")
    request["history_rows"] = request["history_rows"][:3]
    for row, time in zip(request["history_rows"], (0.0, 0.7, 1.0)):
        row["time"] = time
    request["query_times"] = [1.4, 1.7]

    with pytest.raises(ServiceError, match="regular step grid") as exc_info:
        _service().predict(request)

    assert exc_info.value.field == "query_times"


def test_service_rejects_non_grid_absolute_datetime() -> None:
    request = _request_payload()
    request["history_rows"] = request["history_rows"][:3]
    request["history_rows"][2]["timestamp"] = "2026-01-04T10:00:00Z"
    request["query_times"] = ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"]

    with pytest.raises(ServiceError, match="regular step grid") as exc_info:
        _service().predict(request)

    assert exc_info.value.field == "query_times"


def test_service_rejects_fractional_ordinal_query_times() -> None:
    request = _ordinal_payload([1, 2, 4], [5.5])

    with pytest.raises(ServiceError, match="integer step values") as exc_info:
        _service().predict(request)

    assert exc_info.value.field == "query_times"


@_TIME_INDEX_MODES
@pytest.mark.parametrize(
    ("cfg", "message", "field"),
    [
        pytest.param(
            tiny_config(max_time_ctx=4),
            "observed time indices",
            "history_rows",
            id="context",
        ),
        pytest.param(
            tiny_config(max_time_fut=2),
            "forecast horizon indices",
            "query_times",
            id="horizon",
        ),
    ],
)
def test_service_rejects_times_beyond_the_checkpoint_time_capacity(
    time_index_mode: str, cfg: JointFMConfig, message: str, field: str
) -> None:
    """The gapped history reaches grid position 4 and the queries horizon index 2."""
    service = _service(_FixedParameterJointFM(cfg))

    with pytest.raises(ServiceError, match=message) as exc_info:
        service.predict(_gapped_payload(time_index_mode))

    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.field == field


def test_service_rejects_query_times_at_the_last_history_time() -> None:
    """A query at the last history time must not silently re-forecast that row."""
    request = _ordinal_payload([0, 1, 2, 3], [3])

    with pytest.raises(
        ServiceError, match="strictly after the last observed time"
    ) as exc_info:
        _service().predict(request)

    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.field == "query_times"


def test_service_rejects_query_times_inside_the_history() -> None:
    request = _request_payload()
    request["query_times"] = ["2026-01-01T00:00:00Z"]

    with pytest.raises(
        ServiceError, match="strictly after the last observed time"
    ) as exc_info:
        _service().predict(request)

    assert exc_info.value.field == "query_times"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_service_rejects_incomplete_or_non_finite_history(value: float | None) -> None:
    """A JointFM 1 network reads complete numeric histories only."""
    request = _request_payload()
    request["history_rows"][2]["target"] = value

    error = _refusal(_service(), request)

    assert error.code == "VALIDATION_ERROR"
    assert error.field == "history_rows"
    assert "missing or non-finite" in error.message


@pytest.mark.parametrize(
    ("descriptor", "field"),
    [
        pytest.param({"modality": "categorical"}, "columns[1].modality", id="modality"),
        pytest.param({"role": "static"}, "columns[1].role", id="role"),
        pytest.param({"nullable": True}, "columns[1].nullable", id="nullable"),
        pytest.param(
            {"traits": {"seasonal": "present"}}, "columns[1].traits", id="traits"
        ),
        pytest.param({"unit": "kg"}, "columns[1]", id="unknown-key"),
    ],
)
def test_service_refuses_column_descriptors_it_does_not_serve(
    descriptor: dict[str, Any], field: str
) -> None:
    """A descriptor the mounted model would not read is refused rather than dropped."""
    request = _request_payload()
    request["columns"][1].update(descriptor)

    error = _refusal(_service(), request)

    assert error.code == "VALIDATION_ERROR"
    assert error.field == field


def test_service_accepts_column_descriptors_at_their_defaults() -> None:
    request = _request_payload()
    request["columns"][1].update(
        {
            "nullable": False,
            "traits": {},
            "time_value_use_local_normalized_time": False,
            "time_value_calendar_id": "pandas-default",
        }
    )

    assert _service().predict(request)["outputs"]["mean"] == [[200.0], [201.0]]


@pytest.mark.parametrize(
    ("key", "value"),
    [("use_local_normalized_time", True), ("calendar_id", "business-days")],
)
def test_service_refuses_time_options_it_does_not_serve(key: str, value: Any) -> None:
    request = _request_payload()
    request[key] = value

    error = _refusal(_service(), request)

    assert error.code == "VALIDATION_ERROR"
    assert error.field == key


def test_service_accepts_time_options_at_their_defaults() -> None:
    request = _request_payload()
    request["use_local_normalized_time"] = False
    request["calendar_id"] = "pandas-default"

    assert _service().predict(request)["outputs"]["mean"] == [[200.0], [201.0]]


def test_service_reads_time_scale_and_timezone_for_absolute_datetime() -> None:
    """Both options reach the time conversion.

    A valid scale and timezone pair serves; an invalid one is refused.
    """
    service = _service()
    request = _request_payload()
    request["time_scale_seconds"] = 86400.0
    request["timezone"] = "UTC"

    assert service.predict(request)["outputs"]["mean"] == [[200.0], [201.0]]

    for key, value, message in [
        ("time_scale_seconds", 0.0, "must be positive"),
        ("timezone", "Europe/Berlin", "does not match"),
    ]:
        refused = {**request, key: value}
        error = _refusal(service, refused)
        assert error.field == "history_rows"
        assert message in error.message


def test_health_payload_reports_the_served_jointfm_1_contract() -> None:
    assert _service().health_payload() == {
        "status": "ok",
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "image_version": "0.2.0",
        "model_version": MODEL_VERSION,
        "checkpoint_version": "example-1",
        "checkpoint_path": "/models/jointfm.pt",
        "device": "cpu",
        "precision": SERVICE_PRECISION,
        "generation": "jointfm-1",
        "head": "studentt",
        "decoding_strategy": "parallel_dense",
        "supported_query_modes": ["forecast", "condition"],
        "supported_condition_kinds": ["equality", "interval"],
        "supported_metadata_traits": [],
        "supported_return_modes": ["log_prob", "mean", "quantiles", "samples"],
        "supported_time_index_modes": [
            "absolute_datetime",
            "continuous_float",
            "ordinal",
        ],
        "time_index_encoding": "discrete_grid",
        "max_sample_count": _MAX_SAMPLE_COUNT,
        "max_concurrent_requests": 1,
        "data_generation": TINY_CAPACITY_ENVELOPE,
    }


def test_health_payload_reports_no_envelope_for_a_checkpoint_without_one() -> None:
    model = _FixedParameterJointFM()
    checkpoint = checkpoint_mapping(model)
    del _inference_metadata(checkpoint)[CHECKPOINT_CAPACITY_ENVELOPE_KEY]

    payload = _service_and_model(model, checkpoint=checkpoint)[0].health_payload()

    assert payload["max_sample_count"] == _MAX_SAMPLE_COUNT
    assert payload["data_generation"] is None


def test_health_payload_reports_only_the_capacity_envelope_fields() -> None:
    """Keys the checkpoint block carries beyond the envelope never reach a client."""
    model = _FixedParameterJointFM()
    checkpoint = checkpoint_mapping(model)
    _inference_metadata(checkpoint)[CHECKPOINT_CAPACITY_ENVELOPE_KEY]["notes"] = (
        "unreported"
    )

    payload = _service_and_model(model, checkpoint=checkpoint)[0].health_payload()

    assert payload["data_generation"] == TINY_CAPACITY_ENVELOPE


def test_service_serves_only_the_time_index_modes_the_checkpoint_supports() -> None:
    """The service advertises and serves only the modes the checkpoint supports."""
    model = _FixedParameterJointFM()
    checkpoint = checkpoint_mapping(model)
    _inference_metadata(checkpoint)[CHECKPOINT_TIME_INDEX_SUPPORT_KEY][
        "supported_request_time_index_modes"
    ] = ["ordinal", "weekly"]
    service, _ = _service_and_model(model, checkpoint=checkpoint)

    assert service.health_payload()["supported_time_index_modes"] == ["ordinal"]
    response = service.predict(_ordinal_payload([0, 1, 2, 3], [4, 5]))
    assert response["outputs"]["mean"] == [[200.0], [201.0]]
    for mode in ("absolute_datetime", "weekly"):
        request = _request_payload()
        request["time_index_mode"] = mode
        with pytest.raises(
            ServiceError, match="Unsupported time_index_mode: expected one of"
        ) as exc_info:
            service.predict(request)
        assert exc_info.value.field == "time_index_mode"


def test_service_rejects_config_precision_that_differs_from_session() -> None:
    """``/healthz`` must never advertise a precision the forward does not run at."""
    session = make_session(_FixedParameterJointFM())

    with pytest.raises(ValueError, match="does not match the session precision"):
        JointFMHTTPService(
            config=make_service_config(precision="bf16"), session=session
        )


def test_predict_with_health_request_type_returns_health_payload() -> None:
    service = _service()

    assert service.predict({"request_type": "health"}) == service.health_payload()


def test_predict_with_health_request_type_ignores_other_fields() -> None:
    """Health probes succeed even when callers pin a stale schema or model version.

    Discovering the deployed pins is the point of the call: a version check
    ahead of the short-circuit would leave a client of a redeployed container
    with a mismatch error instead of the new pin.
    """
    service = _service()

    response = service.predict(
        {
            "request_type": "health",
            "schema_version": "v0-bogus",
            "model_version": "stale-version",
            "return_mode": "unsupported",
            "unknown_field": True,
        }
    )

    assert response == service.health_payload()


def test_predict_rejects_unknown_request_type() -> None:
    error = _refusal(_service(), {"request_type": "diagnose"})

    assert error.code == "VALIDATION_ERROR"
    assert error.field == "request_type"


def test_a_forecast_without_a_projection_answers_every_declared_column() -> None:
    """The projection default is role-agnostic, so a feature is answered too."""
    payload = _request_payload()
    del payload["requested_columns"]

    response = _service().predict(payload)

    assert response["outputs"]["requested_columns"] == ["known_feature", "target"]
    assert response["outputs"]["mean"] == [[100.0, 200.0], [101.0, 201.0]]


def test_a_forecast_projection_is_answered_in_the_order_it_was_given() -> None:
    """The caller's order is the answer's order, not the declared order."""
    payload = _request_payload()
    payload["requested_columns"] = ["target", "known_feature"]

    response = _service().predict(payload)

    assert response["outputs"]["requested_columns"] == ["target", "known_feature"]
    assert response["outputs"]["mean"] == [[200.0, 100.0], [201.0, 101.0]]


def test_predict_with_predict_request_type_runs_forecast() -> None:
    payload = _request_payload()
    payload["request_type"] = "predict"

    response = _service().predict(payload)

    assert response["return_mode"] == "mean"
    assert response["outputs"]["requested_columns"] == ["target"]


def test_service_rejects_mismatched_schema_version() -> None:
    """A client pinned to another schema version is refused before any forward."""
    request = _request_payload()
    request["schema_version"] = "v2"
    service, model = _service_and_model()

    error = _refusal(service, request)

    assert error.code == "SCHEMA_VERSION_MISMATCH"
    assert error.field == "schema_version"
    assert model.last_horizons is None


def test_service_rejects_an_unknown_query_mode() -> None:
    request = _request_payload()
    request["query_mode"] = "bogus"

    with pytest.raises(ServiceError, match="Unsupported query_mode") as exc_info:
        _service().predict(request)

    assert exc_info.value.code == "VALIDATION_ERROR"
    assert exc_info.value.field == "query_mode"


def test_service_rejects_unsupported_return_mode() -> None:
    request = _request_payload()
    request["return_mode"] = "median"

    with pytest.raises(ServiceError, match="Unsupported return_mode") as exc_info:
        _service().predict(request)

    assert exc_info.value.field == "return_mode"


def test_service_requires_time_column_for_absolute_datetime() -> None:
    request = _request_payload()
    del request["time_column"]

    with pytest.raises(ServiceError, match="must provide time_column") as exc_info:
        _service().predict(request)

    assert exc_info.value.field == "time_column"


def test_service_maps_history_conversion_errors_to_validation_error() -> None:
    request = _request_payload()
    rows = cast(list[dict[str, Any]], request["history_rows"])
    rows[0], rows[1] = rows[1], rows[0]

    error = _refusal(_service(), request)

    assert error.code == "VALIDATION_ERROR"
    assert error.field == "history_rows"
