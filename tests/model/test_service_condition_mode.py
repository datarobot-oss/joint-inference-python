"""Service-level tests for the ``condition`` query mode.

The oracles mirror the served contract's own acceptance evidence. An equality
round trip is checked against the *same* parameters conditioned in process
through ``PredictorAdapter.condition``, because both sides run one closed form
and must therefore agree exactly rather than approximately. An interval
condition is checked against the acceptance rate of plain rejection sampling
from the same mixture, which is the only independent oracle a positive-mass
region has, and a one-column tail region against the Student-t distribution
function itself.

Two rejections are checked for *when* they happen rather than only for their
code: a backend that cannot condition and a malformed block must both be
refused before the model forward, which the fixture counts.
"""

import math
from typing import Any

import pytest
import torch
from scipy.special import stdtr

from jointfm_inference.model.inference.jointfm_1_backend import JointFM1InferenceBackend
from jointfm_inference.model.service import ServiceError
from tests.support.condition_mode import (
    COLUMN_COUNT,
    HORIZON_COUNT,
    PARAMETERIZATIONS,
    STUDENT_T_DF,
    THREE_COLUMNS,
    bounded_columns,
    make_payload,
    make_three_column_payload,
    parameters,
    service_and_model,
    three_column_service,
    random_weight_service,
)
from tests.support.inference import create_student_t_adapter


def _in_process_conditional_mean(pinned: float, horizon: int) -> float:
    """Condition the same parameters in process, through the library entry point."""
    values = torch.zeros(1, HORIZON_COUNT, COLUMN_COUNT)
    values[..., 0] = pinned
    conditional = create_student_t_adapter(parameters()).condition(
        values, torch.tensor([True, False])
    )
    weights = torch.softmax(conditional.params["logits"][0, horizon], dim=-1)
    return float((weights * conditional.params["means"][0, horizon, :, 0]).sum())


def _cannot_condition(backend: JointFM1InferenceBackend) -> bool:
    """Report a backend whose results answer no conditioning request."""
    del backend
    return False


@pytest.fixture
def backend_without_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every backend mounted in the test report that it cannot condition."""
    monkeypatch.setattr(
        JointFM1InferenceBackend, "supports_conditioning", property(_cannot_condition)
    )


def test_health_advertises_condition_for_a_conditioning_backend() -> None:
    """A caller must learn the capability from the advertisement, not from an error."""
    service, _ = service_and_model()

    payload = service.health_payload()

    assert payload["supported_query_modes"] == ["forecast", "condition"]
    assert payload["supported_condition_kinds"] == ["equality", "interval"]


@pytest.mark.usefixtures("backend_without_conditioning")
def test_health_advertises_no_condition_for_a_backend_that_cannot_answer() -> None:
    """The advertisement is derived from the mounted backend, not hardcoded."""
    service, _ = service_and_model()

    payload = service.health_payload()

    assert payload["supported_query_modes"] == ["forecast"]
    assert payload["supported_condition_kinds"] == []


@pytest.mark.usefixtures("backend_without_conditioning")
def test_a_backend_that_cannot_condition_is_refused_before_model_execution() -> None:
    """The rejection must cost no forward pass, and must name the enumerated code."""
    service, forward = service_and_model()

    with pytest.raises(ServiceError) as rejection:
        service.predict(make_payload())

    assert rejection.value.code == "UNSUPPORTED_HEAD_QUERY_COMBINATION"
    assert rejection.value.field == "query_mode"
    assert forward.calls == 0


def test_equality_condition_matches_the_same_parameters_conditioned_in_process() -> (
    None
):
    """Both sides run one closed form, so they must agree exactly, not approximately."""
    service, _ = service_and_model()

    response = service.predict(make_payload())

    assert response["head"] == "studentt"
    assert response["query_mode"] == "condition"
    assert response["outputs"]["mean"][1][0] == pytest.approx(
        _in_process_conditional_mean(0.75, 1), abs=1e-5
    )


def test_a_tail_region_reports_the_student_t_tail_mass() -> None:
    """One interval column on a tail is answered by the distribution function.

    One interval column is exact, so the reported mass must be the mixture of
    the components' Student-t tail masses, which at this depth is far above
    what a normal mixture with the same locations and scales would give.
    """
    floor = 8.0
    service, _ = service_and_model()

    response = service.predict(
        make_payload(
            conditions=[
                {
                    "column": "driver",
                    "kind": "interval",
                    "lower": floor,
                    "upper": None,
                    "query_time_indices": [0],
                }
            ]
        )
    )

    params = parameters()
    weights = torch.softmax(params["logits"][0, 0], dim=-1)
    standardized = (floor - params["means"][0, 0, :, 0]) / params["chol"][0, 0, :, 0, 0]
    expected = float(
        (weights * torch.tensor(1.0 - stdtr(STUDENT_T_DF, standardized.numpy()))).sum()
    )
    normal = float((weights * torch.special.ndtr(-standardized)).sum())

    reported = math.exp(response["plausibility"]["region_log_probability"])
    # The walk is exact here; the slack is the float32 the head emits in,
    # which the double-precision walk cannot recover.
    assert reported == pytest.approx(expected, rel=1e-4)
    assert reported > 100.0 * normal


def test_the_response_answers_every_requested_position() -> None:
    """A condition changes what a position answers, never which positions answer."""
    service, _ = service_and_model()

    response = service.predict(make_payload())

    assert response["outputs"]["query_times"] == [
        "2026-01-05T00:00:00+00:00",
        "2026-01-06T00:00:00+00:00",
    ]
    assert response["diagnostics"]["horizon_count"] == 2
    assert len(response["outputs"]["mean"]) == 2


def test_an_equality_request_reports_its_plausibility_and_no_region() -> None:
    """Plausibility is per kind: no interval means no region probability."""
    service, _ = service_and_model()

    response = service.predict(make_payload())

    plausibility = response["plausibility"]
    assert plausibility["equality_log_density"] < 0.0
    assert plausibility["region_log_probability"] is None
    assert response["diagnostics"]["interval_estimator"] is None


def test_an_interval_region_probability_matches_plain_rejection_sampling() -> None:
    """The one oracle a positive-mass region has is the sampler it replaces."""
    service, _ = service_and_model()
    floor = 0.5
    response = service.predict(
        make_payload(
            conditions=[
                {
                    "column": "driver",
                    "kind": "interval",
                    "lower": floor,
                    "upper": None,
                    "query_time_indices": [0],
                }
            ]
        )
    )

    draws = create_student_t_adapter(parameters()).sample(
        n_samples=40_000, rng=torch.Generator().manual_seed(5)
    )
    acceptance = float((draws[0, :, 0, 0] >= floor).float().mean())

    reported = math.exp(response["plausibility"]["region_log_probability"])
    assert reported == pytest.approx(acceptance, abs=0.01)
    assert response["plausibility"]["equality_log_density"] is None


def test_a_single_interval_column_reports_no_estimator() -> None:
    """Exactness reaches the caller: one interval column needs no estimation."""
    service, _ = service_and_model()

    response = service.predict(
        make_payload(
            return_mode="samples",
            n_samples=64,
            conditions=[
                {
                    "column": "driver",
                    "kind": "interval",
                    "lower": 0.0,
                    "upper": 2.0,
                    "query_time_indices": [0],
                }
            ],
        )
    )

    assert response["diagnostics"]["interval_estimator"] is None
    assert response["diagnostics"]["condition_draws"] == 64


def test_a_network_with_its_own_forward_answers_a_condition_request() -> None:
    """What a deployment mounts is a network, not a parameter mapping.

    Every other test here replaces the forward with one that returns the
    emitted mixture and nothing else, so none of them sees what a real forward
    hands back: the history's standardization center and scale ride along with
    the parameters. Conditioning reads that mapping per column and per
    emission position, which those two entries have no business answering.
    """
    service = random_weight_service()

    response = service.predict(make_payload())

    assert len(response["outputs"]["query_times"]) == 2
    assert response["outputs"]["requested_columns"] == ["target"]
    assert math.isfinite(response["outputs"]["mean"][1][0])
    assert math.isfinite(response["plausibility"]["equality_log_density"])


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (
            [
                {
                    "column": "absent",
                    "kind": "equality",
                    "value": 1.0,
                    "query_time_indices": [0],
                }
            ],
            "unknown column",
        ),
        (
            [
                {
                    "column": "driver",
                    "kind": "equality",
                    "value": 1.0,
                    "query_time_indices": [0],
                },
                {
                    "column": "driver",
                    "kind": "interval",
                    "lower": 0.0,
                    "upper": 1.0,
                    "query_time_indices": [0],
                },
            ],
            "more than one condition",
        ),
        (
            [
                {
                    "column": "driver",
                    "kind": "equality",
                    "value": 1.0,
                    "query_time_indices": [5],
                }
            ],
            "outside the 2 requested future positions",
        ),
        (
            [{"column": "driver", "value": 1.0, "query_time_indices": [0]}],
            "kind must be one of",
        ),
        (
            [
                {
                    "column": "driver",
                    "kind": "interval",
                    "lower": 2.0,
                    "upper": 1.0,
                    "query_time_indices": [0],
                }
            ],
            "needs lower < upper",
        ),
        (
            [
                {
                    "column": "driver",
                    "kind": "interval",
                    "lower": None,
                    "upper": None,
                    "query_time_indices": [0],
                }
            ],
            "open on both sides",
        ),
        ([], "at least one condition"),
    ],
)
def test_a_malformed_condition_block_is_refused_before_model_execution(
    block: list[dict[str, Any]], message: str
) -> None:
    """Every request-shape rule is enforced at the edge, not discovered by the model."""
    service, forward = service_and_model()

    with pytest.raises(ServiceError, match=message) as rejection:
        service.predict(make_payload(conditions=block))

    assert rejection.value.code == "VALIDATION_ERROR"
    assert rejection.value.field == "conditions"
    assert forward.calls == 0


@pytest.mark.parametrize("return_mode", ["mean", "samples", "quantiles"])
def test_a_pinned_column_is_readable_and_carries_the_requested_value(
    return_mode: str,
) -> None:
    """A projection is role- and condition-agnostic, so a pin reads back its own value.

    The pinned column is the request's own constant rather than a model output,
    which is exactly why it must appear beside the read-outs: a scenario frame
    and the unconditioned frame it is compared against then carry the same
    columns in the same order, and the pin is visible in the comparison instead
    of having to be remembered.
    """
    service, _ = service_and_model()

    response = service.predict(
        make_payload(
            return_mode=return_mode,
            n_samples=8,
            quantiles=[0.1, 0.9],
            requested_columns=["driver", "target"],
        )
    )

    assert response["outputs"]["requested_columns"] == ["driver", "target"]
    rows = response["outputs"][return_mode]
    if return_mode == "mean":
        drivers = [rows[1][0]]
    elif return_mode == "samples":
        drivers = [row[1][0] for row in rows]
    else:
        drivers = [block["values"][1][0] for block in rows]
    assert drivers == [pytest.approx(0.75)] * len(drivers)


def test_a_projection_may_reorder_a_pinned_column_after_the_read_outs() -> None:
    """The caller's order is the answer's order, whatever a condition says."""
    service, _ = service_and_model()

    response = service.predict(make_payload(requested_columns=["target", "driver"]))

    assert response["outputs"]["requested_columns"] == ["target", "driver"]
    _target, driver = response["outputs"]["mean"][1]
    assert driver == pytest.approx(0.75)


def test_an_interval_column_stays_readable_and_carries_its_truncated_draws() -> None:
    """An interval fixes a region; the column's distribution inside it is an answer."""
    service, _ = service_and_model()

    response = service.predict(
        make_payload(
            return_mode="samples",
            n_samples=128,
            requested_columns=["driver", "target"],
            conditions=[
                {
                    "column": "driver",
                    "kind": "interval",
                    "lower": 0.5,
                    "upper": 1.5,
                    "query_time_indices": [0],
                }
            ],
        )
    )

    rows = response["outputs"]["samples"]
    assert len(rows) == 128
    for row in rows:
        driver, _target = row[0]
        assert 0.5 <= driver <= 1.5


def test_conditioning_every_column_points_at_the_log_prob_return_mode() -> None:
    """An all-pinned request is the density question log_prob already answers."""
    service, forward = service_and_model()
    payload = make_payload()
    del payload["requested_columns"]
    payload["conditions"] = [
        {
            "column": "driver",
            "kind": "equality",
            "value": 1.0,
            "query_time_indices": [0],
        },
        {
            "column": "target",
            "kind": "equality",
            "value": 2.0,
            "query_time_indices": [0],
        },
    ]

    with pytest.raises(ServiceError, match="log_prob return mode"):
        service.predict(payload)

    assert forward.calls == 0


def test_the_default_projection_answers_every_declared_column_in_declared_order() -> (
    None
):
    """No projection answers the whole position, conditioned columns included."""
    service, _ = service_and_model()
    payload = make_payload()
    del payload["requested_columns"]

    response = service.predict(payload)

    assert response["outputs"]["requested_columns"] == ["driver", "target"]
    driver, _target = response["outputs"]["mean"][1]
    assert driver == pytest.approx(0.75)


def test_a_condition_and_a_forecast_answer_in_the_same_layout() -> None:
    """The property the default exists for: a scenario frame lines up with the base one.

    A caller comparing a conditioned slice against the unconditioned forecast
    reads both frames column by column, so the two projections must agree
    without the caller restating one on either request.
    """
    service, _ = service_and_model()
    conditioned = make_payload()
    del conditioned["requested_columns"]
    unconditioned = make_payload()
    del unconditioned["requested_columns"]
    del unconditioned["conditions"]
    unconditioned["query_mode"] = "forecast"

    scenario = service.predict(conditioned)
    base = service.predict(unconditioned)

    assert (
        scenario["outputs"]["requested_columns"]
        == base["outputs"]["requested_columns"]
        == ["driver", "target"]
    )


def test_conditions_without_the_condition_mode_are_refused() -> None:
    """Conditions the service would ignore are a request that means something else."""
    service, _ = service_and_model()

    with pytest.raises(ServiceError, match="query_mode='condition'") as rejection:
        service.predict(make_payload(query_mode="forecast"))

    assert rejection.value.field == "conditions"


def test_the_condition_mode_without_conditions_is_refused() -> None:
    """The mode is the request's shape, so the conditions are not optional."""
    service, _ = service_and_model()
    payload = make_payload()
    del payload["conditions"]

    with pytest.raises(ServiceError, match="must carry a conditions list"):
        service.predict(payload)


@PARAMETERIZATIONS
def test_both_scale_parameterizations_answer_the_closed_form(lowrank: bool) -> None:
    """Dense and diagonal-plus-factor scales serve the same conditional.

    The declared columns carry bounds, which the service hands over as one
    tensor per request column, so the read-out must index them by column.
    """
    service, _ = service_and_model(params=parameters(lowrank=lowrank))

    response = service.predict(
        make_payload(columns=bounded_columns("driver", "target"))
    )

    assert response["outputs"]["requested_columns"] == ["target"]
    assert response["outputs"]["mean"][1][0] == pytest.approx(
        _in_process_conditional_mean(0.75, 1), abs=1e-4
    )


@PARAMETERIZATIONS
def test_both_kinds_compose_with_every_draw_inside(lowrank: bool) -> None:
    """A pin and an interval compose over three columns, on both scale forms."""
    service, _ = three_column_service(lowrank=lowrank)

    response = service.predict(
        make_three_column_payload(
            columns=bounded_columns(*THREE_COLUMNS),
            return_mode="samples",
            n_samples=96,
            requested_columns=["target", "extra"],
            conditions=[
                {
                    "column": "driver",
                    "kind": "equality",
                    "value": 0.75,
                    "query_time_indices": [0],
                },
                {
                    "column": "target",
                    "kind": "interval",
                    "lower": 0.0,
                    "upper": 1.5,
                    "query_time_indices": [0],
                },
            ],
        )
    )

    plausibility = response["plausibility"]
    assert plausibility["equality_log_density"] < 0.0
    assert plausibility["region_log_probability"] < 0.0
    rows = response["outputs"]["samples"]
    assert len(rows) == 96
    for row in rows:
        target, _extra = row[0]
        assert 0.0 <= target <= 1.5


def test_one_answer_carries_a_pin_an_interval_and_a_read_out_together() -> None:
    """The three kinds of column share one layout, each saying what it means.

    Every draw has to be internally consistent: the pin is the request's own
    constant on every row, the bounded column lies inside its region on the
    same row, and the read-out is drawn from the conditional of that very draw
    rather than from a separately summarized marginal.
    """
    service, _ = three_column_service()
    payload = make_three_column_payload(
        columns=bounded_columns(*THREE_COLUMNS),
        return_mode="samples",
        n_samples=64,
        conditions=[
            {
                "column": "driver",
                "kind": "equality",
                "value": 0.75,
                "query_time_indices": [0],
            },
            {
                "column": "target",
                "kind": "interval",
                "lower": 0.0,
                "upper": 1.5,
                "query_time_indices": [0],
            },
        ],
    )
    del payload["requested_columns"]

    response = service.predict(payload)

    assert response["outputs"]["requested_columns"] == list(THREE_COLUMNS)
    rows = response["outputs"]["samples"]
    assert len(rows) == 64
    extras = []
    for row in rows:
        driver, target, extra = row[0]
        assert driver == pytest.approx(0.75)
        assert 0.0 <= target <= 1.5
        extras.append(extra)
    # A constant read-out would satisfy every assertion above while carrying no
    # conditional at all, so the draws must actually vary.
    assert len(set(extras)) > 1


def test_declared_bounds_filter_the_drawn_read_out_columns() -> None:
    """Declared bounds filter the drawn read-out columns, as on the forecast path."""
    service, _ = service_and_model()
    columns = bounded_columns("driver", "target")
    columns[1]["lower_bound"] = 0.0

    response = service.predict(
        make_payload(columns=columns, return_mode="samples", n_samples=96)
    )

    rows = response["outputs"]["samples"]
    assert len(rows) == 96
    for row in rows:
        (target,) = row[0]
        assert target >= 0.0


def test_quantiles_cover_the_requested_read_out_column() -> None:
    """Quantiles come back for the request's read-out column alone, in order."""
    service, _ = service_and_model()

    response = service.predict(
        make_payload(
            columns=bounded_columns("driver", "target"),
            return_mode="quantiles",
            quantiles=[0.1, 0.5, 0.9],
            n_samples=512,
        )
    )

    quantiles = response["outputs"]["quantiles"]
    assert [entry["quantile"] for entry in quantiles] == [0.1, 0.5, 0.9]
    values = [entry["values"][0][0] for entry in quantiles]
    assert len(quantiles[0]["values"][0]) == 1
    assert values[0] < values[1] < values[2]


def test_parameters_of_another_width_are_refused() -> None:
    """A network whose output does not match the declared columns is a defect."""
    service, _ = three_column_service()

    with pytest.raises(RuntimeError, match="cover 3 columns, the request declares 2"):
        service.predict(make_payload())
