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

"""Service-level tests for which future positions a condition covers.

Every condition names its positions by index into ``query_times``, ``null``
covering all of them, and the response answers every requested position. The
oracles follow from the served result drawing positions independently: a
covered position is the library's conditional at that position alone, an
uncovered one is exactly what the forecast mode answers under the same seed,
and the plausibility numbers of a multi-position request are the sums of the
per-position ones.
"""

from typing import Any

import pytest
import torch

from jointfm_inference.model.inference import (
    ConditionalReadOut,
    ConditionRequest,
    condition_at_position,
)
from jointfm_inference.model.service import ServiceError
from tests.support.condition_mode import (
    COLUMN_COUNT,
    HORIZON_COUNT,
    THREE_COLUMNS,
    bounded_columns,
    make_payload,
    make_three_column_payload,
    parameters,
    service_and_model,
    three_column_service,
)
from tests.support.inference import create_student_t_adapter


def _equality(column: str, value: float, positions: list[int] | None) -> dict:
    """Build one wire equality condition covering ``positions``."""
    return {
        "column": column,
        "kind": "equality",
        "value": value,
        "query_time_indices": positions,
    }


def _interval(
    column: str, lower: float, upper: float, positions: list[int] | None
) -> dict:
    """Build one wire interval condition covering ``positions``."""
    return {
        "column": column,
        "kind": "interval",
        "lower": lower,
        "upper": upper,
        "query_time_indices": positions,
    }


def _library_equality(pinned: float, horizon: int) -> ConditionalReadOut:
    """Condition the two-column fixture on a pinned driver at one position."""
    return condition_at_position(
        create_student_t_adapter(parameters()),
        ConditionRequest(
            horizon=horizon, equality_columns=(0,), equality_values=(pinned,)
        ),
        n_samples=100,
        column_count=COLUMN_COUNT,
    )


def test_a_condition_naming_no_positions_conditions_every_position() -> None:
    """``null`` is every position, each conditioned on its own emitted joint."""
    service, _ = service_and_model()

    response = service.predict(
        make_payload(conditions=[_equality("driver", 0.75, None)])
    )

    served = response["outputs"]["mean"]
    assert len(served) == HORIZON_COUNT
    for horizon in range(HORIZON_COUNT):
        expected = _library_equality(0.75, horizon).mean()[0, 0, 1]
        assert served[horizon][0] == pytest.approx(float(expected), abs=1e-5)


@pytest.mark.parametrize("return_mode", ["mean", "samples", "quantiles"])
def test_an_uncovered_position_answers_what_the_forecast_mode_answers(
    return_mode: str,
) -> None:
    """Positions are drawn independently, so a condition elsewhere changes nothing.

    The service draws the unconditioned answer before any conditional, so the
    uncovered position is the forecast mode's answer bit for bit under the
    same seed, which is what lets a caller line a scenario up with its base.
    """
    service, _ = service_and_model()
    shared = {
        "return_mode": return_mode,
        "n_samples": 16,
        "quantiles": [0.1, 0.9],
        "requested_columns": ["driver", "target"],
    }
    forecast = make_payload(query_mode="forecast", **shared)
    del forecast["conditions"]

    scenario = service.predict(
        make_payload(conditions=[_equality("driver", 0.75, [1])], **shared)
    )
    base = service.predict(forecast)

    served = scenario["outputs"][return_mode]
    unconditioned = base["outputs"][return_mode]
    first_position = {
        "mean": lambda rows: rows[0],
        "samples": lambda rows: [row[0] for row in rows],
        "quantiles": lambda rows: [block["values"][0] for block in rows],
    }[return_mode]
    assert first_position(served) == first_position(unconditioned)


def test_one_column_may_carry_different_conditions_at_disjoint_positions() -> None:
    """Disjoint positions never meet, so each keeps its own pin."""
    service, _ = service_and_model()

    response = service.predict(
        make_payload(
            requested_columns=["driver", "target"],
            conditions=[
                _equality("driver", 0.5, [0]),
                _equality("driver", 1.25, [1]),
            ],
        )
    )

    served = response["outputs"]["mean"]
    for horizon, pinned in enumerate([0.5, 1.25]):
        driver, target = served[horizon]
        assert driver == pytest.approx(pinned)
        expected = _library_equality(pinned, horizon).mean()[0, 0, 1]
        assert target == pytest.approx(float(expected), abs=1e-5)


@pytest.mark.parametrize(
    ("first", "second", "overlap"),
    [
        ([0], [0, 1], "[0]"),
        (None, [1], "[1]"),
        (None, None, "[0, 1]"),
    ],
    ids=["explicit", "null_against_explicit", "null_against_null"],
)
def test_two_conditions_on_one_column_at_a_shared_position_are_refused(
    first: list[int] | None, second: list[int] | None, overlap: str
) -> None:
    """A column carries one condition per position, and the refusal names where."""
    service, forward = service_and_model()

    with pytest.raises(ServiceError) as rejection:
        service.predict(
            make_payload(
                conditions=[
                    _equality("driver", 0.5, first),
                    _interval("driver", 0.0, 1.0, second),
                ]
            )
        )

    assert rejection.value.code == "VALIDATION_ERROR"
    assert rejection.value.field == "conditions"
    message = str(rejection.value)
    expected = (
        f"'driver' carries more than one condition at query_time_indices {overlap}"
    )
    assert expected in message
    assert "conditions[1] overlaps conditions[0]" in message
    assert forward.calls == 0


def test_conditions_on_different_columns_may_cover_overlapping_positions() -> None:
    """A pin everywhere and a band at one position meet only where both apply."""
    service, _ = three_column_service()

    response = service.predict(
        make_three_column_payload(
            columns=bounded_columns(*THREE_COLUMNS),
            return_mode="samples",
            n_samples=64,
            requested_columns=list(THREE_COLUMNS),
            conditions=[
                _equality("driver", 0.75, None),
                _interval("target", 0.0, 0.5, [1]),
            ],
        )
    )

    rows = response["outputs"]["samples"]
    assert len(rows) == 64
    first_targets = []
    for row in rows:
        assert row[0][0] == pytest.approx(0.75)
        assert row[1][0] == pytest.approx(0.75)
        assert 0.0 <= row[1][1] <= 0.5
        first_targets.append(row[0][1])
    # The band covers the second position alone, so the first must spill out.
    assert any(not 0.0 <= target <= 0.5 for target in first_targets)
    assert response["plausibility"]["region_log_probability"] < 0.0


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        ("missing", "must carry 'query_time_indices', null for every"),
        ([], "must name at least one position"),
        ([0, 0], "names a position more than once"),
        ([2], "entry 2 is outside the 2 requested future positions"),
        (["0"], "must hold integers"),
        ([True], "must hold integers"),
        ("0", "must be a list or null"),
    ],
    ids=["missing", "empty", "duplicate", "outside", "string", "bool", "scalar"],
)
def test_a_malformed_position_list_is_refused_before_model_execution(
    positions: Any, message: str
) -> None:
    """Every position rule is enforced at the edge, not discovered by the model."""
    service, forward = service_and_model()
    condition = _equality("driver", 0.75, None)
    if positions == "missing":
        del condition["query_time_indices"]
    else:
        condition["query_time_indices"] = positions

    with pytest.raises(ServiceError, match=message) as rejection:
        service.predict(make_payload(conditions=[condition]))

    assert rejection.value.field == "conditions"
    assert forward.calls == 0


def test_pinning_every_column_at_one_position_names_that_position() -> None:
    """The read-out rule holds per position, and the refusal says which one broke it."""
    service, forward = service_and_model()

    with pytest.raises(
        ServiceError, match="every column at query_time_indices position 1"
    ):
        service.predict(
            make_payload(
                conditions=[
                    _equality("driver", 0.75, None),
                    _equality("target", 2.0, [1]),
                ]
            )
        )

    assert forward.calls == 0


def test_the_equality_plausibility_sums_over_the_covered_positions() -> None:
    """Independent positions multiply, so their log densities add exactly."""
    service, _ = service_and_model()

    response = service.predict(
        make_payload(conditions=[_equality("driver", 0.75, None)])
    )

    densities = [
        _library_equality(0.75, horizon).equality_log_density
        for horizon in range(HORIZON_COUNT)
    ]
    expected = sum(float(density) for density in densities if density is not None)
    assert all(density is not None for density in densities)
    assert response["plausibility"]["equality_log_density"] == pytest.approx(
        expected, abs=1e-5
    )
    assert response["plausibility"]["region_log_probability"] is None


def test_a_box_at_every_position_reports_the_least_supported_estimate() -> None:
    """The region adds up over positions, and the weakest estimate is the one shown.

    The service answers covered positions in ascending order from one seeded
    generator, so the library replays the same walk position by position.
    """
    service, _ = three_column_service()
    lower, upper = (-1.0, -0.5), (1.5, 2.5)

    response = service.predict(
        make_three_column_payload(
            columns=bounded_columns(*THREE_COLUMNS),
            n_samples=64,
            requested_columns=["extra"],
            conditions=[
                _interval("driver", lower[0], upper[0], None),
                _interval("target", lower[1], upper[1], None),
            ],
        )
    )

    rng = torch.Generator().manual_seed(7)
    adapter = create_student_t_adapter(parameters(column_count=len(THREE_COLUMNS)))
    read_outs = [
        condition_at_position(
            adapter,
            ConditionRequest(
                horizon=horizon,
                interval_columns=(0, 1),
                interval_lower=lower,
                interval_upper=upper,
            ),
            n_samples=64,
            rng=rng,
            column_count=len(THREE_COLUMNS),
        )
        for horizon in range(HORIZON_COUNT)
    ]
    regions = [read_out.region_log_probability for read_out in read_outs]
    sample_sizes = [read_out.interval_effective_sample_size for read_out in read_outs]
    assert all(region is not None for region in regions)
    assert all(size is not None for size in sample_sizes)
    region = sum(float(value) for value in regions if value is not None)
    assert response["plausibility"]["region_log_probability"] == pytest.approx(
        region, abs=1e-5
    )
    estimator = response["diagnostics"]["interval_estimator"]
    assert estimator["effective_sample_size"] == pytest.approx(
        min(float(size) for size in sample_sizes if size is not None)
    )
    assert estimator["points"] == read_outs[0].interval_points


def _scored_rows(first_driver: float) -> list[dict[str, Any]]:
    """Two scored rows whose second carries the pinned driver of the tests below."""
    return [
        {"timestamp": "2026-01-05T00:00:00Z", "driver": first_driver, "target": 0.5},
        {"timestamp": "2026-01-06T00:00:00Z", "driver": 0.75, "target": 0.5},
    ]


def test_log_prob_scores_every_position_under_what_covers_it() -> None:
    """An uncovered row scores its whole joint; a covered one scores its conditional."""
    service, _ = service_and_model()
    shared = {
        "return_mode": "log_prob",
        "requested_columns": ["driver", "target"],
        "query_rows": _scored_rows(first_driver=1.4),
    }
    forecast = make_payload(query_mode="forecast", **shared)
    del forecast["conditions"]

    scenario = service.predict(
        make_payload(conditions=[_equality("driver", 0.75, [1])], **shared)
    )
    base = service.predict(forecast)

    scores = scenario["outputs"]["log_prob"]["values"]
    assert len(scores) == HORIZON_COUNT
    assert scores[0] == base["outputs"]["log_prob"]["values"][0]
    expected = _library_equality(0.75, 1).log_prob(torch.tensor([[[0.75, 0.5]]]))
    assert scores[1] == pytest.approx(float(expected), abs=1e-5)


def test_a_scored_row_contradicting_its_position_names_that_position() -> None:
    """A row is checked against the conditions of its own position, and no other."""
    service, _ = service_and_model()

    with pytest.raises(ServiceError, match="query_rows position 1") as rejection:
        service.predict(
            make_payload(
                return_mode="log_prob",
                requested_columns=["driver", "target"],
                conditions=[_equality("driver", 0.9, [1])],
                query_rows=_scored_rows(first_driver=0.9),
            )
        )

    assert rejection.value.field == "query_rows"
