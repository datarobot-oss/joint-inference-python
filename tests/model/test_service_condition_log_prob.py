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

"""Service-level tests for the ``log_prob`` return mode under conditions.

Scoring under a condition is the one return mode where the caller supplies the
values, so its oracles are the library's own conditional scorer on the same
parameters, and its rejections are rows that contradict what a condition
fixed at their position.
"""

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from jointfm_inference.model.inference import ConditionRequest, condition_at_position
from jointfm_inference.model.service import ServiceError
from tests.support.condition_mode import (
    COLUMN_COUNT,
    THREE_COLUMNS,
    bounded_columns,
    make_payload,
    make_three_column_payload,
    parameters,
    service_and_model,
    three_column_service,
)
from tests.support.inference import create_student_t_adapter


def test_log_prob_under_an_interval_matches_the_library() -> None:
    """The served score is the library's own answer under the same seed.

    Both sides draw the same truncated points and pool the same conditionals,
    so they agree to float precision.
    """
    service, _ = service_and_model()
    block = [
        {
            "column": "driver",
            "kind": "interval",
            "lower": 0.0,
            "upper": 2.0,
            "query_time_indices": [1],
        }
    ]

    response = service.predict(
        make_payload(
            columns=bounded_columns("driver", "target"),
            return_mode="log_prob",
            n_samples=256,
            requested_columns=["driver", "target"],
            conditions=block,
            query_rows=[
                {"timestamp": "2026-01-05T00:00:00Z", "driver": 1.0, "target": 0.5},
                {"timestamp": "2026-01-06T00:00:00Z", "driver": 1.0, "target": 0.5},
            ],
        )
    )

    scores = response["outputs"]["log_prob"]["values"]
    assert len(scores) == 2
    expected = condition_at_position(
        create_student_t_adapter(parameters()),
        ConditionRequest(
            horizon=1,
            interval_columns=(0,),
            interval_lower=(0.0,),
            interval_upper=(2.0,),
        ),
        n_samples=256,
        rng=torch.Generator().manual_seed(7),
        column_count=COLUMN_COUNT,
    ).log_prob(torch.tensor([[[1.0, 0.5]]]))
    assert scores[1] == pytest.approx(float(expected), abs=1e-4)


def _log_prob_rows(**values: float) -> list[dict[str, Any]]:
    """Supply one scored row per requested future position, all carrying ``values``."""
    return [{"timestamp": f"2026-01-0{day}T00:00:00Z", **values} for day in (5, 6)]


def _in_process_conditional_log_prob(
    request: ConditionRequest,
    values: Sequence[float],
    *,
    column_count: int,
    n_samples: int,
) -> float:
    """Score values over every column through the library on the same parameters."""
    read_out = condition_at_position(
        create_student_t_adapter(parameters(column_count=column_count)),
        request,
        n_samples=n_samples,
        rng=torch.Generator().manual_seed(7),
        column_count=column_count,
    )
    return float(read_out.log_prob(torch.tensor([[list(values)]])))


def test_an_equality_conditioned_log_prob_is_served_and_matches_the_library() -> None:
    """A pinned column sits in the projection, yet adds nothing to the score.

    The scoring tensor is sourced from every declared column of ``query_rows``,
    and the score stays the conditional density of the columns the request does
    not condition. Equality-only conditioning is closed form, so both sides must
    agree exactly.
    """
    service, _ = service_and_model()
    payload = make_payload(
        return_mode="log_prob",
        query_rows=_log_prob_rows(driver=0.75, target=0.5),
    )
    del payload["requested_columns"]

    response = service.predict(payload)

    assert response["outputs"]["requested_columns"] == ["driver", "target"]
    scores = response["outputs"]["log_prob"]["values"]
    assert len(scores) == 2
    expected = condition_at_position(
        create_student_t_adapter(parameters()),
        ConditionRequest(horizon=1, equality_columns=(0,), equality_values=(0.75,)),
        n_samples=100,
        column_count=COLUMN_COUNT,
    ).log_prob(torch.tensor([[[0.75, 0.5]]]))
    assert scores[1] == pytest.approx(float(expected), abs=1e-5)
    assert response["plausibility"]["equality_log_density"] < 0.0


def test_a_mixed_conditioned_log_prob_matches_the_library() -> None:
    """Both kinds together: the pinned column is checked, the bounded one confined."""
    service, _ = three_column_service()
    block = [
        {
            "column": "driver",
            "kind": "equality",
            "value": 0.75,
            "query_time_indices": [1],
        },
        {
            "column": "target",
            "kind": "interval",
            "lower": 0.0,
            "upper": 2.0,
            "query_time_indices": [1],
        },
    ]

    response = service.predict(
        make_three_column_payload(
            columns=bounded_columns(*THREE_COLUMNS),
            return_mode="log_prob",
            n_samples=256,
            requested_columns=list(THREE_COLUMNS),
            conditions=block,
            query_rows=_log_prob_rows(driver=0.75, target=1.0, extra=0.5),
        )
    )

    scores = response["outputs"]["log_prob"]["values"]
    assert len(scores) == 2
    expected = _in_process_conditional_log_prob(
        ConditionRequest(
            horizon=1,
            equality_columns=(0,),
            equality_values=(0.75,),
            interval_columns=(1,),
            interval_lower=(0.0,),
            interval_upper=(2.0,),
        ),
        [0.75, 1.0, 0.5],
        column_count=len(THREE_COLUMNS),
        n_samples=256,
    )
    assert scores[1] == pytest.approx(expected, abs=1e-4)


def test_a_scored_value_contradicting_the_pinned_column_is_refused() -> None:
    """The density of an event the request excluded is not a number to return."""
    service, _ = service_and_model()
    payload = make_payload(
        return_mode="log_prob",
        query_rows=_log_prob_rows(driver=0.9, target=0.5),
    )
    del payload["requested_columns"]

    with pytest.raises(
        ServiceError, match="contradict the equality condition"
    ) as rejection:
        service.predict(payload)

    assert rejection.value.field == "query_rows"


def test_a_scored_value_outside_the_interval_is_refused() -> None:
    """A bounded column's supplied value must lie inside its region."""
    service, _ = three_column_service()

    with pytest.raises(
        ServiceError, match="outside the interval condition"
    ) as rejection:
        service.predict(
            make_three_column_payload(
                columns=bounded_columns(*THREE_COLUMNS),
                return_mode="log_prob",
                requested_columns=list(THREE_COLUMNS),
                conditions=[
                    {
                        "column": "driver",
                        "kind": "equality",
                        "value": 0.75,
                        "query_time_indices": [1],
                    },
                    {
                        "column": "target",
                        "kind": "interval",
                        "lower": 0.0,
                        "upper": 2.0,
                        "query_time_indices": [1],
                    },
                ],
                query_rows=_log_prob_rows(driver=0.75, target=5.0, extra=0.5),
            )
        )

    assert rejection.value.field == "query_rows"


def test_a_conditioned_log_prob_still_demands_every_declared_column() -> None:
    """The scorer covers the whole joint, so a partial projection is refused."""
    service, _ = three_column_service()

    with pytest.raises(ServiceError, match="every declared column") as rejection:
        service.predict(
            make_three_column_payload(
                columns=bounded_columns(*THREE_COLUMNS),
                return_mode="log_prob",
                requested_columns=["target"],
                query_rows=_log_prob_rows(driver=0.75, target=1.0, extra=0.5),
            )
        )

    assert rejection.value.field == "requested_columns"
    assert "driver" in str(rejection.value)


def test_a_conditioned_log_prob_needs_no_projection_at_all() -> None:
    """The projection log_prob demands is what omitting the field already resolves to.

    The score stays the conditional density of the columns the request does not
    condition, whatever the projection says: a pinned column's conditional is a
    point mass and has no density to add.
    """
    service, _ = service_and_model()
    rows = _log_prob_rows(driver=0.75, target=0.5)

    stated = service.predict(
        make_payload(
            return_mode="log_prob",
            requested_columns=["driver", "target"],
            query_rows=rows,
        )
    )
    omitted_payload = make_payload(return_mode="log_prob", query_rows=rows)
    del omitted_payload["requested_columns"]
    omitted = service.predict(omitted_payload)

    assert omitted["outputs"]["requested_columns"] == ["driver", "target"]
    assert omitted["outputs"]["log_prob"] == stated["outputs"]["log_prob"]
