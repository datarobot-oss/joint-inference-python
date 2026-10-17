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

"""``PredictorAdapter.sample`` redraws NaN draws and reports the ones it cannot replace."""

from unittest.mock import MagicMock

import pytest
import torch

from jointfm_inference.model.inference import PredictorAdapter
from jointfm_inference.model.jointfm_1.head import StudentTHead

N_SAMPLES = 5


def _params() -> dict[str, torch.Tensor]:
    """Return finite single-column parameters; the mocked head ignores them."""
    return {
        "logits": torch.zeros(1, 1, 1),
        "means": torch.zeros(1, 1, 1, 1),
        "chol": torch.ones(1, 1, 1, 1, 1),
        "df": torch.full((1, 1, 1), 5.0),
    }


def test_persistent_nan_draws_fail_with_a_nan_violation() -> None:
    head = MagicMock(spec=StudentTHead)
    head.sample.return_value = torch.full((N_SAMPLES, 1, 1), float("nan"))

    with pytest.raises(
        RuntimeError, match="Failed to satisfy constraints.*NaN violation"
    ):
        PredictorAdapter(head, _params()).sample(n_samples=N_SAMPLES, max_retries=2)
    assert head.sample.call_count == 3


def test_a_nan_draw_is_replaced_by_the_next_clean_draw() -> None:
    bad = torch.zeros(N_SAMPLES, 1, 1)
    bad[0, 0, 0] = float("nan")
    head = MagicMock(spec=StudentTHead)
    head.sample.side_effect = [bad, torch.ones(N_SAMPLES, 1, 1)]

    result = PredictorAdapter(head, _params()).sample(
        n_samples=N_SAMPLES, max_retries=10
    )

    assert result.shape == (1, N_SAMPLES, 1, 1)
    assert result[0, 0, 0, 0] == 1.0
    assert (result[0, 1:] == 0.0).all()
    assert head.sample.call_count == 2
