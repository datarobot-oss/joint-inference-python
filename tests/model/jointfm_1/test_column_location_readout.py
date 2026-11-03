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

"""Tests for the per-column location readout of the Student-t head.

Every column's mean and raw covariance diagonal is a base shared by all
columns plus a term the readout decodes from that column's own horizon token: a
column's mean and diagonal follow that column's token rather than its output
position.
"""

import pytest
import torch

from jointfm_inference.model.jointfm_1.head import StudentTHead
from jointfm_inference.model.jointfm_1.readouts import (
    LOCATION_OUTPUTS,
    ColumnLocationReadout,
)

from tests.support.inference import student_t_head

B, H, D_CAP, K, D_MODEL = 2, 3, 5, 2, 16


def _head() -> StudentTHead:
    """Build a low-rank Student-t head over ``D_CAP`` columns."""
    return student_t_head(D=D_CAP, K=K, d_model=D_MODEL, covariance_rank=2)


def _inputs(active_dim: int = D_CAP) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a pooled context and per-column tokens."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(B, H, D_MODEL, generator=generator)
    tokens = torch.randn(B, H, active_dim, D_MODEL, generator=generator)
    return x, tokens


def test_means_and_diagonal_follow_the_column_token() -> None:
    """Swapping two columns' tokens swaps their means and diagonals."""
    head = _head()
    x, tokens = _inputs()
    swapped_tokens = tokens.clone()
    swapped_tokens[..., [0, 1], :] = tokens[..., [1, 0], :]
    with torch.no_grad():
        base = head(x, column_tokens=tokens)
        swapped = head(x, column_tokens=swapped_tokens)

    for key in ("means", "lowrank_diag"):
        torch.testing.assert_close(swapped[key][..., 0], base[key][..., 1])
        torch.testing.assert_close(swapped[key][..., 1], base[key][..., 0])
        torch.testing.assert_close(swapped[key][..., 2:], base[key][..., 2:])
    assert not torch.allclose(base["means"][..., 0], base["means"][..., 1])


def test_without_column_terms_every_column_sits_at_the_shared_base() -> None:
    """No output position carries a location of its own."""
    head = _head()
    with torch.no_grad():
        head.column_location.readout.output.weight.zero_()
    x, tokens = _inputs()
    with torch.no_grad():
        params = head(x, column_tokens=tokens)

    for key in ("means", "lowrank_diag"):
        torch.testing.assert_close(
            params[key], params[key][..., :1].expand(-1, -1, -1, D_CAP)
        )


def test_readout_pads_absent_columns_with_zero_terms_and_refuses_extra_ones() -> None:
    torch.manual_seed(2)
    readout = ColumnLocationReadout(
        column_dim=D_MODEL, context_dim=8, hidden=8, n_components=K
    )
    context = torch.randn(B, H, 8)
    active_dim = D_CAP - 2

    terms = readout(torch.randn(B, H, active_dim, D_MODEL), context, capacity=D_CAP)

    assert terms.shape == (B, H, K, D_CAP, LOCATION_OUTPUTS)
    assert terms[..., :active_dim, :].abs().sum() > 0
    assert torch.equal(
        terms[..., active_dim:, :], torch.zeros_like(terms[..., active_dim:, :])
    )
    with pytest.raises(ValueError, match="exceeding the head capacity"):
        readout(torch.randn(B, H, D_CAP + 1, D_MODEL), context, capacity=D_CAP)
