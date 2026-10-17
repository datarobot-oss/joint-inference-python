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

The readout adds a per-column offset to the pooled means and to the raw
covariance diagonal, decoded from each column's own horizon token: a column's
mean and diagonal follow that column's token rather than its output position.
"""

import pytest
import torch

from jointfm_inference.model.jointfm_1.head import StudentTHead
from jointfm_inference.model.jointfm_1.readouts import ColumnLocationReadout

from tests.support.inference import student_t_head

B, H, D_CAP, K, D_MODEL = 2, 3, 5, 2, 16


def _head() -> StudentTHead:
    """Build a head whose pooled means and diagonal are the same for every column.

    With the position-specific pooled outputs flattened, any difference between
    columns comes from the per-column readout alone.
    """
    head = student_t_head(D=D_CAP, K=K, d_model=D_MODEL, covariance_rank=2)
    with torch.no_grad():
        head.out_means.weight.zero_()
        head.out_means.bias.zero_()
        head.out_diag.weight.zero_()
        head.out_diag.bias.zero_()
    return head


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


def test_readout_refuses_more_columns_than_the_head_has() -> None:
    torch.manual_seed(2)
    readout = ColumnLocationReadout(
        column_dim=D_MODEL, context_dim=8, hidden=8, n_components=K
    )
    context = torch.randn(B, H, 8)

    mean_offset, raw_diag_offset = readout(
        torch.randn(B, H, D_CAP, D_MODEL), context, capacity=D_CAP
    )

    assert mean_offset.shape == (B, H, K, D_CAP)
    assert raw_diag_offset.shape == (B, H, K, D_CAP)
    with pytest.raises(ValueError, match="exceeding the head capacity"):
        readout(torch.randn(B, H, D_CAP + 1, D_MODEL), context, capacity=D_CAP)
