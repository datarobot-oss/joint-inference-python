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

"""Tests for the covariance factorization and whitening helpers."""

import pytest
import torch

from jointfm_inference.model.linalg import (
    apply_selected_cholesky,
    cholesky_from_lowrank,
    compute_whitened_residuals,
    robust_cholesky,
)


def test_a_singular_lowrank_covariance_still_factorizes() -> None:
    """An all-zero covariance has no plain Cholesky factor; the jitter supplies one."""
    with pytest.raises(RuntimeError):
        torch.linalg.cholesky(torch.zeros(3, 3))

    factor = cholesky_from_lowrank(torch.zeros(3), torch.zeros(3, 1))

    assert torch.all(torch.diagonal(factor, dim1=-2, dim2=-1) > 0)


def test_cholesky_from_lowrank_factorizes_diag_plus_factor() -> None:
    generator = torch.Generator().manual_seed(0)
    diag = torch.rand(2, 4, generator=generator, dtype=torch.float64) + 0.5
    U = torch.randn(2, 4, 2, generator=generator, dtype=torch.float64)

    factor = cholesky_from_lowrank(diag, U)
    torch.testing.assert_close(
        factor @ factor.transpose(-1, -2),
        torch.diag_embed(diag) + U @ U.transpose(-1, -2),
    )

    diagonal_only = cholesky_from_lowrank(diag, None)
    torch.testing.assert_close(diagonal_only, torch.diag_embed(diag.sqrt()))


def test_robust_cholesky_is_equivariant_under_per_variable_rescaling() -> None:
    """Rescaling a variable must rescale its factor, not perturb the others.

    An absolute jitter would inflate a variable whose variance is small in the
    caller's units by a multiple of its true value; the jitter is a fraction
    of each variance, so the identity holds to floating-point precision.
    """
    correlation = torch.tensor(
        [[1.0, 0.9, 0.4], [0.9, 1.0, 0.35], [0.4, 0.35, 1.0]], dtype=torch.float64
    )
    scales = torch.tensor([1e-3, 1e2, 1.0], dtype=torch.float64)
    rescaled = correlation * torch.outer(scales, scales)

    factor = robust_cholesky(correlation)
    rescaled_factor = robust_cholesky(rescaled)

    assert torch.allclose(
        rescaled_factor, scales.unsqueeze(-1) * factor, rtol=1e-12, atol=0.0
    )


def test_robust_cholesky_survives_one_degenerate_variable() -> None:
    """A variable with no variance has no scale to divide by and must not produce NaN."""
    sigma = torch.tensor(
        [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )

    factor = robust_cholesky(sigma)

    assert torch.isfinite(factor).all()
    assert torch.all(torch.diagonal(factor) > 0)


def test_robust_cholesky_keeps_reduced_precision_inputs_in_their_dtype() -> None:
    sigma = torch.eye(2, dtype=torch.bfloat16) * 4.0
    factor = robust_cholesky(sigma)
    assert factor.dtype == torch.bfloat16
    torch.testing.assert_close(factor.float(), 2.0 * torch.eye(2), atol=1e-2, rtol=0.0)


def test_apply_selected_cholesky_applies_each_draws_own_component() -> None:
    generator = torch.Generator().manual_seed(1)
    chol = torch.linalg.cholesky(
        torch.eye(3) + 0.1 * torch.ones(2, 4, 3, 3)
    ) * torch.rand(2, 4, 1, 1, generator=generator)
    noise = torch.randn(2, 5, 3, generator=generator)
    picks = torch.randint(0, 4, (2, 5), generator=generator)
    selector = torch.nn.functional.one_hot(picks, num_classes=4).float()

    out = apply_selected_cholesky(selector, chol, noise)

    expected = torch.stack(
        [
            torch.stack([chol[b, picks[b, n]] @ noise[b, n] for n in range(5)])
            for b in range(2)
        ]
    )
    torch.testing.assert_close(out, expected)


def test_whitened_residuals_solve_the_triangular_system_with_and_without_folding() -> (
    None
):
    """Broadcasting over a scored-value axis folds into one multi-column solve."""
    generator = torch.Generator().manual_seed(2)
    L = torch.linalg.cholesky(torch.eye(3) + 0.2 * torch.ones(2, 3, 3))
    mean = torch.randn(2, 3, generator=generator)
    x = torch.randn(2, 5, 3, generator=generator)

    folded = compute_whitened_residuals(x, mean.unsqueeze(1), L.unsqueeze(1))
    expected = torch.linalg.solve_triangular(
        L.unsqueeze(1), (x - mean.unsqueeze(1)).unsqueeze(-1), upper=False
    ).squeeze(-1)
    torch.testing.assert_close(folded, expected)

    direct = compute_whitened_residuals(x[:, 0], mean, L)
    torch.testing.assert_close(direct, expected[:, 0])


def test_robust_cholesky_gives_up_on_a_matrix_no_jitter_can_repair() -> None:
    sigma = torch.full((2, 2), float("nan"), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="jitter"):
        robust_cholesky(sigma)
