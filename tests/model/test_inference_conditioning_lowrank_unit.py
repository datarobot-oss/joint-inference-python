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

"""Unit tests for conditioning the diagonal-plus-factor covariance.

The reference in every agreement test is the *dense* conditioning path applied
to the same covariance written out as a full Cholesky factor. That is the only
oracle that does not restate the Woodbury reduction under test: the two paths
reach the same conditional through different algebra, so agreement pins the
reduction rather than the code implementing it.

Two tolerances appear below, and the difference between them is the point.
The reduction itself is exact -- with stabilization disabled the two paths
agree to float64 noise, which
``test_lowrank_reduction_is_exact_without_stabilization`` asserts directly.
With the shared Cholesky policy in force they agree only to ``AGREEMENT_ATOL``,
because the two paths factorize different matrices and so add the policy's
floor in different bases: the dense path stabilizes ``Sigma_aa`` over the
observed variables, the low-rank path stabilizes ``I + U_a^T diag(d_a)^-1 U_a``
over the factor rank. That residual scales with the floor, not with the
conditioning problem, and an algebra error would be orders of magnitude larger.
"""

import pytest
import torch

import jointfm_inference.model.inference.conditioning as conditioning_module
from jointfm_inference.model.inference.conditioning import (
    MixtureCovariance,
    condition_student_t_mixture,
)

# Slack absorbing the stabilization floor that the two paths apply in different
# bases; measured worst case is ~3e-8 across the fixtures here.
AGREEMENT_ATOL = 1e-6
# Float64 noise once stabilization is removed and the comparison is pure algebra.
EXACT_ATOL = 1e-12


def _lowrank_mixture(
    n_batch: int = 2,
    n_horizons: int = 3,
    n_components: int = 4,
    n_vars: int = 6,
    rank: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, MixtureCovariance, torch.Tensor, torch.Tensor]:
    """Build a Student-t mixture whose scale matrix is strictly ``diag(d) + U U^T``.

    Float64 throughout: these tests compare two factorization routes against
    each other, and float32 noise on a Cholesky factor would swamp the margin
    that separates agreement from a genuine algebraic error. The degrees of
    freedom differ per component so a component mix-up in the posterior
    weights or the scale inflation cannot cancel out.

    Returns:
        ``(logits, means, covariance, df, values)``.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (n_batch, n_horizons, n_components)

    logits = torch.randn(*shape, generator=generator, dtype=torch.float64)
    means = torch.randn(*shape, n_vars, generator=generator, dtype=torch.float64)
    lowrank_diag = (
        torch.rand(*shape, n_vars, generator=generator, dtype=torch.float64) + 0.25
    )
    lowrank_U = torch.randn(
        *shape, n_vars, rank, generator=generator, dtype=torch.float64
    )
    df = 3.0 + 10.0 * torch.rand(*shape, generator=generator, dtype=torch.float64)
    values = torch.randn(
        n_batch, n_horizons, n_vars, generator=generator, dtype=torch.float64
    )

    covariance = MixtureCovariance(lowrank_diag=lowrank_diag, lowrank_U=lowrank_U)
    return logits, means, covariance, df, values


def _dense_scale(covariance: MixtureCovariance) -> torch.Tensor:
    """Write the diagonal-plus-factor scale matrix out in full."""
    lowrank_diag, lowrank_U = covariance.lowrank_parts()
    return torch.diag_embed(lowrank_diag) + torch.matmul(
        lowrank_U, lowrank_U.transpose(-1, -2)
    )


def _densified(covariance: MixtureCovariance) -> MixtureCovariance:
    """Write the same covariance out as a full Cholesky factor."""
    return MixtureCovariance(chol=torch.linalg.cholesky(_dense_scale(covariance)))


def _mask(n_vars: int, known: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros(n_vars, dtype=torch.bool)
    mask[list(known)] = True
    return mask


@pytest.mark.parametrize("known", [(0,), (0, 3), (2, 5), (1, 2, 4, 5)])
def test_lowrank_conditional_matches_the_dense_path(known: tuple[int, ...]) -> None:
    """Every returned quantity must survive the reduction, the scale inflation included.

    The inflation reads the observed block's Mahalanobis distance out of the
    reduction, so a Woodbury slip there shows up as a wrong conditional scale
    even when the Schur complement is right.
    """
    logits, means, covariance, df, values = _lowrank_mixture(seed=1)
    mask = _mask(means.shape[-1], known)

    lowrank = condition_student_t_mixture(logits, means, covariance, df, values, mask)
    dense = condition_student_t_mixture(
        logits, means, _densified(covariance), df, values, mask
    )

    for from_lowrank, from_dense in zip(lowrank, dense):
        torch.testing.assert_close(
            from_lowrank, from_dense, atol=AGREEMENT_ATOL, rtol=AGREEMENT_ATOL
        )


def test_lowrank_reduction_is_exact_without_stabilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate the reduction's exactness from the Cholesky policy's floor.

    Every other agreement test here runs with the shared stabilization in force
    and therefore cannot distinguish "the identity is exact" from "the identity
    is nearly right". Dropping the floor makes the comparison pure algebra, and
    the two paths must then agree to float64 noise.
    """
    monkeypatch.setattr(
        conditioning_module, "robust_cholesky", torch.linalg.cholesky, raising=True
    )

    logits, means, covariance, df, values = _lowrank_mixture(seed=6)
    mask = _mask(means.shape[-1], (0, 2))

    lowrank = condition_student_t_mixture(logits, means, covariance, df, values, mask)
    dense = condition_student_t_mixture(
        logits, means, _densified(covariance), df, values, mask
    )

    for from_lowrank, from_dense in zip(lowrank, dense):
        torch.testing.assert_close(
            from_lowrank, from_dense, atol=EXACT_ATOL, rtol=EXACT_ATOL
        )


def test_lowrank_conditional_keeps_the_diagonal_plus_factor_form_at_the_same_rank() -> (
    None
):
    """Pin the rank-preservation identity the reduction rests on.

    ``Sigma_bb|a = diag(d_b) + U_b M U_b^T`` with ``M = (I + U_a^T diag(d_a)^-1
    U_a)^-1``: the Schur complement is the same parameterization at the same
    rank, its factor rotated by ``M^(1/2)``. The returned Cholesky is dense and
    factorizes the *inflated* scale, so this reconstructs the scale from it,
    divides out the Student-t inflation computed from the dense observed block,
    and compares against the identity written out directly.
    """
    logits, means, covariance, df, values = _lowrank_mixture(seed=2)
    lowrank_diag, lowrank_U = covariance.lowrank_parts()
    rank = lowrank_U.shape[-1]
    known = [0, 4]
    unknown = [i for i in range(means.shape[-1]) if i not in known]
    mask = _mask(means.shape[-1], tuple(known))

    _, _, new_chol, _ = condition_student_t_mixture(
        logits, means, covariance, df, values, mask
    )
    attained = torch.matmul(new_chol, new_chol.transpose(-1, -2))

    sigma_aa = _dense_scale(covariance)[..., known, :][..., :, known]
    delta = (values.unsqueeze(-2) - means)[..., known].unsqueeze(-1)
    mahalanobis = (delta * torch.linalg.solve(sigma_aa, delta)).sum(dim=(-2, -1))
    inflation = (df + mahalanobis) / (df + len(known))

    diag_a = lowrank_diag[..., known]
    factor_a = lowrank_U[..., known, :]
    diag_b = lowrank_diag[..., unknown]
    factor_b = lowrank_U[..., unknown, :]

    gram = torch.matmul(factor_a.transpose(-1, -2), factor_a / diag_a.unsqueeze(-1))
    m_matrix = torch.linalg.inv(torch.eye(rank, dtype=torch.float64) + gram)
    expected = torch.diag_embed(diag_b) + torch.matmul(
        torch.matmul(factor_b, m_matrix), factor_b.transpose(-1, -2)
    )

    torch.testing.assert_close(
        attained / inflation[..., None, None],
        expected,
        atol=AGREEMENT_ATOL,
        rtol=AGREEMENT_ATOL,
    )

    rotated_factor = torch.matmul(factor_b, torch.linalg.cholesky(m_matrix))
    assert int(torch.linalg.matrix_rank(rotated_factor).max()) == rank


def test_lowrank_conditional_never_materializes_the_joint_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank far below the variable count must not allocate a ``D x D`` tensor.

    The reduction's whole point is that the observed-block solve happens at the
    factor rank. Guarding it by allocation rather than by timing keeps the test
    fast and deterministic: ``torch.diag_embed`` at the *joint* width is the
    only way a dense joint covariance can appear in this path, so a width-``D``
    call is the signature of a path that densifies before conditioning.
    """
    n_vars = 32
    n_known = 3
    logits, means, covariance, df, values = _lowrank_mixture(
        n_batch=1, n_horizons=1, n_components=2, n_vars=n_vars, rank=2, seed=3
    )
    mask = _mask(n_vars, tuple(range(n_known)))
    widths: list[int] = []
    original = torch.diag_embed

    def _recording_diag_embed(
        input: torch.Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1
    ) -> torch.Tensor:
        widths.append(int(input.shape[-1]))
        return original(input, offset, dim1, dim2)

    monkeypatch.setattr(torch, "diag_embed", _recording_diag_embed)
    condition_student_t_mixture(logits, means, covariance, df, values, mask)

    assert widths == [n_vars - n_known], (
        "only the unknown block may be embedded as a dense diagonal; "
        f"saw widths {widths} for a joint width of {n_vars}"
    )


def test_lowrank_all_known_mask_returns_posterior_weights_only() -> None:
    logits, means, covariance, df, values = _lowrank_mixture(seed=4)
    mask = torch.ones(means.shape[-1], dtype=torch.bool)

    new_logits, new_means, new_chol, new_df = condition_student_t_mixture(
        logits, means, covariance, df, values, mask
    )
    dense_logits, _, _, _ = condition_student_t_mixture(
        logits, means, _densified(covariance), df, values, mask
    )

    assert tuple(new_means.shape) == tuple(logits.shape) + (0,)
    assert tuple(new_chol.shape) == tuple(logits.shape) + (0, 0)
    torch.testing.assert_close(new_df, df + means.shape[-1])
    torch.testing.assert_close(
        new_logits, dense_logits, atol=AGREEMENT_ATOL, rtol=AGREEMENT_ATOL
    )


def test_lowrank_all_unknown_mask_densifies_the_unchanged_covariance() -> None:
    """Conditioning on nothing is a no-op, but the return contract is a Cholesky."""
    logits, means, covariance, df, values = _lowrank_mixture(seed=5)
    mask = torch.zeros(means.shape[-1], dtype=torch.bool)

    new_logits, new_means, new_chol, new_df = condition_student_t_mixture(
        logits, means, covariance, df, values, mask
    )

    assert new_logits is logits
    assert new_means is means
    assert new_df is df
    attained = torch.matmul(new_chol, new_chol.transpose(-1, -2))
    torch.testing.assert_close(
        attained, _dense_scale(covariance), atol=AGREEMENT_ATOL, rtol=AGREEMENT_ATOL
    )


def test_mixture_covariance_rejects_ambiguous_parameterizations() -> None:
    chol = torch.eye(2).expand(1, 1, 2, 2).clone()
    diag = torch.ones(1, 1, 2)
    factor = torch.zeros(1, 1, 2, 1)

    with pytest.raises(ValueError, match="exactly one parameterization"):
        MixtureCovariance()
    with pytest.raises(ValueError, match="exactly one parameterization"):
        MixtureCovariance(chol=chol, lowrank_diag=diag, lowrank_U=factor)
    with pytest.raises(ValueError, match="together"):
        MixtureCovariance(lowrank_diag=diag)


def test_mixture_covariance_from_params_rejects_a_mapping_without_a_covariance() -> (
    None
):
    with pytest.raises(KeyError, match="neither"):
        MixtureCovariance.from_params({"logits": torch.zeros(1, 1)})
