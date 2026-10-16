"""Tests for the per-column covariance factor readout and the low-rank emission.

The readout's contract is that column ``i``'s factor rows are a function of
column ``i``'s own token plus the shared pooled context: identical tokens give
identical rows (a perfectly correlated pair), and permuting columns permutes
rows.
"""

import pytest
import torch
import torch.nn.functional as F

from jointfm_inference.model.jointfm_1.head import StudentTHead
from jointfm_inference.model.jointfm_1.readouts import (
    MAX_SOFTPLUS,
    ColumnFactorReadout,
    emit_lowrank_covariance,
)

from tests.support.inference import student_t_head

B, H, D_CAP, K, RANK, D_MODEL, CONTEXT = 2, 3, 6, 2, 2, 16, 8
# Drives the softplus of the raw diagonal to effectively zero.
NEGLIGIBLE_RAW_DIAGONAL = -30.0


def _head(*, min_scale: float = 1e-4) -> StudentTHead:
    """Build a low-rank Student-t head reading per-column tokens."""
    return student_t_head(
        D=D_CAP, K=K, d_model=D_MODEL, covariance_rank=RANK, min_scale=min_scale
    )


def _readout() -> ColumnFactorReadout:
    """Build a bare readout."""
    torch.manual_seed(0)
    return ColumnFactorReadout(
        column_dim=D_MODEL,
        context_dim=CONTEXT,
        hidden=CONTEXT,
        n_components=K,
        rank=RANK,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a pooled context and per-column tokens from a fixed seed."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(B, H, D_MODEL, generator=generator)
    tokens = torch.randn(B, H, D_CAP, D_MODEL, generator=generator)
    return x, tokens


def _implied_correlation(params: dict, i: int, j: int) -> torch.Tensor:
    """Component-0 correlation between columns ``i`` and ``j`` implied by the head."""
    diag = params["lowrank_diag"][..., 0, :]
    U = params["lowrank_U"][..., 0, :, :]
    sigma = torch.diag_embed(diag) + U @ U.transpose(-1, -2)
    return sigma[..., i, j] / torch.sqrt(sigma[..., i, i] * sigma[..., j, j])


def test_identical_column_tokens_give_identical_factor_rows() -> None:
    """Two columns with the same token co-move perfectly in the factor channel."""
    head = _head(min_scale=1e-8)
    with torch.no_grad():
        # A negligible diagonal makes the implied correlation read the factor alone.
        head.out_diag.bias.fill_(NEGLIGIBLE_RAW_DIAGONAL)
        head.out_diag.weight.zero_()
    x, tokens = _inputs()
    tokens[..., 1, :] = tokens[..., 0, :]
    with torch.no_grad():
        params = head(x, column_tokens=tokens)

    U = params["lowrank_U"]
    torch.testing.assert_close(U[..., 0, :], U[..., 1, :])
    torch.testing.assert_close(
        _implied_correlation(params, 0, 1), torch.ones(B, H), atol=1e-4, rtol=0
    )
    assert (_implied_correlation(params, 0, 2).abs() < 0.999).all()


def test_factor_rows_are_permutation_equivariant() -> None:
    """Permuting the columns permutes the factor rows and nothing else."""
    head = _head()
    x, tokens = _inputs()
    perm = torch.tensor([3, 0, 5, 1, 4, 2])

    with torch.no_grad():
        U = head(x, column_tokens=tokens)["lowrank_U"]
        U_perm = head(x, column_tokens=tokens[:, :, perm])["lowrank_U"]
    torch.testing.assert_close(U_perm, U[..., perm, :])


def test_head_refuses_more_columns_than_it_was_built_for() -> None:
    head = _head()
    with pytest.raises(ValueError, match="exceeding the head capacity"):
        head(
            torch.randn(B, H, D_MODEL),
            column_tokens=torch.randn(B, H, D_CAP + 1, D_MODEL),
        )


def test_readout_shape_contract() -> None:
    readout = _readout()
    tokens = torch.randn(B, H, 4, D_MODEL)
    context = torch.randn(B, H, CONTEXT)

    assert readout.output.bias is None
    assert readout(tokens, context).shape == (B, H, K, 4, RANK)
    with pytest.raises(ValueError, match="column_tokens must have shape"):
        readout(tokens[0], context)
    with pytest.raises(ValueError, match="column_tokens must have shape"):
        readout(torch.randn(B, H, 4, D_MODEL + 1), context)
    with pytest.raises(ValueError, match="context must have shape"):
        readout(tokens, torch.randn(B, H, CONTEXT + 1))
    with pytest.raises(ValueError, match="context must have shape"):
        readout(tokens, context[:1])


def test_emitted_diagonal_is_a_capped_softplus_plus_the_floor() -> None:
    readout = _readout()
    raw_diag = torch.tensor([-50.0, 0.0, 3.0, 1e6]).expand(B, H, K, 4)
    min_scale = 0.05

    out = emit_lowrank_covariance(
        raw_diag=raw_diag,
        column_factor=readout,
        hidden=torch.randn(B, H, CONTEXT),
        column_tokens=torch.randn(B, H, 4, D_MODEL),
        capacity=4,
        min_scale=min_scale,
        dtype=torch.float64,
    )

    expected = torch.clamp(F.softplus(raw_diag), max=MAX_SOFTPLUS) + min_scale
    torch.testing.assert_close(out["lowrank_diag"], expected.double())
    assert out["lowrank_diag"].dtype == torch.float64
    assert float(out["lowrank_diag"].max()) == pytest.approx(MAX_SOFTPLUS + min_scale)
    assert out["lowrank_U"].shape == (B, H, K, 4, RANK)


def test_emission_refuses_a_misshaped_raw_diagonal() -> None:
    with pytest.raises(ValueError, match="raw_diag must have shape"):
        emit_lowrank_covariance(
            raw_diag=torch.zeros(B, H, K, 3),
            column_factor=_readout(),
            hidden=torch.randn(B, H, CONTEXT),
            column_tokens=torch.randn(B, H, 3, D_MODEL),
            capacity=4,
            min_scale=1e-4,
            dtype=torch.float32,
        )
