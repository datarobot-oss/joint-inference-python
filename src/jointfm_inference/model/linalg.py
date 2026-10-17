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

"""Linear algebra utilities for stable covariance and distance computations."""

import logging
import math
from typing import Optional
import torch
from torch import Tensor


logger = logging.getLogger(__name__)

# The retry jitter grows geometrically and stops at the size of the unit
# correlation diagonal, beyond which no matrix is repaired, only replaced.
JITTER_GROWTH = 1.1
MAX_RELATIVE_JITTER = 1.0


def _equilibration_scale(sigma: Tensor) -> Tensor:
    """Return the per-variable scale that maps ``sigma`` to its correlation form.

    A variable with no positive variance has no scale of its own to measure a
    jitter against, so it borrows the largest variance in its own matrix, and a
    matrix with no positive variance at all borrows one. Both fallbacks keep the
    correlation matrix finite where a division by the variance would produce
    ``nan``, and the second reproduces the unequilibrated result for an
    all-zero covariance, whose factor is the stabilization jitter and nothing
    else however it is reached.
    """
    diagonal = torch.diagonal(sigma, dim1=-2, dim2=-1)
    largest = diagonal.amax(dim=-1, keepdim=True)
    reference = torch.where(largest > 0, largest, torch.ones_like(largest))
    return torch.where(diagonal > 0, diagonal, reference.expand_as(diagonal)).sqrt()


def robust_cholesky(
    sigma: Tensor, initial_jitter: float = 1e-9, retry_start: float = 1e-8
) -> Tensor:
    """
    Compute Cholesky decomposition (L @ L.T = sigma) with robust jitter injection.

    The matrix is equilibrated to its correlation form before any jitter is
    added, stabilized there, and scaled back, so every jitter below is a
    *fraction of each variable's own variance* rather than an absolute
    quantity. That is what makes the factor of a rescaled covariance equal the
    rescaled factor of the original: ``cholesky(S sigma S) == S cholesky(sigma)``
    for any positive diagonal ``S``. An absolute floor instead inflates a
    variable whose variance is small in the caller's units by a multiple of its
    true value, and the error it leaves grows with the Mahalanobis distance of
    the point being scored, so it is worst for exactly the far-tail values that
    a density is reported to distinguish. Equilibration is also the standard
    remedy for a badly scaled positive definite system, so the factorization it
    hands to LAPACK is the better conditioned one.

    Tries to compute the factorization at ``initial_jitter``; if that fails,
    iteratively adds more jitter until it succeeds. Performs computation in
    float32 for stability if input is lower precision.

    Args:
        sigma: Symmetric positive definite matrix (..., D, D).
        initial_jitter: Jitter added to the correlation diagonal before the
            first attempt, therefore a fraction of each variance.
        retry_start: Starting jitter for the retry loop, on the same relative
            scale. Grown on every attempt.

    Returns:
        Lower triangular Cholesky factor L.

    Raises:
        RuntimeError: when the factorization fails at every jitter up to
            ``MAX_RELATIVE_JITTER``, as for a matrix holding NaN.
    """
    # Ensure float32 for stability/support
    orig_dtype = sigma.dtype
    if sigma.dtype in (torch.bfloat16, torch.float16):
        sigma_run = sigma.to(dtype=torch.float32)
    else:
        sigma_run = sigma

    scale = _equilibration_scale(sigma_run)
    # A fresh tensor, so the jitter below never reaches the caller's matrix.
    correlation = sigma_run / (scale.unsqueeze(-1) * scale.unsqueeze(-2))
    rows = scale.unsqueeze(-1)

    D = sigma.shape[-1]
    diag_ind = torch.arange(D, device=sigma.device)

    # 1. Initial stabilization
    if initial_jitter > 0:
        correlation[..., diag_ind, diag_ind] += initial_jitter

    try:
        L = torch.linalg.cholesky(correlation)
        return (rows * L).to(dtype=orig_dtype)
    except RuntimeError:
        # 2. Retry loop
        jitter = retry_start

        while jitter <= MAX_RELATIVE_JITTER:
            try:
                correlation[..., diag_ind, diag_ind] += jitter
                chol = torch.linalg.cholesky(correlation)
                logger.debug(
                    f"Could not compute Cholesky without jitter ({jitter:.1e})"
                )
                return (rows * chol).to(dtype=orig_dtype)
            except RuntimeError:
                jitter *= JITTER_GROWTH
        raise RuntimeError(
            "Cholesky factorization failed even at a relative jitter of "
            f"{MAX_RELATIVE_JITTER}; the matrix is not a covariance"
        )


def cholesky_from_lowrank(
    lowrank_diag: Tensor, lowrank_U: Optional[Tensor], initial_jitter: float = 1e-8
) -> Tensor:
    """
    Construct Sigma = diag(d) + U U^T and compute robust Cholesky.

    Args:
        lowrank_diag: (..., D)
        lowrank_U: (..., D, R) or None
        initial_jitter: Initial stabilization handed to the robust Cholesky
            retry loop, relative to each variable's own variance.

    Returns:
        Lower triangular Cholesky factor L (..., D, D).
    """
    D_mat = torch.diag_embed(lowrank_diag)
    # sigma = D + U @ U.T
    if lowrank_U is None:
        sigma = D_mat
    else:
        sigma = D_mat + torch.matmul(lowrank_U, lowrank_U.transpose(-1, -2))
    return robust_cholesky(sigma, initial_jitter=initial_jitter)


def apply_selected_cholesky(selector: Tensor, chol: Tensor, noise: Tensor) -> Tensor:
    """Apply per-draw selected Cholesky factors without per-draw matrices.

    Computes ``out[b, n] = chol[b, k(n)] @ noise[b, n]`` for a one-hot
    ``selector`` ``(B, N, K)``, component factors ``chol`` ``(B, K, D, D)``
    and ``noise`` ``(B, N, D)``. Contracting ``(chol, noise)`` first keeps the
    intermediate at ``(B, K, N, D)``, scaling with K instead of materializing a
    selected ``(B, N, D, D)`` matrix copy per draw.
    """
    candidates = torch.einsum("bkde,bne->bknd", chol, noise)
    return torch.einsum("bnk,bknd->bnd", selector, candidates)


def compute_whitened_residuals(x: Tensor, mean: Tensor, L: Tensor) -> Tensor:
    """Compute whitened residuals ``z = L^{-1} (x - mean)``.

    Batch dims where ``L`` is singleton but the residual broadcasts (e.g. the
    scored-value axis) are folded into right-hand-side columns of one
    multi-RHS triangular solve. The batched CUDA solver would otherwise
    materialize the broadcast ``L`` per value, inflating memory by a factor
    of D.
    """
    # Align shapes
    if x.ndim == mean.ndim + 1:
        mean = mean.unsqueeze(-2)
        L = L.unsqueeze(-3)

    diff = x - mean

    # Ensure L is float32 for solve_triangular on CUDA
    orig_dtype = x.dtype
    if L.dtype in (torch.bfloat16, torch.float16):
        L_solve = L.to(dtype=torch.float32)
        diff_solve = diff.to(dtype=torch.float32)
    else:
        L_solve = L
        diff_solve = diff

    batch_shape = torch.broadcast_shapes(diff_solve.shape[:-1], L_solve.shape[:-2])
    n_batch = len(batch_shape)
    L_batch = (1,) * (n_batch - (L_solve.ndim - 2)) + tuple(L_solve.shape[:-2])
    fold = [i for i in range(n_batch) if L_batch[i] == 1 and batch_shape[i] > 1]

    if not fold:
        # Solve L @ z = diff
        diff_col = diff_solve.unsqueeze(-1)
        z = torch.linalg.solve_triangular(L_solve, diff_col, upper=False)
        return z.squeeze(-1).to(dtype=orig_dtype)

    keep = [i for i in range(n_batch) if i not in fold]
    D = diff_solve.shape[-1]
    keep_shape = [batch_shape[i] for i in keep]
    fold_shape = [batch_shape[i] for i in fold]
    n_columns = math.prod(fold_shape)

    rhs = (
        diff_solve.expand(*batch_shape, D)
        .permute(*keep, *fold, n_batch)
        .reshape(*keep_shape, n_columns, D)
        .transpose(-1, -2)
    )
    L_kept = L_solve.reshape(*[L_batch[i] for i in keep], D, D)
    z_cols = torch.linalg.solve_triangular(L_kept, rhs, upper=False)

    z = z_cols.transpose(-1, -2).reshape(*keep_shape, *fold_shape, D)
    inverse_order = [0] * n_batch
    for new_position, original_dim in enumerate(keep + fold):
        inverse_order[original_dim] = new_position
    return z.permute(*inverse_order, n_batch).to(dtype=orig_dtype)
