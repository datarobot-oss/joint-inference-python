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

"""Unit tests for the closed-form Student-t mixture conditional.

Every fixture writes the emitted parameter mapping (``logits``, ``means``,
``chol``, ``df``) by hand, so the conditional is checked without a head, a
network, or a checkpoint.
"""

import math

import pytest
import torch

from jointfm_inference.model.inference.conditioning import (
    MixtureCovariance,
    condition_student_t_mixture,
    conditioned_log_density,
)
from tests.support.inference import student_t_log_density, student_t_pair_mixture


def _normal_log_density(
    value: torch.Tensor, location: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    """Evaluate a univariate normal log density from its defining formula."""
    return -0.5 * (
        math.log(2 * math.pi) + variance.log() + (value - location) ** 2 / variance
    )


@pytest.mark.parametrize(
    "known", [(True, False, False), (True, False, True), (False, True, True)]
)
def test_student_t_conditional_matches_the_ding_closed_form(
    known: tuple[bool, ...],
) -> None:
    """Check one component against the closed form written out by hand.

    The expectation restates Ding (2016) rather than the implementation: the
    location is the Gaussian conditional mean (the Schur-complement shift), the
    degrees of freedom rise by the number of known variables, and the scale
    matrix is the Schur complement times ``(nu + d_a) / (nu + p)``. Deriving
    the expectation from the returned parameters instead would make the
    assertion follow whatever the conditioning path happens to compute.
    """
    dimension = 3
    covariance = torch.tensor(
        [[2.0, 0.6, -0.4], [0.6, 1.5, 0.3], [-0.4, 0.3, 1.0]], dtype=torch.float64
    )
    location = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    df = torch.tensor(7.0, dtype=torch.float64)
    values = torch.tensor([1.75, 0.25, -0.5], dtype=torch.float64)
    mask = torch.tensor(known)
    n_known = int(mask.sum())

    _, new_means, new_chol, new_df = condition_student_t_mixture(
        logits=torch.zeros(1, 1, 1, dtype=torch.float64),
        means=location.reshape(1, 1, 1, dimension),
        covariance=MixtureCovariance(
            chol=torch.linalg.cholesky(covariance).reshape(
                1, 1, 1, dimension, dimension
            )
        ),
        df=df.reshape(1, 1, 1),
        values=values.reshape(1, 1, dimension),
        mask=mask,
    )

    sigma_aa = covariance[mask][:, mask]
    sigma_ab = covariance[mask][:, ~mask]
    sigma_ba = covariance[~mask][:, mask]
    sigma_bb = covariance[~mask][:, ~mask]
    delta = (values[mask] - location[mask]).unsqueeze(-1)

    gaussian_conditional_mean = location[~mask] + (
        sigma_ba @ torch.linalg.solve(sigma_aa, delta)
    ).squeeze(-1)
    schur = sigma_bb - sigma_ba @ torch.linalg.solve(sigma_aa, sigma_ab)
    mahalanobis = (
        delta.squeeze(-1) * torch.linalg.solve(sigma_aa, delta).squeeze(-1)
    ).sum()
    inflation = (df + mahalanobis) / (df + n_known)
    expected_chol = torch.linalg.cholesky(inflation * schur)

    torch.testing.assert_close(new_means[0, 0, 0], gaussian_conditional_mean)
    torch.testing.assert_close(new_chol[0, 0, 0], expected_chol)
    torch.testing.assert_close(new_df[0, 0, 0], df + n_known)


def test_student_t_posterior_weights_use_the_marginal_t_density() -> None:
    """The posterior must reweight by the *t* marginal, not the Gaussian one.

    A multivariate-t marginal keeps the joint's degrees of freedom with the
    corresponding scale submatrix, so the observed block is scored under
    ``t(nu_k)`` and not under a normal of the same scale. At a far-tail
    conditioning value the two disagree sharply, because the t density decays
    polynomially where the Gaussian decays in the exponent. The normalizer of
    those weights is the mixture's marginal density of the known value.
    """
    params = student_t_pair_mixture()
    mask = torch.tensor([True, False])
    conditioning_value = torch.tensor(3.0, dtype=torch.float64)
    values = torch.tensor([[[3.0, 0.0]]], dtype=torch.float64)

    new_logits, _, _, _ = condition_student_t_mixture(
        params["logits"],
        params["means"],
        MixtureCovariance(chol=params["chol"]),
        params["df"],
        values,
        mask,
    )

    prior_log_weights = torch.log_softmax(params["logits"][0, 0], dim=-1)
    observed_locations = params["means"][0, 0, :, 0]
    observed_scale = params["chol"][0, 0, :, 0, 0] ** 2
    expected_log_weights = prior_log_weights + student_t_log_density(
        conditioning_value, observed_locations, observed_scale, params["df"][0, 0]
    )
    torch.testing.assert_close(new_logits[0, 0], expected_log_weights)
    torch.testing.assert_close(
        conditioned_log_density(new_logits)[0, 0],
        torch.logsumexp(expected_log_weights, dim=-1),
    )

    gaussian_log_weights = prior_log_weights + _normal_log_density(
        conditioning_value, observed_locations, observed_scale
    )
    assert not torch.allclose(
        torch.softmax(new_logits[0, 0], dim=-1),
        torch.softmax(gaussian_log_weights, dim=-1),
        atol=1e-3,
    )


def test_student_t_conditional_scale_grows_with_a_surprising_value() -> None:
    """The conditional scale carries the observed block's quadratic form.

    The Schur complement does not see the conditioning value at all; the
    Student-t inflation does, so the conditional widens by exactly
    ``(nu + d_far) / (nu + d_center)`` between a central and a far-tail value
    (Azzalini & Capitanio, 2003, for the elliptical statement).
    """
    dimension = 2
    covariance = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
    chol = torch.linalg.cholesky(covariance).reshape(1, 1, 1, dimension, dimension)
    logits = torch.zeros(1, 1, 1, dtype=torch.float64)
    means = torch.zeros(1, 1, 1, dimension, dtype=torch.float64)
    df = torch.tensor(9.0, dtype=torch.float64).reshape(1, 1, 1)
    mask = torch.tensor([True, False])

    center = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
    far_tail = torch.tensor([[[4.0, 0.0]]], dtype=torch.float64)

    _, _, t_center, _ = condition_student_t_mixture(
        logits, means, MixtureCovariance(chol=chol), df, center, mask
    )
    _, _, t_far, _ = condition_student_t_mixture(
        logits, means, MixtureCovariance(chol=chol), df, far_tail, mask
    )

    mahalanobis_far = far_tail[0, 0, 0] ** 2 / covariance[0, 0]
    expected_ratio = (df[0, 0, 0] + mahalanobis_far) / df[0, 0, 0]
    attained_ratio = (t_far[0, 0, 0, 0, 0] / t_center[0, 0, 0, 0, 0]) ** 2

    torch.testing.assert_close(attained_ratio, expected_ratio)


def test_student_t_all_unknown_mask_returns_inputs_unchanged() -> None:
    """Conditioning on nothing is the identity, down to the tensor objects."""
    params = student_t_pair_mixture()

    unchanged = condition_student_t_mixture(
        params["logits"],
        params["means"],
        MixtureCovariance(chol=params["chol"]),
        params["df"],
        torch.tensor([[[3.0, -1.0]]], dtype=torch.float64),
        torch.zeros(2, dtype=torch.bool),
    )

    assert unchanged[0] is params["logits"]
    assert unchanged[1] is params["means"]
    assert unchanged[2] is params["chol"]
    assert unchanged[3] is params["df"]


def test_student_t_all_known_mask_reweights_components_only() -> None:
    """With nothing left to read out, the weights and the raised df still carry information.

    Unit-scale identity components make the bivariate t density of the known
    pair a closed form: ``Gamma((nu + 2) / 2) / (Gamma(nu / 2) nu pi)
    (1 + |x - mu|^2 / nu)^(-(nu + 2) / 2)``.
    """
    dtype = torch.float64
    logits = torch.tensor([[[0.2, -0.4]]], dtype=dtype)
    means = torch.tensor([[[[0.0, 1.0], [2.0, -1.0]]]], dtype=dtype)
    chol = torch.eye(2, dtype=dtype).expand(1, 1, 2, 2, 2).clone()
    df = torch.tensor([[[4.0, 10.0]]], dtype=dtype)
    values = torch.tensor([[[0.25, 0.75]]], dtype=dtype)

    new_logits, new_means, new_chol, new_df = condition_student_t_mixture(
        logits,
        means,
        MixtureCovariance(chol=chol),
        df,
        values,
        torch.ones(2, dtype=torch.bool),
    )

    assert tuple(new_means.shape) == (1, 1, 2, 0)
    assert tuple(new_chol.shape) == (1, 1, 2, 0, 0)
    torch.testing.assert_close(new_df, df + 2.0)

    squared_distance = ((values.unsqueeze(-2) - means) ** 2).sum(dim=-1)
    log_density = (
        torch.lgamma((df + 2.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - torch.log(df * math.pi)
        - 0.5 * (df + 2.0) * torch.log1p(squared_distance / df)
    )
    torch.testing.assert_close(
        new_logits, torch.log_softmax(logits, dim=-1) + log_density
    )


def test_student_t_conditional_broadcasts_a_batched_mask() -> None:
    """A mask repeated across the leading axes must match the vector form.

    The degrees of freedom are flattened alongside the other parameters, so
    this also guards the ``df`` reshape against a batch axis being folded in
    the wrong order.
    """
    params = student_t_pair_mixture()
    batch, horizons, components = 2, 3, 2
    logits = params["logits"].expand(batch, horizons, components).clone()
    means = params["means"].expand(batch, horizons, components, 2).clone()
    chol = params["chol"].expand(batch, horizons, components, 2, 2).clone()
    df = torch.linspace(4.0, 9.0, batch * horizons * components, dtype=torch.float64)
    df = df.reshape(batch, horizons, components)
    values = torch.randn(
        batch,
        horizons,
        2,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(0),
    )

    vector_mask = torch.tensor([True, False])
    batched_mask = vector_mask.reshape(1, 1, 2).expand(batch, horizons, 2).clone()

    vector_out = condition_student_t_mixture(
        logits, means, MixtureCovariance(chol=chol), df, values, vector_mask
    )
    batched_out = condition_student_t_mixture(
        logits, means, MixtureCovariance(chol=chol), df, values, batched_mask
    )
    for from_vector, from_batched in zip(vector_out, batched_out):
        torch.testing.assert_close(from_vector, from_batched)
    assert tuple(vector_out[1].shape) == (batch, horizons, components, 1)
    assert tuple(vector_out[2].shape) == (batch, horizons, components, 1, 1)
    torch.testing.assert_close(vector_out[3], df + 1.0)


def test_student_t_conditional_does_not_inflate_a_tight_conditional_scale() -> None:
    """A near-deterministic pair must keep its small conditional scale.

    The Schur complement of a strongly correlated pair is orders of magnitude
    below unit scale, so a stabilization floor applied to the returned factor
    shows up as a multiplicative inflation rather than a rounding-level
    correction. Pinning the known column at its location makes the Student-t
    inflation exactly ``nu / (nu + 1)``. The bound is deliberately loose: it
    separates "conditional scale recovered" from "conditional scale swamped by
    the floor" without pinning the shared Cholesky policy's exact jitter.
    """
    rho = 0.9999999
    covariance = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64)
    df = 5.0
    _, _, new_chol, _ = condition_student_t_mixture(
        torch.zeros(1, 1, 1, dtype=torch.float64),
        torch.zeros(1, 1, 1, 2, dtype=torch.float64),
        MixtureCovariance(
            chol=torch.linalg.cholesky(covariance).reshape(1, 1, 1, 2, 2)
        ),
        torch.full((1, 1, 1), df, dtype=torch.float64),
        torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.tensor([True, False]),
    )

    exact_conditional_scale = (1.0 - rho * rho) * df / (df + 1.0)
    attained = float(new_chol[0, 0, 0, 0, 0] ** 2)

    assert attained == pytest.approx(exact_conditional_scale, rel=5e-2)


def test_student_t_conditional_recovers_a_singular_known_block() -> None:
    """Condition on perfectly correlated known columns without raising.

    Two known columns that move together make the known block exactly
    singular, which a request pinning a duplicated series produces. At float32
    and a large scale, a fixed stabilization floor is lost to rounding and the
    factorization fails outright, so recovery has to come from the escalating
    retry the shared Cholesky policy owns.
    """
    scale = 1e3
    chol = torch.tensor(
        [[scale, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    ).reshape(1, 1, 3, 3)

    conditioned = condition_student_t_mixture(
        torch.zeros(1, 1, dtype=torch.float32),
        torch.zeros(1, 1, 3, dtype=torch.float32),
        MixtureCovariance(chol=chol),
        torch.full((1, 1), 6.0, dtype=torch.float32),
        torch.tensor([[scale, scale, 0.0]], dtype=torch.float32),
        torch.tensor([True, True, False]),
    )

    for tensor in conditioned:
        assert torch.isfinite(tensor).all()


def test_student_t_conditioning_rejects_non_positive_degrees_of_freedom() -> None:
    """A component with non-positive ``df`` has no marginal t density to score."""
    params = student_t_pair_mixture()
    df = params["df"].clone()
    df[0, 0, 1] = 0.0

    with pytest.raises(ValueError, match="strictly positive degrees of freedom"):
        condition_student_t_mixture(
            params["logits"],
            params["means"],
            MixtureCovariance(chol=params["chol"]),
            df,
            torch.zeros(1, 1, 2, dtype=torch.float64),
            torch.tensor([True, False]),
        )


def test_student_t_conditioning_rejects_a_mismatched_degrees_of_freedom_shape() -> None:
    """``df`` must carry one value per component on the logits' leading axes.

    The flattening folds ``df`` alongside the other parameters, so a tensor of
    the wrong shape would reshape silently into a valid but meaningless
    component assignment instead of failing.
    """
    params = student_t_pair_mixture()

    with pytest.raises(ValueError, match="df shaped like logits"):
        condition_student_t_mixture(
            params["logits"],
            params["means"],
            MixtureCovariance(chol=params["chol"]),
            params["df"].reshape(1, 2, 1),
            torch.zeros(1, 1, 2, dtype=torch.float64),
            torch.tensor([True, False]),
        )
