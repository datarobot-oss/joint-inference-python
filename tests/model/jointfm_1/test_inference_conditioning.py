"""Student-t conditioning through ``PredictorAdapter`` and the Student-t head.

These are the tests that need the head itself: the Monte Carlo oracle draws
from the unconditional head's sampler, and the density identity scores both
the joint and the reduced conditional through the head's own scoring path. The
closed form they exercise is unit-tested without any head in
``test_inference_conditioning_unit.py``.
"""

import pytest
import torch

from jointfm_inference.model.inference import PredictorAdapter
from tests.support.inference import student_t_head, student_t_pair_mixture

# Half-width of the acceptance band and draw count of the Monte Carlo oracle,
# paired so both settings retain roughly the same number of draws: the band
# controls the estimator's bias and the draw count its noise, and only a
# matched pair separates the two.
BAND_SETTINGS = ((0.20, 500_000), (0.05, 2_000_000))
MONTE_CARLO_SEED = 3
# Conditioning value of the oracle, about three marginal scales out on the
# narrow component, where the two components' scale inflations differ most.
CONDITIONING_VALUE = 3.0


def _reduced_mixture_moments(
    params: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and standard deviation of a one-variable t mixture.

    A ``t(nu, mu, s)`` component has variance ``nu / (nu - 2) * s``, so the
    caller must keep ``df`` above two for this to be defined.
    """
    weights = torch.softmax(params["logits"].flatten(), dim=-1)
    locations = params["means"].flatten()
    scales = params["chol"].flatten() ** 2
    df = params["df"].flatten()

    variances = df / (df - 2.0) * scales
    mean = (weights * locations).sum()
    second_moment = (weights * (variances + locations**2)).sum()
    return mean, (second_moment - mean**2).sqrt()


def test_student_t_conditional_reproduces_the_joint_density_ratio() -> None:
    """Pin the whole conditional against ``p(x_b | x_a) = p(x_a, x_b) / p(x_a)``.

    This is the one oracle that checks every returned quantity at once. The
    posterior weights, the location shift, the raised degrees of freedom and
    the scale inflation all enter the reduced density, so any one of them being
    wrong breaks the identity; and the normalizer of the returned unnormalized
    log weights is by construction ``log p(x_a)``, so the two sides come from
    genuinely different code paths — the head's scoring path on one side, the
    conditioning algebra on the other.
    """
    generator = torch.Generator().manual_seed(0)
    dimension, components = 3, 2
    factor = torch.randn(
        components, dimension, dimension, dtype=torch.float64, generator=generator
    )
    covariance = factor @ factor.transpose(-1, -2) + torch.eye(
        dimension, dtype=torch.float64
    )
    params = {
        "logits": torch.tensor([[[0.4, -0.9]]], dtype=torch.float64),
        "means": torch.randn(
            1, 1, components, dimension, dtype=torch.float64, generator=generator
        ),
        "chol": torch.linalg.cholesky(covariance).reshape(
            1, 1, components, dimension, dimension
        ),
        "df": torch.tensor([[[5.0, 11.0]]], dtype=torch.float64),
    }
    adapter = PredictorAdapter(student_t_head(D=dimension, K=components), params)

    mask = torch.tensor([True, False, True])
    values = torch.tensor([[[1.3, -0.7, 0.4]]], dtype=torch.float64)
    conditioned = adapter.condition(values, mask)

    log_joint = adapter.log_prob(values)
    log_observed = torch.logsumexp(conditioned.params["logits"], dim=-1)
    log_conditional = conditioned.log_prob(values[..., ~mask])

    # The head scores in float32 regardless of the parameter dtype.
    torch.testing.assert_close(
        log_conditional, log_joint - log_observed, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(("half_width", "n_draws"), BAND_SETTINGS)
def test_student_t_conditional_matches_a_monte_carlo_band_oracle(
    half_width: float, n_draws: int
) -> None:
    """Converge the analytic conditional onto draws kept inside a narrow band.

    Drawing from the *unconditional* head and keeping only the draws whose
    observed column landed within ``half_width`` of the conditioning value
    approximates ``p(x_b | x_a)`` through a code path that shares nothing with
    the conditioning algebra. Shrinking the band while raising the draw count
    holds the retained count roughly fixed, so a drift between the two settings
    would be band-width bias rather than Monte Carlo noise.

    The spread assertion is what carries the scale inflation: at this
    conditioning value the two components inflate by 2.14 and 1.18, and
    dropping the factor entirely moves the mixture standard deviation from
    1.599 to 1.500, which is far outside the tolerance below.
    """
    head = student_t_head(D=2, K=2)
    params = student_t_pair_mixture()
    mask = torch.tensor([True, False])
    values = torch.tensor([[[CONDITIONING_VALUE, 0.0]]], dtype=torch.float64)

    conditioned = PredictorAdapter(head, params).condition(values, mask)
    analytic_mean, analytic_std = _reduced_mixture_moments(conditioned.params)

    rng = torch.Generator().manual_seed(MONTE_CARLO_SEED)
    draws = head.sample(params, num_samples=n_draws, rng=rng).reshape(-1, 2)
    inside_band = (draws[:, 0] - CONDITIONING_VALUE).abs() <= half_width
    kept = draws[inside_band, 1]

    assert int(inside_band.sum()) > 2_000
    # The mean tolerance is a few standard errors of the retained draws; the
    # spread tolerance is relative because that is where the inflation lives.
    assert float(kept.mean()) == pytest.approx(float(analytic_mean), abs=0.08)
    assert float(kept.std()) == pytest.approx(float(analytic_std), rel=0.03)


def test_conditioned_sampling_draws_the_unknown_columns() -> None:
    """The conditioned adapter samples the reduced block with the head unchanged.

    The head's sampler takes its width from ``means.shape[-1]`` rather than
    from the width the head was built with, which is why a reduced adapter
    works without a head of its own.
    """
    params = student_t_pair_mixture()
    mask = torch.tensor([True, False])
    values = torch.tensor([[[CONDITIONING_VALUE, 0.0]]], dtype=torch.float64)

    conditioned = PredictorAdapter(student_t_head(D=2, K=2), params).condition(
        values, mask
    )

    reduced = conditioned.sample(n_samples=16, rng=torch.Generator().manual_seed(0))

    assert reduced.shape == (1, 16, 1, 1)
    assert torch.isfinite(reduced).all()


def test_adapter_condition_reduces_every_parameter_on_batched_horizons() -> None:
    """Every batch item and horizon is conditioned, and the mapping keeps its keys.

    The emitted mapping carries the diagonal-plus-factor scale; the conditioned
    one carries a dense Cholesky factor over the unknown columns, which is what
    the head's sampler and scorer read back.
    """
    generator = torch.Generator().manual_seed(42)
    B, H, K, D, rank = 2, 3, 4, 5, 2
    params = {
        "logits": torch.randn(B, H, K, generator=generator),
        "means": torch.randn(B, H, K, D, generator=generator),
        "lowrank_diag": torch.rand(B, H, K, D, generator=generator) + 0.5,
        "lowrank_U": torch.randn(B, H, K, D, rank, generator=generator),
        "df": torch.full((B, H, K), 6.0),
    }
    adapter = PredictorAdapter(student_t_head(D=D, K=K, covariance_rank=rank), params)

    values = torch.randn(B, H, D, generator=generator)
    mask = torch.tensor([True, False, True, False, False])

    conditioned = adapter.condition(values, mask)

    assert set(conditioned.params) == {"logits", "means", "chol", "df"}
    assert conditioned.params["logits"].shape == (B, H, K)
    assert conditioned.params["means"].shape == (B, H, K, 3)
    assert conditioned.params["chol"].shape == (B, H, K, 3, 3)
    torch.testing.assert_close(conditioned.params["df"], params["df"] + 2.0)
    assert conditioned.sample(n_samples=4).shape == (B, 4, H, 3)


def test_conditioning_a_low_rank_head_matches_its_dense_covariance() -> None:
    """The head's emitted diagonal-plus-factor covariance conditions end to end.

    The reference is the same covariance written out as a full Cholesky
    factor: the two describe one distribution, so their conditionals must
    agree, and the conditioned adapter must sample at the reduced width without
    further help.
    """
    head = student_t_head(D=3, K=2, covariance_rank=1)
    lowrank_diag = torch.tensor([[[[1.0, 0.5, 2.0], [0.75, 1.25, 0.5]]]])
    lowrank_U = torch.tensor([[[[[0.8], [-0.6], [0.4]], [[0.3], [0.9], [-0.7]]]]])
    shared = {
        "logits": torch.tensor([[[0.2, -0.3]]]),
        "means": torch.tensor([[[[0.0, 1.0, -1.0], [0.5, -0.5, 2.0]]]]),
        "df": torch.full((1, 1, 2), 8.0),
    }
    values = torch.tensor([[[1.5, 0.0, 0.0]]])
    mask = torch.tensor([True, False, False])

    from_lowrank = PredictorAdapter(
        head, {**shared, "lowrank_diag": lowrank_diag, "lowrank_U": lowrank_U}
    ).condition(values, mask)

    sigma = torch.diag_embed(lowrank_diag) + torch.matmul(
        lowrank_U, lowrank_U.transpose(-1, -2)
    )
    from_dense = PredictorAdapter(
        head, {**shared, "chol": torch.linalg.cholesky(sigma)}
    ).condition(values, mask)

    for key in ("logits", "means", "chol", "df"):
        torch.testing.assert_close(
            from_lowrank.params[key],
            from_dense.params[key],
            atol=1e-5,
            rtol=1e-5,
        )

    assert tuple(from_lowrank.sample(n_samples=4).shape) == (1, 4, 1, 2)
