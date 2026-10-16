"""Contract tests for ``PredictorAdapter`` over batched Student-t mixture parameters."""

import pytest
import torch

from jointfm_inference.model.inference import PredictorAdapter
from tests.support.inference import create_student_t_adapter, student_t_head

B, H, K, D = 2, 3, 2, 3


def _params() -> dict[str, torch.Tensor]:
    """Return low-rank mixture parameters shaped like one forward's output."""
    generator = torch.Generator().manual_seed(0)
    logits = torch.zeros(B, H, K)
    logits[..., 0] = 2.0
    means = torch.arange(K, dtype=torch.float32).reshape(1, 1, K, 1).expand(B, H, K, D)
    return {
        "logits": logits,
        "means": means.contiguous(),
        "df": torch.full((B, H, K), 8.0),
        "lowrank_diag": torch.full((B, H, K, D), 0.5),
        "lowrank_U": 0.3 * torch.randn(B, H, K, D, 1, generator=generator),
    }


def _adapter() -> PredictorAdapter:
    return create_student_t_adapter(_params())


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"logits": torch.zeros(H, K)}, r"logits must have shape \(B, H, K\)"),
        (
            {**_params(), "df": torch.full((B, H, K), float("nan"))},
            "'df' contains NaNs",
        ),
    ],
)
def test_construction_refuses_malformed_parameters(
    params: dict[str, torch.Tensor], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        PredictorAdapter(student_t_head(D=D, K=K), params)


def test_log_prob_scores_one_continuation_per_row() -> None:
    adapter = _adapter()
    values = torch.randn(B, H, D, generator=torch.Generator().manual_seed(1))

    scores = adapter.log_prob(values)

    assert scores.shape == (B, H)
    torch.testing.assert_close(scores, adapter.head.log_prob(adapter.params, values))


@pytest.mark.parametrize(
    ("values", "error", "match"),
    [
        (torch.zeros(B, H, D, dtype=torch.int64), TypeError, "floating-point"),
        (torch.full((B, H, D), float("nan")), ValueError, "NaNs or Infs"),
        (torch.zeros(H, D), ValueError, r"shape \(B, H, D\)"),
        (torch.zeros(B, 4, H, D), ValueError, r"shape \(B, H, D\)"),
        (torch.zeros(B + 1, H, D), ValueError, r"matching the parameters' \(2, 3\)"),
        (torch.zeros(B, H - 1, D), ValueError, r"matching the parameters' \(2, 3\)"),
    ],
)
def test_log_prob_refuses_malformed_values(
    values: torch.Tensor, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        _adapter().log_prob(values)


def test_point_forecast_and_quantiles_follow_the_mixture() -> None:
    adapter = _adapter()
    params = adapter.params
    weights = torch.softmax(params["logits"], dim=-1)
    expected = (weights.unsqueeze(-1) * params["means"]).sum(dim=-2)

    mean = adapter.mean(n_samples=20_000, rng=torch.Generator().manual_seed(2))
    torch.testing.assert_close(mean, expected, atol=0.05, rtol=0.0)

    quantiles = adapter.quantiles([0.1, 0.9], n_samples=64)
    assert quantiles.shape == (2, B, H, D)
    assert (quantiles[0] < quantiles[1]).all()


def test_sample_draws_every_column() -> None:
    samples = _adapter().sample(n_samples=4)
    assert samples.shape == (B, 4, H, D)


def test_conditioning_reduces_the_width_to_the_unknown_columns() -> None:
    adapter = _adapter()
    values = torch.randn(B, H, D, generator=torch.Generator().manual_seed(3))
    mask = torch.tensor([False, True, False])

    conditioned = adapter.condition(values, mask)

    assert conditioned.params.keys() == {"logits", "means", "chol", "df"}
    assert conditioned.params["means"].shape == (B, H, K, D - 1)
    assert conditioned.sample(n_samples=4).shape == (B, 4, H, D - 1)
    assert conditioned.log_prob(values[..., ~mask]).shape == (B, H)


def test_conditioning_on_nothing_keeps_the_full_joint() -> None:
    adapter = _adapter()
    values = torch.zeros(B, H, D)

    unconditioned = adapter.condition(values, torch.zeros(D, dtype=torch.bool))

    assert unconditioned.params["means"].shape == (B, H, K, D)
    assert unconditioned.sample(n_samples=2).shape == (B, 2, H, D)
    probe = torch.randn(B, H, D, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(unconditioned.log_prob(probe), adapter.log_prob(probe))


def test_conditioning_refuses_misshaped_values_and_masks() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="full width D"):
        adapter.condition(torch.zeros(B, H, D - 1), torch.ones(D - 1, dtype=torch.bool))
    for mask in (torch.ones(D - 1), torch.ones(B, H, D)):
        with pytest.raises(ValueError, match=r"mask must have shape \(3,\)"):
            adapter.condition(torch.zeros(B, H, D), mask.to(torch.bool))
    with pytest.raises(TypeError, match="mask must be a bool tensor"):
        adapter.condition(torch.zeros(B, H, D), torch.tensor([1.0, 0.0, 0.0]))


def test_conditioning_requires_a_covariance() -> None:
    params = {key: _params()[key] for key in ("logits", "means", "df")}
    adapter = PredictorAdapter(student_t_head(D=D, K=K), params)
    with pytest.raises(KeyError, match="neither"):
        adapter.condition(torch.zeros(B, H, D), torch.tensor([True, False, False]))
