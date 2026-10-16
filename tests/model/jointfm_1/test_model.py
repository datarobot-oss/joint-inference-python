"""End-to-end invariants of the JointFM 1 network and its configuration."""

from dataclasses import asdict

import pytest
import torch

from jointfm_inference.model.jointfm_1.config import JointFMConfig
from jointfm_inference.model.jointfm_1.head import (
    AUTO_STANDARDIZATION_CENTER_KEY,
    AUTO_STANDARDIZATION_SCALE_KEY,
)
from jointfm_inference.model.jointfm_1.model import SinusoidalPosition
from tests.support.models import TINY_CONFIG, tiny_config, tiny_model

BATCH, STEPS, COLUMNS, HORIZONS = 2, 8, 3, 5
HEAD_KEYS = ("logits", "means", "df", "lowrank_diag", "lowrank_U")


def _past() -> torch.Tensor:
    return torch.randn(
        BATCH, STEPS, COLUMNS, generator=torch.Generator().manual_seed(4)
    )


def test_forward_emits_data_space_parameters_for_the_active_columns() -> None:
    model = tiny_model()
    with torch.no_grad():
        out = model(_past(), torch.arange(HORIZONS))

    K, rank = TINY_CONFIG.n_mixtures, TINY_CONFIG.covariance_rank
    assert out["logits"].shape == (BATCH, HORIZONS, K)
    assert out["df"].shape == (BATCH, HORIZONS, K)
    assert out["means"].shape == (BATCH, HORIZONS, K, COLUMNS)
    assert out["lowrank_diag"].shape == (BATCH, HORIZONS, K, COLUMNS)
    assert out["lowrank_U"].shape == (BATCH, HORIZONS, K, COLUMNS, rank)
    assert out[AUTO_STANDARDIZATION_CENTER_KEY].shape == (BATCH, 1, COLUMNS)
    assert out[AUTO_STANDARDIZATION_SCALE_KEY].shape == (BATCH, 1, COLUMNS)


def test_a_horizon_forecast_does_not_depend_on_the_other_queried_horizons() -> None:
    model = tiny_model()
    past = _past()
    with torch.no_grad():
        alone = model(past, torch.tensor([2]))
        together = model(past, torch.tensor([0, 2, 5]))

    for key in HEAD_KEYS:
        torch.testing.assert_close(alone[key][:, 0], together[key][:, 1])


def test_a_per_column_affine_change_of_units_moves_the_forecast_with_it() -> None:
    """Standardizing by the last value and the history spread makes units irrelevant."""
    model = tiny_model()
    past = _past()
    shift = torch.tensor([100.0, -3.0, 0.5])
    factor = torch.tensor([10.0, 0.1, 2.0])
    horizons = torch.arange(HORIZONS)
    with torch.no_grad():
        base = model(past, horizons)
        moved = model(shift + factor * past, horizons)

    torch.testing.assert_close(moved["logits"], base["logits"])
    torch.testing.assert_close(moved["df"], base["df"])
    torch.testing.assert_close(
        moved["means"], shift + factor * base["means"], rtol=1e-4, atol=1e-3
    )
    torch.testing.assert_close(
        moved["lowrank_diag"], factor**2 * base["lowrank_diag"], rtol=1e-4, atol=1e-6
    )
    torch.testing.assert_close(
        moved["lowrank_U"],
        factor.unsqueeze(-1) * base["lowrank_U"],
        rtol=1e-4,
        atol=1e-5,
    )


def test_forward_refuses_more_columns_than_the_capacity_and_misshaped_time_indices() -> (
    None
):
    model = tiny_model()
    with pytest.raises(ValueError, match="the capacity is 4"):
        model(torch.randn(1, STEPS, TINY_CONFIG.column_capacity + 1), torch.arange(2))
    with pytest.raises(ValueError, match=r"past_time_indices must have shape \(T,\)"):
        model(_past(), torch.arange(2), past_time_indices=torch.arange(STEPS - 1))


def test_config_round_trips_through_its_checkpoint_mapping() -> None:
    assert JointFMConfig.from_mapping(asdict(TINY_CONFIG)) == TINY_CONFIG


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"d_model": 0}, "d_model must be positive"),
        ({"pooling_seeds": 0}, "pooling_seeds must be positive"),
        ({"attention_time_blocks": -1}, "must be >= 0"),
        ({"attention_time_blocks": 0, "attention_row_blocks": 0}, "at least one"),
        ({"nhead": 3}, "divisible by nhead"),
        ({"pooling_nhead": 3}, "divisible by pooling_nhead"),
        ({"covariance_rank": 5}, "covariance_rank must not exceed column_capacity"),
        ({"attention_cnn_first_layers": -1}, "attention_cnn_first_layers must be >= 0"),
        ({"d_model": 9, "nhead": 3, "pooling_nhead": 3}, "d_model must be even"),
        (
            {"attention_cnn_kernel_size": 2},
            "attention_cnn_kernel_size must be a positive odd",
        ),
    ],
)
def test_config_refuses_sizes_the_network_cannot_be_built_with(
    overrides: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        tiny_config(**overrides)


def test_sinusoidal_position_requires_an_even_width() -> None:
    with pytest.raises(ValueError, match="even dim"):
        SinusoidalPosition(3, max_len=4, base_freq=100.0)
