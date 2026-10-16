"""Tests for the factored time/row attention encoder of JointFM 1."""

import pytest
import torch

from jointfm_inference.model.jointfm_1.encoder import (
    FactoredEncoder,
    FactoredEncoderLayer,
    _query_row_isolation_mask,
)

GRID_BATCH = 2
GRID_STEPS = 6
GRID_COLUMNS = 3
GRID_CHANNELS = 8
QUERY_ROWS = 3
PERTURBATION = 5.0


def _grid_layer(*, time_blocks: int, row_blocks: int) -> FactoredEncoderLayer:
    """Build one small layer with the token-wise MLP feed-forward block."""
    torch.manual_seed(0)
    layer = FactoredEncoderLayer(
        d_model=GRID_CHANNELS,
        nhead=2,
        dim_feedforward=2 * GRID_CHANNELS,
        layer_norm_eps=1e-5,
        time_blocks=time_blocks,
        row_blocks=row_blocks,
        cnn_kernel_size=None,
    )
    return layer.eval()


def _encoder(*, cnn_first_layers: int = 1) -> FactoredEncoder:
    """Build a two-layer encoder mixing convolutional and MLP feed-forward blocks."""
    torch.manual_seed(0)
    encoder = FactoredEncoder(
        num_layers=2,
        d_model=GRID_CHANNELS,
        nhead=2,
        dim_feedforward=2 * GRID_CHANNELS,
        layer_norm_eps=1e-5,
        time_blocks=1,
        row_blocks=1,
        cnn_kernel_size=3,
        cnn_first_layers=cnn_first_layers,
    )
    return encoder.eval()


def _run_grid(
    module: torch.nn.Module, grid: torch.Tensor, *, query_rows: int = 0
) -> torch.Tensor:
    """Run ``module`` on a time-major ``(B, T, D, C)`` grid and return the grid."""
    B, T, D, C = grid.shape
    with torch.no_grad():
        out = module(
            grid.reshape(B, T * D, C), T_ctx=T - query_rows, D=D, query_rows=query_rows
        )
    return out.view(B, T, D, C)


def _grid(steps: int = GRID_STEPS) -> torch.Tensor:
    """Draw a reproducible time-major token grid."""
    generator = torch.Generator().manual_seed(1)
    return torch.randn(
        GRID_BATCH, steps, GRID_COLUMNS, GRID_CHANNELS, generator=generator
    )


def test_time_blocks_attend_within_each_column_over_all_steps() -> None:
    """Time attention reads one column's whole history and no other column.

    The tokens are flattened time-major, so a bare ``view(B * D, T, C)`` would
    group runs of ``T / D`` steps across all columns instead.
    """
    layer = _grid_layer(time_blocks=1, row_blocks=0)
    grid = _grid()
    baseline = _run_grid(layer, grid)

    other_columns = grid.clone()
    other_columns[:, :, 1:, :] += PERTURBATION
    torch.testing.assert_close(
        _run_grid(layer, other_columns)[:, :, 0], baseline[:, :, 0]
    )

    own_last_step = grid.clone()
    own_last_step[:, -1, 0, :] += PERTURBATION
    assert not torch.allclose(
        _run_grid(layer, own_last_step)[:, 0, 0], baseline[:, 0, 0]
    )


def test_row_blocks_attend_across_columns_within_one_step() -> None:
    layer = _grid_layer(time_blocks=0, row_blocks=1)
    grid = _grid()
    baseline = _run_grid(layer, grid)

    perturbed = grid.clone()
    perturbed[:, -1, -1, :] += PERTURBATION
    shifted = _run_grid(layer, perturbed)
    assert not torch.allclose(shifted[:, -1, 0], baseline[:, -1, 0])
    torch.testing.assert_close(shifted[:, :-1], baseline[:, :-1])


def test_isolation_mask_blocks_history_to_query_and_query_to_query() -> None:
    mask = _query_row_isolation_mask(2, 2, torch.device("cpu"))
    assert mask is not None
    expected = torch.tensor(
        [
            [False, False, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )
    assert torch.equal(mask, expected)
    assert _query_row_isolation_mask(4, 0, torch.device("cpu")) is None
    with pytest.raises(ValueError, match="query_rows"):
        _query_row_isolation_mask(4, -1, torch.device("cpu"))


def test_history_rows_never_read_query_rows() -> None:
    """Appending query rows leaves every encoded history row unchanged."""
    encoder = _encoder()
    grid = _grid(GRID_STEPS + QUERY_ROWS)

    with_queries = _run_grid(encoder, grid, query_rows=QUERY_ROWS)
    history_only = _run_grid(encoder, grid[:, :GRID_STEPS])

    torch.testing.assert_close(with_queries[:, :GRID_STEPS], history_only)


def test_a_query_row_does_not_depend_on_the_other_query_rows() -> None:
    encoder = _encoder()
    grid = _grid(GRID_STEPS + QUERY_ROWS)
    middle = GRID_STEPS + 1

    together = _run_grid(encoder, grid, query_rows=QUERY_ROWS)
    alone = _run_grid(encoder, grid[:, [*range(GRID_STEPS), middle]], query_rows=1)

    torch.testing.assert_close(alone[:, -1], together[:, middle])


@pytest.mark.parametrize(
    ("cnn_first_layers", "convolutional"), [(1, [True, False]), (0, [True, True])]
)
def test_each_layer_owns_exactly_the_feed_forward_block_it_runs(
    cnn_first_layers: int, convolutional: list[bool]
) -> None:
    """``cnn_first_layers`` leading layers convolve; ``0`` means every layer does."""
    encoder = _encoder(cnn_first_layers=cnn_first_layers)
    for layer, expected in zip(encoder.layers, convolutional, strict=True):
        assert isinstance(layer, FactoredEncoderLayer)
        assert (layer.ffn_cnn is not None) == expected
        assert (layer.ffn_mlp is None) == expected


def test_layer_refuses_inconsistent_grid_sizes_and_even_kernels() -> None:
    layer = _grid_layer(time_blocks=1, row_blocks=1)
    with pytest.raises(ValueError, match=r"\(T_ctx \+ query_rows\) \* D"):
        layer(torch.randn(1, 7, GRID_CHANNELS), T_ctx=2, D=3, query_rows=0)
    with pytest.raises(ValueError, match="positive odd integer"):
        FactoredEncoderLayer(
            d_model=GRID_CHANNELS,
            nhead=2,
            dim_feedforward=2 * GRID_CHANNELS,
            layer_norm_eps=1e-5,
            time_blocks=1,
            row_blocks=1,
            cnn_kernel_size=2,
        )
