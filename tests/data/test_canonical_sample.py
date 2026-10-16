"""Tests for the invariants a canonical sample enforces on construction."""

from typing import Any

import pytest
import torch

from jointfm_inference.data.types import CanonicalSample


def _sample(**overrides: Any) -> CanonicalSample:
    fields: dict[str, Any] = {
        "values": torch.zeros(3, 2),
        "time_positions": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        "time_index_mode": "ordinal",
        "query_row_mask": torch.tensor([False, False, True]),
        "metadata": {"column_names": ["a", "b"]},
    }
    fields.update(overrides)
    return CanonicalSample(**fields)


def test_accepts_history_rows_followed_by_query_rows() -> None:
    sample = _sample()

    assert sample.query_row_mask.tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"values": torch.zeros(3, 2, dtype=torch.long)}, "floating-point"),
        ({"values": torch.zeros(3)}, "floating-point"),
        (
            {"values": torch.tensor([[0.0, 0.0], [float("nan"), 0.0], [0.0, 0.0]])},
            "values must be finite",
        ),
        ({"time_positions": torch.zeros(2)}, r"shape \(rows,\)"),
        (
            {"time_positions": torch.tensor([0.0, 1.0, float("inf")])},
            "time_positions must be finite",
        ),
        (
            {"time_positions": torch.tensor([0.0, 2.0, 1.0])},
            "strictly monotonically increasing",
        ),
        ({"time_index_mode": "weekly"}, "time_index_mode must be one of"),
        ({"query_row_mask": torch.tensor([0, 0, 1])}, "bool tensor"),
        ({"query_row_mask": torch.tensor([False, True])}, "bool tensor"),
        (
            {"query_row_mask": torch.tensor([False, True, False])},
            "query rows must follow every history row",
        ),
        ({"metadata": {}}, "one name per column"),
        (
            {
                "values": torch.zeros(0, 2),
                "time_positions": torch.zeros(0, dtype=torch.float64),
                "query_row_mask": torch.zeros(0, dtype=torch.bool),
            },
            "at least one row",
        ),
        ({"metadata": {"column_names": ["a", "a"]}}, "must be unique"),
    ],
)
def test_refuses_a_malformed_sample(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _sample(**overrides)
