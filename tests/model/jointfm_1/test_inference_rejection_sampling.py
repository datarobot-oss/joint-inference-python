"""Rejection bounds of ``PredictorAdapter.sample`` over a real Student-t head."""

import pytest
import torch

from tests.support.inference import create_student_t_adapter


def _adapter(*, mean: float, scale: float = 1.0):
    """Return an adapter over one single-column Student-t at ``mean``."""
    return create_student_t_adapter(
        {
            "logits": torch.zeros(1, 1, 1),
            "means": torch.full((1, 1, 1, 1), mean),
            "chol": torch.full((1, 1, 1, 1, 1), scale),
            "df": torch.full((1, 1, 1), 30.0),
        }
    )


def test_a_lower_bound_redraws_values_below_it() -> None:
    torch.manual_seed(0)
    adapter = _adapter(mean=-0.1)

    assert (adapter.sample(n_samples=50) < 0).any()
    assert (adapter.sample(n_samples=50, min_value=0.0, max_retries=100) >= 0.0).all()


def test_an_upper_bound_redraws_values_above_it() -> None:
    torch.manual_seed(0)
    adapter = _adapter(mean=0.1)

    assert (adapter.sample(n_samples=50) > 0).any()
    assert (adapter.sample(n_samples=50, max_value=0.0, max_retries=100) <= 0.0).all()


def test_a_per_column_bound_applies_to_its_own_column() -> None:
    torch.manual_seed(0)
    adapter = create_student_t_adapter(
        {
            "logits": torch.zeros(1, 1, 1),
            "means": torch.zeros(1, 1, 1, 2),
            "chol": torch.eye(2).reshape(1, 1, 1, 2, 2),
            "df": torch.full((1, 1, 1), 30.0),
        }
    )

    samples = adapter.sample(
        n_samples=50, min_value=torch.tensor([0.0, -1e9]), max_retries=100
    )

    assert (samples[..., 0] >= 0.0).all()
    assert (samples[..., 1] < 0.0).any()


@pytest.mark.parametrize(
    ("mean", "bound", "match"),
    [
        (-100.0, {"min_value": 0.0}, "min_value=0.0 violation"),
        (100.0, {"max_value": 0.0}, "max_value=0.0 violation"),
    ],
)
def test_an_unreachable_bound_fails_after_the_retries(
    mean: float, bound: dict, match: str
) -> None:
    torch.manual_seed(7)
    with pytest.raises(RuntimeError, match=match):
        _adapter(mean=mean, scale=0.1).sample(n_samples=10, max_retries=5, **bound)


def test_a_lower_bound_above_the_upper_bound_is_refused() -> None:
    with pytest.raises(ValueError, match="min_value must not exceed max_value"):
        _adapter(mean=0.0).sample(n_samples=2, min_value=1.0, max_value=-1.0)
