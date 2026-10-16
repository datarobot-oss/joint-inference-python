"""Unit tests for confining Student-t mixture components to a box.

The oracles are closed forms wherever one exists. One interval column is one
Student-t CDF difference, so the walk must reproduce it to floating-point
noise. A centered orthant has the same probability under every elliptical
family, so ``1/4 + arcsin(rho) / (2 pi)`` prices it exactly at any degrees of
freedom. An off-center box is checked against brute-force draws from
``multivariate_t``, the only oracle that would notice a walk quietly behaving
like a normal one.

The property that makes the walk usable on a served surface is checked
directly: every draw lands inside the box, at any depth of tail, so a response
never has fewer draws than it asked for.
"""

import math

import pytest
import torch
from scipy.special import stdtr
from scipy.stats import multivariate_t

from jointfm_inference.model.inference.interval_conditioning import (
    effective_sample_size,
    walk_box,
)

# The quasi-Monte-Carlo error of the two-column walk at this point count;
# measured worst case across the correlations below is ~4e-4 relative.
ORTHANT_RTOL = 2e-3
# Float64 noise: the single-column path evaluates the same CDF difference the
# oracle does, through the same primitive.
EXACT_ATOL = 1e-12


def _df(value: float) -> torch.Tensor:
    """Return the degrees of freedom of one component."""
    return torch.tensor([value], dtype=torch.float64)


def _standard_pair(correlation: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and Cholesky factor of a standard correlated pair."""
    covariance = torch.tensor(
        [[1.0, correlation], [correlation, 1.0]], dtype=torch.float64
    )
    return (
        torch.zeros(1, 2, dtype=torch.float64),
        torch.linalg.cholesky(covariance).reshape(1, 2, 2),
    )


def test_an_open_lower_side_is_the_left_tail() -> None:
    """A half-open interval keeps its closed-form probability."""
    means = torch.zeros(1, 1, dtype=torch.float64)
    chol = torch.ones(1, 1, 1, dtype=torch.float64)
    lower = torch.tensor([-float("inf")], dtype=torch.float64)
    upper = torch.tensor([-1.5], dtype=torch.float64)

    walk = walk_box(means, chol, lower, upper, points=32, df=_df(5.0))

    assert float(walk.log_box_probability.exp()) == pytest.approx(
        float(stdtr(5.0, -1.5)), abs=EXACT_ATOL
    )


def test_effective_sample_size_counts_equal_weights() -> None:
    """Equal weights are worth their own count; one dominant weight is worth one draw."""
    equal = torch.zeros(256, dtype=torch.float64)
    assert float(effective_sample_size(equal)) == pytest.approx(256.0, rel=1e-9)

    dominant = torch.full((256,), -100.0, dtype=torch.float64)
    dominant[0] = 0.0
    assert float(effective_sample_size(dominant)) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        ((1.0, 1.0), (0.0, 2.0), "lower < upper"),
        ((-float("inf"), 0.0), (float("inf"), 1.0), "open on both sides"),
    ],
)
def test_walk_rejects_a_degenerate_interval(
    lower: tuple[float, ...], upper: tuple[float, ...], message: str
) -> None:
    """A degenerate bound pair is a malformed request, not an empty region."""
    means, chol = _standard_pair(0.0)
    with pytest.raises(ValueError, match=message):
        walk_box(
            means,
            chol,
            torch.tensor(lower, dtype=torch.float64),
            torch.tensor(upper, dtype=torch.float64),
            points=16,
            df=_df(6.0),
        )


def test_walk_rejects_a_non_positive_point_count() -> None:
    """An estimator with no points would silently return a meaningless number."""
    means, chol = _standard_pair(0.0)
    with pytest.raises(ValueError, match="points must be positive"):
        walk_box(
            means,
            chol,
            torch.zeros(2, dtype=torch.float64),
            torch.full((2,), float("inf"), dtype=torch.float64),
            points=0,
            df=_df(6.0),
        )


def test_walk_rejects_bounds_that_do_not_match_the_block() -> None:
    """Bounds are positional, so a length mismatch cannot be repaired by guessing."""
    means, chol = _standard_pair(0.0)
    with pytest.raises(ValueError, match="shape"):
        walk_box(
            means,
            chol,
            torch.zeros(3, dtype=torch.float64),
            torch.full((3,), float("inf"), dtype=torch.float64),
            points=16,
            df=_df(6.0),
        )


def test_student_t_single_column_box_probability_is_the_exact_cdf_difference() -> None:
    """One interval column is one Student-t CDF difference, at the component's own df."""
    means = torch.tensor([[0.5]], dtype=torch.float64)
    chol = torch.tensor([[[2.0]]], dtype=torch.float64)
    lower = torch.tensor([1.0], dtype=torch.float64)
    upper = torch.tensor([3.0], dtype=torch.float64)
    df = torch.tensor([6.0], dtype=torch.float64)

    walk = walk_box(means, chol, lower, upper, points=64, df=df)

    expected = float(stdtr(6.0, 1.25) - stdtr(6.0, 0.25))
    assert float(walk.log_box_probability.exp()) == pytest.approx(
        expected, abs=EXACT_ATOL
    )
    assert walk.is_exact


def test_student_t_single_column_weights_are_constant_within_a_component() -> None:
    """Without a second column no degrees of freedom rise, so the draws stay exact."""
    means = torch.zeros(1, 1, dtype=torch.float64)
    chol = torch.ones(1, 1, 1, dtype=torch.float64)

    walk = walk_box(
        means,
        chol,
        torch.tensor([-0.5], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        points=128,
        df=torch.tensor([4.0], dtype=torch.float64),
    )

    spread = walk.log_weights.max() - walk.log_weights.min()
    assert float(spread) == pytest.approx(0.0, abs=EXACT_ATOL)


def test_a_student_t_tail_is_heavier_than_the_normal_one_at_the_same_bound() -> None:
    """The df must reach the distribution function, not only the scale."""
    means = torch.zeros(1, 1, dtype=torch.float64)
    chol = torch.ones(1, 1, 1, dtype=torch.float64)
    lower = torch.tensor([4.0], dtype=torch.float64)
    upper = torch.tensor([float("inf")], dtype=torch.float64)

    student = walk_box(
        means,
        chol,
        lower,
        upper,
        points=16,
        df=torch.tensor([5.0], dtype=torch.float64),
    )

    assert float(student.log_box_probability.exp()) == pytest.approx(
        float(1.0 - stdtr(5.0, 4.0)), abs=EXACT_ATOL
    )
    normal_tail = float(
        1.0 - torch.special.ndtr(torch.tensor(4.0, dtype=torch.float64))
    )
    assert float(student.log_box_probability) > math.log(normal_tail) + 4.0


def test_a_huge_degrees_of_freedom_reproduces_the_normal_box_probability() -> None:
    """The Student-t law must degenerate to the normal one as df grows."""
    means = torch.tensor([[0.5]], dtype=torch.float64)
    chol = torch.tensor([[[2.0]]], dtype=torch.float64)
    lower = torch.tensor([1.0], dtype=torch.float64)
    upper = torch.tensor([3.0], dtype=torch.float64)

    walk = walk_box(means, chol, lower, upper, points=16, df=_df(1e8))

    normal = torch.special.ndtr(
        torch.tensor(1.25, dtype=torch.float64)
    ) - torch.special.ndtr(torch.tensor(0.25, dtype=torch.float64))
    assert float(walk.log_box_probability.exp()) == pytest.approx(
        float(normal), rel=1e-6
    )


@pytest.mark.parametrize("correlation", [-0.7, 0.0, 0.4, 0.9])
@pytest.mark.parametrize("degrees_of_freedom", [3.0, 12.0])
def test_a_centered_orthant_is_the_elliptical_closed_form_at_any_df(
    correlation: float, degrees_of_freedom: float
) -> None:
    """Every elliptical family shares the orthant probability of its correlation.

    A scale mixture of normals about its own center cannot move the sign
    pattern's probability, so ``1/4 + arcsin(rho) / (2 pi)`` prices the
    Student-t orthant exactly as it prices the Gaussian one. That makes it the
    one exact multi-column oracle available here, and it holds only if the
    degrees-of-freedom rise and the scale inflation are applied together: either
    one alone would tilt the walk away from the family.
    """
    means, chol = _standard_pair(correlation)

    walk = walk_box(
        means,
        chol,
        torch.zeros(2, dtype=torch.float64),
        torch.full((2,), float("inf"), dtype=torch.float64),
        points=4096,
        df=torch.tensor([degrees_of_freedom], dtype=torch.float64),
    )

    expected = 0.25 + math.asin(correlation) / (2.0 * math.pi)
    assert float(walk.log_box_probability.exp()) == pytest.approx(
        expected, rel=ORTHANT_RTOL
    )


def test_an_off_center_student_t_box_matches_brute_force_sampling() -> None:
    """Off center no orthant identity applies, so sampling checks the t box."""
    correlation, degrees_of_freedom = 0.6, 4.0
    means, chol = _standard_pair(correlation)
    lower = torch.tensor([0.75, -0.25], dtype=torch.float64)
    upper = torch.tensor([float("inf"), 2.0], dtype=torch.float64)

    walk = walk_box(
        means,
        chol,
        lower,
        upper,
        points=16384,
        df=torch.tensor([degrees_of_freedom], dtype=torch.float64),
    )

    scale = [[1.0, correlation], [correlation, 1.0]]
    draws = multivariate_t(loc=[0.0, 0.0], shape=scale, df=degrees_of_freedom).rvs(
        size=400_000, random_state=17
    )
    inside = (
        (draws[:, 0] >= 0.75) & (draws[:, 1] >= -0.25) & (draws[:, 1] <= 2.0)
    ).mean()

    assert float(walk.log_box_probability.exp()) == pytest.approx(
        float(inside), rel=2e-2
    )


@pytest.mark.parametrize("threshold", [0.0, 3.0, 8.0])
def test_every_student_t_draw_lands_inside_the_box_at_any_depth(
    threshold: float,
) -> None:
    """A heavier tail changes the region's mass, not the guarantee the draws carry."""
    means, chol = _standard_pair(0.5)
    lower = torch.full((2,), threshold, dtype=torch.float64)
    upper = torch.full((2,), float("inf"), dtype=torch.float64)

    walk = walk_box(
        means,
        chol,
        lower,
        upper,
        points=1024,
        df=torch.tensor([3.0], dtype=torch.float64),
    )

    assert bool(torch.isfinite(walk.draws).all())
    assert bool((walk.draws >= lower).all())


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (torch.tensor([6.0, 6.0], dtype=torch.float64), "one value per component"),
        (torch.tensor([0.0], dtype=torch.float64), "strictly positive"),
    ],
)
def test_walk_rejects_degrees_of_freedom_it_cannot_use(
    df: torch.Tensor, message: str
) -> None:
    """A df that does not line up with the components would silently mix them up."""
    means, chol = _standard_pair(0.3)

    with pytest.raises(ValueError, match=message):
        walk_box(
            means,
            chol,
            torch.zeros(2, dtype=torch.float64),
            torch.full((2,), float("inf"), dtype=torch.float64),
            points=32,
            df=df,
        )


def test_a_region_past_the_representable_tail_reports_no_mass_rather_than_a_nan() -> (
    None
):
    """A NaN region probability would reach the caller as a number it cannot read.

    At an absurd enough bound the Student-t distribution function underflows
    at both ends, and the difference of the two resulting values is no
    probability at all. What the caller is told instead is minus infinity, and
    the draws stay inside the bounds they were given.
    """
    means = torch.zeros(1, 1, dtype=torch.float64)
    chol = torch.ones(1, 1, 1, dtype=torch.float64)
    lower = torch.tensor([-1e300], dtype=torch.float64)
    upper = torch.tensor([-1e200], dtype=torch.float64)

    walk = walk_box(means, chol, lower, upper, points=8, df=_df(3.0))

    assert float(walk.log_box_probability) == -float("inf")
    assert bool((walk.draws >= lower).all() and (walk.draws <= upper).all())
