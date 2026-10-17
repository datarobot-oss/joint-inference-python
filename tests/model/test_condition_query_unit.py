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

"""Unit tests for answering one conditioning request at one emission position.

The oracles are chosen so that no test restates the implementation. A
single-component Student-t has a closed-form conditional: its location is the
Gaussian Schur-complement shift, its degrees of freedom rise by the number of
pinned columns, and its scale is inflated by how surprising the pinned values
were. Its marginal density of the pinned block is a closed form too, so the
equality step and the plausibility number are both checked against arithmetic
written out in the test. A centered orthant has the same probability under
every elliptical family, which prices the interval step's region probability
exactly at any degrees of freedom, and a single interval column is one
Student-t distribution function difference. Only the composed answer, where no
closed form exists, is checked against plain rejection sampling from the same
reduced mixture.

The measure-zero asymmetry between the two condition kinds shows up here as a
practical matter: the rejection oracle can check an interval condition but
could never check an equality one, because no draw would survive it.
"""

import math

import pytest
import torch
from scipy.special import stdtr

from jointfm_inference.model.inference import PredictorAdapter
from jointfm_inference.model.inference.conditional_query import (
    ConditionalReadOut,
    ConditionRequest,
    condition_at_position,
)
from jointfm_inference.model.jointfm_1.head import (
    AUTO_STANDARDIZATION_CENTER_KEY,
    AUTO_STANDARDIZATION_SCALE_KEY,
)
from tests.support.inference import (
    create_student_t_adapter,
    student_t_head,
    student_t_log_density,
    student_t_pair_mixture,
)

# Slack for the Cholesky policy's stabilization floor on an otherwise exact path.
EXACT_ATOL = 1e-5
# Monte Carlo slack against a rejection oracle at the draw budgets used here.
ORACLE_ATOL = 0.06
# Location of the two-column fixture, shared by every closed form below.
_LOCATION = (0.25, -0.5)
# Degrees of freedom of the two-column fixture.
_DF = 7.0
# Half-width of the interval that stands in for an equality condition below;
# the truncated conditional converges to the pinned one as it shrinks.
_NARROW_HALF_WIDTH = 1e-4


def _region_probability(answer: ConditionalReadOut) -> float:
    """Return the probability of the region an interval condition reports."""
    assert answer.region_log_probability is not None
    return float(answer.region_log_probability.exp())


def _single_component(correlation: float = 0.6) -> PredictorAdapter:
    """Build a one-component two-column Student-t with unit marginal scales.

    Unit scales make the correlation the conditional location's slope, so
    every closed form below reads directly off ``_LOCATION``, ``_DF`` and the
    correlation.
    """
    scale = torch.tensor([[1.0, correlation], [correlation, 1.0]])
    return create_student_t_adapter(
        {
            "logits": torch.zeros(1, 1, 1),
            "means": torch.tensor(_LOCATION).reshape(1, 1, 1, 2),
            "chol": torch.linalg.cholesky(scale).reshape(1, 1, 1, 2, 2),
            "df": torch.full((1, 1, 1), _DF),
        }
    )


def _correlated_mixture() -> PredictorAdapter:
    """Build a four-column, two-position mixture whose components disagree.

    The degrees of freedom differ across components, so the posterior weights,
    the raised degrees of freedom and the scale inflations all carry
    component-specific numbers.
    """
    generator = torch.Generator().manual_seed(7)
    return create_student_t_adapter(
        {
            "logits": torch.randn(1, 2, 3, generator=generator),
            "means": torch.randn(1, 2, 3, 4, generator=generator),
            "chol": torch.tril(torch.randn(1, 2, 3, 4, 4, generator=generator) * 0.4)
            + torch.eye(4),
            "df": torch.tensor([5.0, 9.0, 14.0]).expand(1, 2, 3).clone(),
        }
    )


def _centered(scale: torch.Tensor, df: float) -> PredictorAdapter:
    """Build a one-component zero-location Student-t over ``scale``'s columns."""
    width = int(scale.shape[-1])
    return create_student_t_adapter(
        {
            "logits": torch.zeros(1, 1, 1),
            "means": torch.zeros(1, 1, 1, width),
            "chol": torch.linalg.cholesky(scale).reshape(1, 1, 1, width, width),
            "df": torch.full((1, 1, 1), df),
        }
    )


def _pin_first(pinned: float) -> ConditionRequest:
    """Pin the first column of the first position."""
    return ConditionRequest(horizon=0, equality_columns=(0,), equality_values=(pinned,))


def test_the_plausibility_moves_by_the_jacobian_when_a_column_changes_unit() -> None:
    """Quoting a column in another unit may move the density only by its Jacobian.

    A density carries the reciprocal units of the columns it scores, so
    rescaling them shifts the reported log density by exactly the sum of the
    log scales and must change nothing else. The stabilization floor inside the
    Cholesky policy is what can break this: an absolute one inflates a small
    variance by a multiple of itself, and the residue it leaves grows with the
    Mahalanobis distance of the pinned point, so a far-tail pin -- the case the
    number is reported for -- drifts most.
    """
    scales = (1000.0, 0.01, 1.0)
    pins = (2.0, -2.0, 1.5)
    correlation = torch.tensor(
        [
            [1.00, 0.90, 0.40, 0.30],
            [0.90, 1.00, 0.35, 0.25],
            [0.40, 0.35, 1.00, 0.20],
            [0.30, 0.25, 0.20, 1.00],
        ],
        dtype=torch.float64,
    )
    equality_columns = (0, 1, 2)

    def _served(rescale: bool) -> float:
        """Serve the pin with the first three columns in their own units."""
        factors = torch.tensor(
            (*scales, 1.0) if rescale else (1.0,) * 4, dtype=torch.float64
        )
        covariance = correlation / torch.outer(factors, factors)
        adapter = create_student_t_adapter(
            {
                "logits": torch.zeros(1, 1, 1, dtype=torch.float64),
                "means": torch.zeros(1, 1, 1, 4, dtype=torch.float64),
                "chol": torch.linalg.cholesky(covariance).reshape(1, 1, 1, 4, 4),
                "df": torch.full((1, 1, 1), 6.0, dtype=torch.float64),
            }
        )
        scaled = ConditionRequest(
            horizon=0,
            equality_columns=equality_columns,
            equality_values=tuple(
                pin / float(factors[column])
                for column, pin in zip(equality_columns, pins)
            ),
        )
        read_out = condition_at_position(adapter, scaled, n_samples=8, column_count=4)
        assert read_out.equality_log_density is not None
        return float(read_out.equality_log_density)

    expected = sum(math.log(scale) for scale in scales)

    assert _served(rescale=True) - _served(rescale=False) == pytest.approx(
        expected, abs=1e-9
    )


def test_the_standardization_metadata_is_not_a_conditioned_parameter() -> None:
    """The emitted mapping carries more than the mixture's parameters.

    When the network standardizes its input it attaches the history window's
    center and scale to what it returns, and the session hands the whole
    mapping to the adapter. They are one window-wide pair with no column
    layout to narrow, so a conditioning path that treats every entry as a
    per-column parameter refuses the request instead of answering it. The
    answer must be the one the same parameters give without the metadata.
    """
    adapter = _single_component()
    with_metadata = PredictorAdapter(
        adapter.head,
        {
            **adapter.params,
            AUTO_STANDARDIZATION_CENTER_KEY: torch.tensor([[[0.25, -0.5]]]),
            AUTO_STANDARDIZATION_SCALE_KEY: torch.tensor([[[2.0, 3.0]]]),
        },
    )

    answered = condition_at_position(
        with_metadata, _pin_first(1.75), n_samples=16, column_count=2
    )
    oracle = condition_at_position(
        adapter, _pin_first(1.75), n_samples=16, column_count=2
    )

    assert answered.read_out_columns == oracle.read_out_columns
    assert torch.allclose(answered.mean(), oracle.mean(), atol=EXACT_ATOL)
    assert answered.equality_log_density is not None
    assert oracle.equality_log_density is not None
    assert torch.allclose(
        answered.equality_log_density, oracle.equality_log_density, atol=EXACT_ATOL
    )


def test_equality_only_mean_is_the_gaussian_conditional_location() -> None:
    """A Student-t conditional location is the Schur-complement shift."""
    correlation, pinned = 0.6, 1.75

    read_out = condition_at_position(
        _single_component(correlation), _pin_first(pinned), n_samples=16, column_count=2
    )

    expected = _LOCATION[1] + correlation * (pinned - _LOCATION[0])
    assert read_out.read_out_columns == (1,)
    assert float(read_out.mean()[0, 0, 1]) == pytest.approx(expected, abs=EXACT_ATOL)
    assert float(read_out.mean()[0, 0, 0]) == pytest.approx(pinned, abs=EXACT_ATOL)


def test_equality_plausibility_is_the_marginal_t_density_of_the_pinned_values() -> None:
    """The plausibility number is the model's t density at the pinned values."""
    pinned = 1.75

    read_out = condition_at_position(
        _single_component(), _pin_first(pinned), n_samples=16, column_count=2
    )

    expected = float(student_t_log_density(pinned, _LOCATION[0], 1.0, _DF))
    assert read_out.equality_log_density is not None
    assert float(read_out.equality_log_density) == pytest.approx(
        expected, abs=EXACT_ATOL
    )
    assert read_out.region_log_probability is None


def test_log_prob_scores_the_read_out_columns_under_the_raised_df() -> None:
    """Scoring is about what the request asks, not about what it already fixed.

    The expected density is the univariate Student-t conditional of Ding
    (2016): location from the Schur complement, ``nu + 1`` degrees of freedom,
    and the Schur complement inflated by ``(nu + d) / (nu + 1)`` with ``d`` the
    pinned value's squared standardized distance.
    """
    correlation, pinned, scored = 0.6, 1.75, 0.25
    read_out = condition_at_position(
        _single_component(correlation), _pin_first(pinned), n_samples=16, column_count=2
    )

    distance = (pinned - _LOCATION[0]) ** 2
    location = _LOCATION[1] + correlation * (pinned - _LOCATION[0])
    scale = (1.0 - correlation**2) * (_DF + distance) / (_DF + 1.0)
    expected = float(student_t_log_density(scored, location, scale, _DF + 1.0))

    values = torch.tensor([[[pinned, scored]]])
    assert float(read_out.log_prob(values)) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("correlation", [-0.5, 0.0, 0.75])
def test_interval_region_probability_matches_the_elliptical_orthant_closed_form(
    correlation: float,
) -> None:
    """A centered two-column orthant is priced exactly at any degrees of freedom."""
    pair = torch.tensor([[1.0, correlation], [correlation, 1.0]])

    read_out = condition_at_position(
        _centered(torch.block_diag(pair, torch.ones(1, 1)), df=5.0),
        ConditionRequest(
            horizon=0,
            interval_columns=(0, 1),
            interval_lower=(0.0, 0.0),
            interval_upper=(float("inf"), float("inf")),
        ),
        n_samples=64,
        rng=torch.Generator().manual_seed(3),
        column_count=3,
    )

    expected = 0.25 + math.asin(correlation) / (2.0 * math.pi)
    assert _region_probability(read_out) == pytest.approx(expected, rel=5e-3)
    assert read_out.equality_log_density is None


def test_region_probability_is_the_t_tail_not_the_normal_one() -> None:
    """One interval column is one t distribution function difference.

    The difference is taken at the head's own degrees of freedom.
    """
    floor = 2.5

    read_out = condition_at_position(
        _single_component(),
        ConditionRequest(
            horizon=0,
            interval_columns=(0,),
            interval_lower=(floor,),
            interval_upper=(float("inf"),),
        ),
        n_samples=64,
        rng=torch.Generator().manual_seed(5),
        column_count=2,
    )

    standardized = floor - _LOCATION[0]
    assert _region_probability(read_out) == pytest.approx(
        float(1.0 - stdtr(_DF, standardized)), abs=EXACT_ATOL
    )
    normal_tail = float(1.0 - torch.special.ndtr(torch.tensor(standardized)))
    assert _region_probability(read_out) > 2.0 * normal_tail
    assert read_out.interval_points is None
    assert read_out.interval_effective_sample_size is None


def test_several_interval_columns_report_the_estimator_accuracy() -> None:
    """A multi-column region is estimated, so its accounting must reach the caller."""
    read_out = condition_at_position(
        _correlated_mixture(),
        ConditionRequest(
            horizon=0,
            interval_columns=(0, 1),
            interval_lower=(0.0, -1.0),
            interval_upper=(float("inf"), 1.0),
        ),
        n_samples=64,
        interval_points=512,
        rng=torch.Generator().manual_seed(5),
        column_count=4,
    )

    assert read_out.interval_points == 512 * 3
    assert read_out.interval_effective_sample_size is not None
    assert (
        0.0 < float(read_out.interval_effective_sample_size) <= read_out.interval_points
    )


def test_samples_respect_every_condition_and_carry_the_pinned_value() -> None:
    """A draw that violated its own request would make the read-out meaningless."""
    request = ConditionRequest(
        horizon=1,
        equality_columns=(0,),
        equality_values=(0.4,),
        interval_columns=(1,),
        interval_lower=(0.75,),
        interval_upper=(2.0,),
    )

    read_out = condition_at_position(
        _correlated_mixture(),
        request,
        n_samples=256,
        rng=torch.Generator().manual_seed(1),
        column_count=4,
    )
    samples = read_out.samples()

    assert tuple(samples.shape) == (1, 256, 1, 4)
    assert torch.allclose(samples[..., 0], torch.tensor(0.4), atol=1e-6)
    assert bool((samples[..., 1] >= 0.75).all() and (samples[..., 1] <= 2.0).all())
    assert read_out.read_out_columns == (2, 3)


def test_quantiles_carry_each_condition_and_center_on_the_conditional_location() -> (
    None
):
    """A symmetric conditional's median is its location.

    Conditioned columns keep what their condition said about them.
    """
    # The read-out's own quantile draws come from the global generator.
    torch.manual_seed(0)
    correlation, pinned = 0.6, 1.75
    read_out = condition_at_position(
        _single_component(correlation),
        _pin_first(pinned),
        n_samples=4000,
        column_count=2,
    )

    quantiles = read_out.quantiles([0.25, 0.5, 0.75])

    assert tuple(quantiles.shape) == (3, 1, 1, 2)
    torch.testing.assert_close(quantiles[..., 0], torch.full((3, 1, 1), pinned))
    location = _LOCATION[1] + correlation * (pinned - _LOCATION[0])
    assert float(quantiles[1, 0, 0, 1]) == pytest.approx(location, abs=ORACLE_ATOL)

    bounded = condition_at_position(
        _correlated_mixture(),
        ConditionRequest(
            horizon=0,
            interval_columns=(1,),
            interval_lower=(0.5,),
            interval_upper=(1.5,),
        ),
        n_samples=256,
        rng=torch.Generator().manual_seed(6),
        column_count=4,
    ).quantiles([0.1, 0.9])
    assert bool((bounded[..., 1] >= 0.5).all() and (bounded[..., 1] <= 1.5).all())


def test_composed_answer_matches_rejection_sampling_on_the_reduced_mixture() -> None:
    """The one case with no closed form is pinned against the sampler it replaces."""
    adapter = _correlated_mixture()
    pinned, floor = 0.3, 0.5
    composed = condition_at_position(
        adapter,
        ConditionRequest(
            horizon=1,
            equality_columns=(0,),
            equality_values=(pinned,),
            interval_columns=(1,),
            interval_lower=(floor,),
            interval_upper=(float("inf"),),
        ),
        n_samples=4000,
        rng=torch.Generator().manual_seed(11),
        column_count=4,
    )
    equality_only = condition_at_position(
        adapter,
        ConditionRequest(horizon=1, equality_columns=(0,), equality_values=(pinned,)),
        n_samples=60_000,
        rng=torch.Generator().manual_seed(3),
        column_count=4,
    )

    draws = equality_only.samples()[0, :, 0, :]
    kept = draws[draws[:, 1] >= floor]
    acceptance = float(kept.shape[0]) / float(draws.shape[0])

    assert composed.read_out_columns == (2, 3)
    assert _region_probability(composed) == pytest.approx(acceptance, abs=0.01)
    for column in composed.read_out_columns:
        assert float(composed.mean()[0, 0, column]) == pytest.approx(
            float(kept[:, column].mean()), abs=ORACLE_ATOL
        )


def test_a_multi_column_tail_region_still_returns_the_full_sample_count() -> None:
    """A stress region stays answerable: every draw lands inside it."""
    read_out = condition_at_position(
        _centered(torch.full((3, 3), 0.5) + torch.eye(3) * 0.5, df=4.0),
        ConditionRequest(
            horizon=0,
            interval_columns=(0, 1),
            interval_lower=(4.0, 4.0),
            interval_upper=(float("inf"), float("inf")),
        ),
        n_samples=500,
        rng=torch.Generator().manual_seed(2),
        column_count=3,
    )
    samples = read_out.samples()

    assert int(samples.shape[1]) == 500
    assert bool(torch.isfinite(samples).all())
    assert bool((samples[..., 0] >= 4.0).all() and (samples[..., 1] >= 4.0).all())
    assert _region_probability(read_out) < 1e-2


def test_a_far_single_column_tail_region_stays_finite() -> None:
    """The exact single-column path inverts the t distribution function far out."""
    params = student_t_pair_mixture(dtype=torch.float32)
    params["means"] = torch.zeros(1, 1, 2, 2)

    read_out = condition_at_position(
        create_student_t_adapter(params),
        ConditionRequest(
            horizon=0,
            interval_columns=(0,),
            interval_lower=(9.0,),
            interval_upper=(float("inf"),),
        ),
        n_samples=256,
        rng=torch.Generator().manual_seed(13),
        column_count=2,
    )

    samples = read_out.samples()
    assert samples.shape == (1, 256, 1, 2)
    assert bool(torch.isfinite(samples).all())
    assert bool((samples[..., 0] >= 9.0).all())
    assert read_out.region_log_probability is not None
    assert float(read_out.region_log_probability) < 0.0


def test_log_prob_under_an_interval_scores_the_pooled_conditional() -> None:
    """The pooled mixture carries ``S * K`` components and must score as one.

    A vanishing interval is the equality condition it surrounds, so the score
    under a narrow interval must approach the closed-form conditional density;
    a scorer that assumed the head's configured component count could not
    reshape the pooled parameters at all.
    """
    adapter = create_student_t_adapter(student_t_pair_mixture(dtype=torch.float32))
    pinned = 0.5
    values = torch.tensor([[[pinned, 0.4]]])

    pooled = condition_at_position(
        adapter,
        ConditionRequest(
            horizon=0,
            interval_columns=(0,),
            interval_lower=(pinned - _NARROW_HALF_WIDTH,),
            interval_upper=(pinned + _NARROW_HALF_WIDTH,),
        ),
        n_samples=64,
        rng=torch.Generator().manual_seed(1),
        column_count=2,
    ).log_prob(values)
    exact = condition_at_position(
        adapter, _pin_first(pinned), n_samples=64, column_count=2
    ).log_prob(values)

    assert pooled.shape == (1, 1)
    assert torch.allclose(pooled, exact, atol=1e-3)


def test_log_prob_rejects_values_contradicting_the_pin() -> None:
    """A density at an excluded point is not a question this surface answers."""
    read_out = condition_at_position(
        _single_component(), _pin_first(1.75), n_samples=16, column_count=2
    )

    with pytest.raises(ValueError, match="contradict the equality condition"):
        read_out.log_prob(torch.tensor([[[9.0, 0.0]]]))


def test_log_prob_scores_values_agreeing_with_the_pin() -> None:
    read_out = condition_at_position(
        _single_component(), _pin_first(1.75), n_samples=16, column_count=2
    )

    assert torch.isfinite(read_out.log_prob(torch.tensor([[[1.75, 0.0]]]))).all()


def test_log_prob_rejects_a_value_outside_its_interval() -> None:
    """The interval branch has the same contradiction as the equality branch."""
    read_out = condition_at_position(
        _correlated_mixture(),
        ConditionRequest(
            horizon=0,
            interval_columns=(0,),
            interval_lower=(1.0,),
            interval_upper=(2.0,),
        ),
        n_samples=32,
        rng=torch.Generator().manual_seed(4),
        column_count=4,
    )

    with pytest.raises(ValueError, match="outside the interval condition"):
        read_out.log_prob(torch.zeros(1, 1, 4))


def test_conditioning_every_column_points_at_the_log_prob_return_mode() -> None:
    """An all-pinned request is a density question the forecast path answers."""
    with pytest.raises(ValueError, match="log_prob"):
        condition_at_position(
            _single_component(),
            ConditionRequest(
                horizon=0, equality_columns=(0, 1), equality_values=(1.0, 2.0)
            ),
            n_samples=16,
            column_count=2,
        )


def test_conditioning_every_column_with_an_interval_leaves_nothing_to_read() -> None:
    """A box over every column describes no conditional distribution."""
    with pytest.raises(ValueError, match="at least one column unconditioned"):
        condition_at_position(
            _single_component(),
            ConditionRequest(
                horizon=0,
                equality_columns=(0,),
                equality_values=(1.0,),
                interval_columns=(1,),
                interval_lower=(0.0,),
                interval_upper=(1.0,),
            ),
            n_samples=16,
            column_count=2,
        )


def test_log_prob_refuses_values_of_another_width() -> None:
    read_out = condition_at_position(
        _single_component(), _pin_first(1.75), n_samples=16, column_count=2
    )

    with pytest.raises(ValueError, match="must carry all 2 columns, got 1"):
        read_out.log_prob(torch.tensor([[[0.0]]]))


def test_an_emitted_key_that_is_no_mixture_parameter_is_refused() -> None:
    """An unknown key is a defect to surface, not a layout to guess at."""
    adapter = _single_component()
    unknown = PredictorAdapter(
        adapter.head, {**adapter.params, "temperature": torch.ones(1, 1, 1)}
    )

    with pytest.raises(ValueError, match="'temperature' is not a mixture parameter"):
        condition_at_position(unknown, _pin_first(1.0), n_samples=16, column_count=2)


def test_a_horizon_outside_the_emitted_positions_is_rejected() -> None:
    """A condition names one emitted position; naming another is not answerable."""
    with pytest.raises(ValueError, match="out of range"):
        condition_at_position(
            _single_component(),
            ConditionRequest(horizon=4, equality_columns=(0,), equality_values=(1.0,)),
            n_samples=16,
            column_count=2,
        )


def test_a_non_positive_draw_budget_is_rejected() -> None:
    """A read-out with no draws behind it has nothing to report."""
    with pytest.raises(ValueError, match="n_samples must be positive"):
        condition_at_position(
            _single_component(), _pin_first(1.0), n_samples=0, column_count=2
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "at least one condition"),
        (
            {"horizon": -1, "equality_columns": (0,), "equality_values": (1.0,)},
            "negative",
        ),
        (
            {"horizon": 0, "equality_columns": (0,), "equality_values": (1.0, 2.0)},
            "same length",
        ),
        (
            {
                "horizon": 0,
                "equality_columns": (1,),
                "equality_values": (1.0,),
                "interval_columns": (1,),
                "interval_lower": (0.0,),
                "interval_upper": (1.0,),
            },
            "at most one condition",
        ),
        (
            {
                "horizon": 0,
                "equality_columns": (0,),
                "equality_values": (float("inf"),),
            },
            "finite value",
        ),
        (
            {
                "horizon": 0,
                "interval_columns": (0,),
                "interval_lower": (1.0,),
                "interval_upper": (0.0,),
            },
            "lower < upper",
        ),
        (
            {
                "horizon": 0,
                "interval_columns": (0,),
                "interval_lower": (-float("inf"),),
                "interval_upper": (float("inf"),),
            },
            "open on both sides",
        ),
    ],
)
def test_a_malformed_request_is_rejected_before_any_head_is_touched(
    kwargs: dict, message: str
) -> None:
    """Request validation is head-independent, so it belongs to the request object."""
    with pytest.raises(ValueError, match=message):
        ConditionRequest(**kwargs)


def _mixed_request() -> ConditionRequest:
    """Pin one column and bound another, reading out the remaining two."""
    return ConditionRequest(
        horizon=1,
        equality_columns=(0,),
        equality_values=(0.3,),
        interval_columns=(1,),
        interval_lower=(-0.5,),
        interval_upper=(1.0,),
    )


def test_the_diagonal_plus_factor_form_answers_like_its_dense_scale() -> None:
    """Both covariance parameterizations of one mixture give one answer."""
    generator = torch.Generator().manual_seed(9)
    width, components, rank = 4, 2, 1
    logits = torch.randn(1, 2, components, generator=generator)
    means = torch.randn(1, 2, components, width, generator=generator)
    lowrank_diag = torch.rand(1, 2, components, width, generator=generator) + 0.5
    lowrank_U = torch.randn(1, 2, components, width, rank, generator=generator)
    df = torch.tensor([6.0, 10.0]).expand(1, 2, components).clone()
    lowrank = PredictorAdapter(
        student_t_head(D=width, K=components, covariance_rank=rank),
        {
            "logits": logits,
            "means": means,
            "lowrank_diag": lowrank_diag,
            "lowrank_U": lowrank_U,
            "df": df,
        },
    )
    scale = torch.diag_embed(lowrank_diag) + lowrank_U @ lowrank_U.transpose(-1, -2)
    dense = create_student_t_adapter(
        {
            "logits": logits,
            "means": means,
            "chol": torch.linalg.cholesky(scale),
            "df": df,
        }
    )

    answers = [
        condition_at_position(
            adapter,
            _mixed_request(),
            n_samples=128,
            rng=torch.Generator().manual_seed(2),
            column_count=width,
        )
        for adapter in (lowrank, dense)
    ]

    from_lowrank, from_dense = answers
    assert from_lowrank.read_out_columns == from_dense.read_out_columns == (2, 3)
    torch.testing.assert_close(
        from_lowrank.mean(), from_dense.mean(), atol=EXACT_ATOL, rtol=0.0
    )
    for answer in answers:
        assert answer.equality_log_density is not None
        assert answer.region_log_probability is not None
    torch.testing.assert_close(
        from_lowrank.equality_log_density,
        from_dense.equality_log_density,
        atol=EXACT_ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        from_lowrank.region_log_probability,
        from_dense.region_log_probability,
        atol=EXACT_ATOL,
        rtol=0.0,
    )


def test_a_per_column_bound_applies_to_its_read_out_column() -> None:
    """The caller's bounds cover every column; the read-out indexes them."""
    lower = torch.tensor([float("-inf"), -0.2, float("-inf"), float("-inf")])

    read_out = condition_at_position(
        _correlated_mixture(),
        ConditionRequest(horizon=0, equality_columns=(0,), equality_values=(0.3,)),
        n_samples=64,
        rng=torch.Generator().manual_seed(4),
        min_value=lower,
        column_count=4,
    )

    samples = read_out.samples()
    assert samples.shape == (1, 64, 1, 4)
    assert bool((samples[..., 1] >= -0.2).all())


@pytest.mark.parametrize("column_count", [0, 3, 5])
def test_a_request_width_other_than_the_emitted_width_is_rejected(
    column_count: int,
) -> None:
    """The request's width is checked against the emitted joint, not trusted."""
    with pytest.raises(
        RuntimeError,
        match=f"cover 4 columns, the request declares {column_count}",
    ):
        condition_at_position(
            _correlated_mixture(),
            _pin_first(0.3),
            n_samples=8,
            column_count=column_count,
        )


def test_a_condition_beyond_the_request_width_is_rejected() -> None:
    """A condition must name one of the position's columns."""
    with pytest.raises(ValueError, match="has 4 columns"):
        condition_at_position(
            _correlated_mixture(),
            ConditionRequest(horizon=0, equality_columns=(4,), equality_values=(0.3,)),
            n_samples=8,
            column_count=4,
        )


def test_a_seeded_request_returns_reproducible_quantiles() -> None:
    def answer() -> ConditionalReadOut:
        return condition_at_position(
            _correlated_mixture(),
            _pin_first(1.75),
            n_samples=500,
            rng=torch.Generator().manual_seed(0),
            column_count=4,
        )

    torch.manual_seed(1)
    first = answer().quantiles([0.1, 0.5, 0.9])
    torch.manual_seed(2)
    second = answer().quantiles([0.1, 0.5, 0.9])
    assert torch.equal(first, second)
