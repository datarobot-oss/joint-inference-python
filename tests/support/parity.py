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

"""Parity of a head's exact summaries with Monte Carlo draws from its own sampler.

An exact mean or quantile is a deterministic evaluation, so the only
independent check of it is the head's own sampler: a large draw from ``sample``
estimates both summaries, and the estimate's standard error says how far the
exact value may lie from it. Every comparison runs on the parameters as the
network emits them, in data units, so a summary that rescales them a second
time lands many standard errors away and fails.
"""

import math
from collections.abc import Sequence
from typing import Protocol

import torch
from scipy import stats
from torch import Tensor

from jointfm_inference.model.jointfm_1.marginals import component_marginal_scales

# Standard errors an exact summary may lie from its Monte Carlo estimate. Wide
# enough that a pinned draw of a correct head passes in every cell of a small
# table, narrow enough that a rescaled summary does not.
PARITY_STANDARD_ERRORS = 5.0


class ExactSummaryHead(Protocol):
    """The head operations parity compares: the sampler and the exact summaries."""

    def sample(
        self,
        params: dict[str, Tensor],
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Draw ``(B * num_samples, H, D)`` samples, batch-major."""
        ...

    def mean(self, params: dict[str, Tensor]) -> Tensor:
        """Return the exact ``(B, H, D)`` mean."""
        ...

    def quantiles(self, params: dict[str, Tensor], q: Sequence[float]) -> Tensor:
        """Return exact ``(Q, B, H, D)`` marginal quantiles."""
        ...


def _marginal_density(params: dict[str, Tensor], values: Tensor) -> Tensor:
    """Return the ``(Q, B, H, D)`` marginal mixture density at ``values``."""
    weights = torch.softmax(params["logits"].double(), dim=-1)
    locations = params["means"].double()
    scales = component_marginal_scales(params).double()
    standardized = (values.unsqueeze(-2) - locations) / scales
    df = params["df"].double().unsqueeze(-1).expand_as(standardized)
    component = torch.from_numpy(stats.t.pdf(standardized.numpy(), df.numpy()))
    return (weights.unsqueeze(-1) * component / scales).sum(dim=-2)


def assert_exact_summaries_match_draws(
    head: ExactSummaryHead,
    params: dict[str, Tensor],
    *,
    q: Sequence[float],
    draw_count: int,
    rng: torch.Generator,
) -> None:
    """Assert that ``head.mean`` and ``head.quantiles`` agree with ``draw_count`` draws.

    The mean's standard error is the draws' standard deviation over the square
    root of the count, which needs every component above two degrees of
    freedom. A sample quantile's standard error is
    ``sqrt(q (1 - q) / n) / f(x_q)``, with ``f`` the marginal mixture density.
    It is read at the Monte Carlo estimate rather than at the exact value under
    test, because a wrong exact quantile far in a tail would meet a density near
    zero there and an error bar wide enough to hide it.

    Raises:
        AssertionError: when some cell lies more than
            :data:`PARITY_STANDARD_ERRORS` standard errors from its estimate.
    """
    if not bool((params["df"] > 2.0).all()):
        raise ValueError("parity of the mean needs every component above two df")
    batch, horizons, _components = params["logits"].shape
    draws = head.sample(params, num_samples=draw_count, rng=rng).double()
    draws = draws.reshape(batch, draw_count, horizons, draws.shape[-1])

    mean_error = (head.mean(params).double() - draws.mean(dim=1)).abs()
    mean_standard_error = draws.std(dim=1) / math.sqrt(draw_count)
    worst_mean = float((mean_error / mean_standard_error).max())
    assert worst_mean <= PARITY_STANDARD_ERRORS, (
        f"exact mean lies {worst_mean:.1f} standard errors from {draw_count} draws"
    )

    levels = torch.tensor(list(q), dtype=torch.float64)
    exact = head.quantiles(params, q).double()
    estimated = torch.quantile(draws, levels, dim=1)
    level_axis = levels.reshape(-1, 1, 1, 1)
    quantile_standard_error = torch.sqrt(
        level_axis * (1.0 - level_axis) / draw_count
    ) / _marginal_density(params, estimated)
    worst_quantile = float(((exact - estimated).abs() / quantile_standard_error).max())
    assert worst_quantile <= PARITY_STANDARD_ERRORS, (
        f"exact quantile lies {worst_quantile:.1f} standard errors from "
        f"{draw_count} draws"
    )
