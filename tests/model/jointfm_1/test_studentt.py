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

"""Contract tests for the JointFM 1 Student-t mixture head."""

from typing import Any

import pytest
import torch
from scipy.stats import multivariate_t

from jointfm_inference.model.jointfm_1.head import (
    AUTO_STANDARDIZATION_CENTER_KEY,
    AUTO_STANDARDIZATION_SCALE_KEY,
    DF_EPS,
    MAX_DF,
    StudentTHead,
)
from jointfm_inference.model.linalg import cholesky_from_lowrank

from tests.support.inference import student_t_head

B, H, D, K, RANK, D_MODEL = 2, 3, 4, 2, 2, 8


def _head(**overrides: Any) -> StudentTHead:
    """Build a small head; ``overrides`` replace the default sizes."""
    sizes: dict[str, Any] = {
        "D": D,
        "K": K,
        "d_model": D_MODEL,
        "covariance_rank": RANK,
    }
    sizes.update(overrides)
    return student_t_head(**sizes)


def _params(head: StudentTHead) -> dict[str, torch.Tensor]:
    """Run the head on a random pooled context and random column tokens."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(B, H, D_MODEL, generator=generator)
    tokens = torch.randn(B, H, head.D, D_MODEL, generator=generator)
    with torch.no_grad():
        return head(x, column_tokens=tokens)


def _lowrank_mixture(batch: int = 1) -> dict[str, torch.Tensor]:
    """Return a hand-built two-component, two-column, rank-one mixture.

    Every one of the ``batch`` rows carries the same mixture, so one call
    scores several points under it.
    """
    params = {
        "logits": torch.tensor([[[0.3, -0.2]]]),
        "means": torch.tensor([[[[0.5, -1.0], [2.0, 0.0]]]]),
        "df": torch.tensor([[[4.0, 9.0]]]),
        "lowrank_diag": torch.tensor([[[[0.5, 1.5], [2.0, 0.3]]]]),
        "lowrank_U": torch.tensor([[[[[0.8], [-0.4]], [[0.1], [1.2]]]]]),
    }
    return {
        name: value.expand(batch, *value.shape[1:]).contiguous()
        for name, value in params.items()
    }


def test_forward_emits_the_documented_shapes_and_floors() -> None:
    head = _head(min_df=4.5)
    params = _params(head)

    assert params.keys() == {"logits", "means", "df", "lowrank_diag", "lowrank_U"}
    assert params["logits"].shape == (B, H, K)
    assert params["means"].shape == (B, H, K, D)
    assert params["df"].shape == (B, H, K)
    assert params["lowrank_diag"].shape == (B, H, K, D)
    assert params["lowrank_U"].shape == (B, H, K, D, RANK)
    assert (params["df"] > 4.5).all()
    assert (params["lowrank_diag"] >= 1e-4).all()


def test_degrees_of_freedom_are_the_floor_plus_the_exponential_of_the_readout() -> None:
    """A near-Gaussian component sits at a raw output near the log of its excess."""
    min_df = 4.5
    head = _head(min_df=min_df)
    excess = torch.tensor([2.0, 98.0])
    with torch.no_grad():
        head.out_log_df.weight.zero_()
        head.out_log_df.bias.copy_(torch.log(excess))

    expected = (min_df + DF_EPS + excess).expand(B, H, K)
    torch.testing.assert_close(_params(head)["df"], expected)


def test_a_state_dict_of_the_softplus_df_readout_fails_strict_loading() -> None:
    head = _head()
    state = {
        key.replace("out_log_df", "out_df"): value
        for key, value in head.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="out_log_df"):
        _head().load_state_dict(state, strict=True)


def test_log_prob_matches_the_closed_form_student_t_mixture() -> None:
    params = _lowrank_mixture(batch=2)
    y = torch.tensor([[[0.1, 0.4]], [[3.0, -2.0]]])

    log_prob = _head(D=2, K=2, covariance_rank=1).log_prob(params, y)

    assert log_prob.shape == (2, 1)
    weights = torch.softmax(params["logits"][0, 0], dim=-1).double()
    expected = []
    for point in y[:, 0].double().numpy():
        density = 0.0
        for k in range(2):
            U = params["lowrank_U"][0, 0, k].double()
            shape = torch.diag(params["lowrank_diag"][0, 0, k].double()) + U @ U.T
            component = multivariate_t(
                loc=params["means"][0, 0, k].double().numpy(),
                shape=shape.numpy(),
                df=float(params["df"][0, 0, k]),
            )
            density += float(weights[k]) * float(component.pdf(point))
        expected.append(torch.tensor(density).log())
    torch.testing.assert_close(
        log_prob.reshape(-1), torch.stack(expected).float(), atol=1e-4, rtol=1e-4
    )


def test_log_prob_accepts_a_dense_cholesky_mapping() -> None:
    """A conditioned mixture carries ``chol``; it scores like the low-rank form."""
    params = _lowrank_mixture(batch=5)
    dense = {name: params[name] for name in ("logits", "means", "df")}
    dense["chol"] = cholesky_from_lowrank(params["lowrank_diag"], params["lowrank_U"])
    head = _head(D=2, K=2, covariance_rank=1)
    y = torch.randn(5, 1, 2, generator=torch.Generator().manual_seed(2))

    torch.testing.assert_close(head.log_prob(dense, y), head.log_prob(params, y))


def test_log_prob_scores_every_row_independently() -> None:
    head = _head()
    params = _params(head)
    y = torch.randn(B, H, D, generator=torch.Generator().manual_seed(3))

    full = head.log_prob(params, y)

    assert full.shape == (B, H)
    one_row = {name: value[1:2, 2:3] for name, value in params.items()}
    torch.testing.assert_close(head.log_prob(one_row, y[1:2, 2:3]), full[1:2, 2:3])


def test_sample_is_batch_major_and_reproducible() -> None:
    head = _head()
    params = _params(head)

    first = head.sample(params, num_samples=6, rng=torch.Generator().manual_seed(5))
    second = head.sample(params, num_samples=6, rng=torch.Generator().manual_seed(5))

    assert first.shape == (B * 6, H, D)
    torch.testing.assert_close(first, second)


def test_sample_refuses_non_finite_logits() -> None:
    params = _lowrank_mixture()
    params["logits"] = torch.full_like(params["logits"], float("nan"))
    with pytest.raises(RuntimeError, match="invalid mixture probabilities"):
        _head(D=2, K=2, covariance_rank=1).sample(params, num_samples=2)


def test_mean_and_quantiles_track_the_mixture_location() -> None:
    params = _lowrank_mixture()
    head = _head(D=2, K=2, covariance_rank=1)
    weights = torch.softmax(params["logits"], dim=-1)
    expected = (weights.unsqueeze(-1) * params["means"]).sum(dim=-2)

    mean = head.mean(params, num_samples=20_000, rng=torch.Generator().manual_seed(6))
    assert mean.shape == (1, 1, 2)
    torch.testing.assert_close(mean, expected, atol=0.06, rtol=0.0)

    quantiles = head.quantiles(
        params,
        q=[0.1, 0.5, 0.9],
        num_samples=2000,
        rng=torch.Generator().manual_seed(7),
    )
    assert quantiles.shape == (3, 1, 1, 2)
    assert (quantiles[0] < quantiles[1]).all() and (quantiles[1] < quantiles[2]).all()


def test_degrees_of_freedom_stay_bounded_under_extreme_logits() -> None:
    head = _head()
    with torch.no_grad():
        head.out_log_df.weight.fill_(1e4)
        head.out_log_df.bias.fill_(1e4)
    params = _params(head)

    assert torch.isfinite(params["df"]).all()
    assert (params["df"] <= MAX_DF).all()
    y = torch.randn(B, H, D)
    assert torch.isfinite(head.log_prob(params, y)).all()


def test_covariance_and_draws_stay_finite_under_extreme_weights() -> None:
    head = _head()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.fill_(200.0)
    params = _params(head)

    assert torch.isfinite(params["lowrank_diag"]).all()
    assert torch.isfinite(head.sample(params, num_samples=32)).all()


def test_many_draws_stay_finite() -> None:
    """Near-zero chi-square draws would otherwise blow a draw up to infinity."""
    head = _head(D=6, K=3, min_df=2.0)
    samples = head.sample(_params(head), num_samples=512)
    assert samples.shape == (B * 512, H, 6)
    assert torch.isfinite(samples).all()


def test_bf16_log_prob_stays_finite_for_large_values() -> None:
    head = _head().to(dtype=torch.bfloat16)
    x = torch.randn(B, H, D_MODEL, dtype=torch.bfloat16)
    tokens = torch.randn(B, H, D, D_MODEL, dtype=torch.bfloat16)
    with torch.no_grad():
        params = head(x, column_tokens=tokens)
    y = 100.0 * torch.randn(B, H, D, dtype=torch.bfloat16)

    log_prob = head.log_prob(params, y)

    assert log_prob.dtype == torch.bfloat16
    assert torch.isfinite(log_prob.float()).all()


def test_quantiles_read_data_space_parameters_without_rescaling() -> None:
    """The standardization statistics attached to a forward are metadata only.

    The network already maps locations and scales back to data space, so a
    read-out that applied ``center + scale * draw`` again would move the
    quantiles far from the mean and the draws.
    """
    center, scale = 7355.0, 100.0
    params = {
        "logits": torch.zeros(1, 1, 2),
        "means": torch.tensor([[[[7350.0], [7360.0]]]]),
        "chol": torch.full((1, 1, 2, 1, 1), scale),
        "df": torch.full((1, 1, 2), 20.0),
        AUTO_STANDARDIZATION_CENTER_KEY: torch.tensor([[[center]]]),
        AUTO_STANDARDIZATION_SCALE_KEY: torch.tensor([[[scale]]]),
    }
    head = _head(D=1, K=2, covariance_rank=1)

    mean = float(
        head.mean(params, num_samples=4000, rng=torch.Generator().manual_seed(11))
    )
    median = float(
        head.quantiles(
            params, q=[0.5], num_samples=4000, rng=torch.Generator().manual_seed(7)
        )
    )

    assert abs(mean - center) < 10.0
    assert abs(median - center) < 10.0
