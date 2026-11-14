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

"""Mapped Student-t mixtures and float64 oracles of their law for tests.

The oracles are written from the definitions with NumPy and SciPy, apart from
the module under test: the map ``a + s * sinh(delta * asinh(u)) / delta``, its
derivative ``cosh(delta * asinh(u)) / sqrt(1 + u^2)``, and the latent
mixture's density from ``scipy.stats.multivariate_t``.
"""

from typing import Any

import numpy as np
import torch
from scipy import stats
from scipy.special import logsumexp, stdtr

from jointfm_inference.model.jointfm_1.marginals import attach_column_anchors
from jointfm_inference.model.jointfm_1.tail_map import (
    ANCHOR_LOCATION_KEY,
    ANCHOR_SCALE_KEY,
    TAIL_LOWER_KEY,
    TAIL_UPPER_KEY,
)


def mapped_mixture(*, dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    """Return a ``(B=1, H=1, K=2, D=3)`` mixture with a two-sided map on every column.

    The components differ in location, scale, correlation sign, and degrees
    of freedom, and every column has unequal upper and lower tail parameters,
    one heavier than the identity and one lighter, so a swapped side or a
    missing Jacobian moves every read-out.
    """
    covariance = torch.stack(
        [
            torch.tensor(
                [[1.0, 0.6, 0.2], [0.6, 1.5, -0.3], [0.2, -0.3, 0.8]], dtype=dtype
            ),
            torch.tensor(
                [[2.5, -0.9, 0.4], [-0.9, 1.2, 0.1], [0.4, 0.1, 3.0]], dtype=dtype
            ),
        ]
    )
    params = {
        "logits": torch.tensor([[[0.4, -0.3]]], dtype=dtype),
        "means": torch.tensor([[[[0.0, 1.0, -0.5], [1.5, -0.5, 0.8]]]], dtype=dtype),
        "chol": torch.linalg.cholesky(covariance).reshape(1, 1, 2, 3, 3),
        "df": torch.tensor([[[5.0, 8.0]]], dtype=dtype),
    }
    params = attach_column_anchors(params)
    params[TAIL_UPPER_KEY] = torch.tensor([[[1.35, 0.7, 1.2]]], dtype=dtype)
    params[TAIL_LOWER_KEY] = torch.tensor([[[0.6, 1.3, 0.9]]], dtype=dtype)
    return params


def _numpy(params: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Return the first position's parameters as float64 arrays."""
    return {name: value[0, 0].double().numpy() for name, value in params.items()}


def _sides(standardized: Any, upper: Any, lower: Any) -> Any:
    """Return each cell's tail parameter by the sign of its standardized coordinate."""
    return np.where(standardized >= 0, upper, lower)


def oracle_latent(params: dict[str, torch.Tensor], values: Any) -> Any:
    """Return the latent images of data values ``(..., D)`` at the first position."""
    p = _numpy(params)
    location, scale = p[ANCHOR_LOCATION_KEY], p[ANCHOR_SCALE_KEY]
    standardized = (np.asarray(values, dtype=np.float64) - location) / scale
    tail = _sides(standardized, p[TAIL_UPPER_KEY], p[TAIL_LOWER_KEY])
    return location + scale * np.sinh(np.arcsinh(tail * standardized) / tail)


def oracle_data(params: dict[str, torch.Tensor], latent: Any) -> Any:
    """Return the data values of latent coordinates ``(..., D)`` at the first position."""
    p = _numpy(params)
    location, scale = p[ANCHOR_LOCATION_KEY], p[ANCHOR_SCALE_KEY]
    standardized = (np.asarray(latent, dtype=np.float64) - location) / scale
    tail = _sides(standardized, p[TAIL_UPPER_KEY], p[TAIL_LOWER_KEY])
    return location + scale * np.sinh(tail * np.arcsinh(standardized)) / tail


def oracle_log_derivative(params: dict[str, torch.Tensor], values: Any) -> Any:
    """Return ``log g'(u)`` per column at data values, from the derivative's definition."""
    p = _numpy(params)
    location, scale = p[ANCHOR_LOCATION_KEY], p[ANCHOR_SCALE_KEY]
    latent = (oracle_latent(params, values) - location) / scale
    tail = _sides(latent, p[TAIL_UPPER_KEY], p[TAIL_LOWER_KEY])
    return np.log(np.cosh(tail * np.arcsinh(latent))) - 0.5 * np.log1p(latent**2)


def oracle_log_prob(
    params: dict[str, torch.Tensor], values: Any, columns: Any = None
) -> float:
    """Return the mapped mixture's log-density of ``values`` over ``columns`` (all by default)."""
    p = _numpy(params)
    columns = (
        np.arange(p["means"].shape[-1]) if columns is None else np.asarray(columns)
    )
    values = np.asarray(values, dtype=np.float64)
    full = np.zeros(p["means"].shape[-1])
    full[columns] = values
    latent = oracle_latent(params, full)[columns]
    jacobian = oracle_log_derivative(params, full)[columns].sum()
    log_weights = p["logits"] - logsumexp(p["logits"])
    component = [
        stats.multivariate_t.logpdf(
            latent,
            loc=p["means"][k][columns],
            shape=(p["chol"][k] @ p["chol"][k].T)[np.ix_(columns, columns)],
            df=p["df"][k],
        )
        for k in range(p["means"].shape[0])
    ]
    return float(logsumexp(log_weights + np.asarray(component)) - jacobian)


def oracle_column_cdf(params: dict[str, torch.Tensor], column: int, value: Any) -> Any:
    """Return column ``column``'s marginal distribution function at data values."""
    p = _numpy(params)
    values = np.asarray(value, dtype=np.float64)
    full = np.zeros(values.shape + (p["means"].shape[-1],))
    full[..., column] = values
    latent = oracle_latent(params, full)[..., column]
    weights = np.exp(p["logits"] - logsumexp(p["logits"]))
    scales = np.sqrt((p["chol"] ** 2).sum(axis=-1))[:, column]
    return sum(
        weights[k] * stdtr(p["df"][k], (latent - p["means"][k, column]) / scales[k])
        for k in range(weights.shape[0])
    )


def oracle_column_density(
    params: dict[str, torch.Tensor], column: int, value: float
) -> float:
    """Return column ``column``'s marginal density at one data value."""
    p = _numpy(params)
    full = np.zeros(p["means"].shape[-1])
    full[column] = value
    latent = oracle_latent(params, full)[column]
    log_derivative = oracle_log_derivative(params, full)[column]
    weights = np.exp(p["logits"] - logsumexp(p["logits"]))
    scales = np.sqrt((p["chol"] ** 2).sum(axis=-1))[:, column]
    latent_density = sum(
        weights[k]
        * stats.t.pdf(latent, p["df"][k], loc=p["means"][k, column], scale=scales[k])
        for k in range(weights.shape[0])
    )
    return float(latent_density * np.exp(-log_derivative))
