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

"""Unit tests for sample rejection bound helpers used by predictor sampling."""

from typing import cast

import pytest
import torch

from jointfm_inference.model.inference._bounds import (
    _bound_error_fragment,
    _normalize_sample_bound,
    _sample_constraint_violations,
)


def _samples() -> torch.Tensor:
    return torch.zeros(2, 3, 4)


def test_normalize_sample_bound_none_passthrough() -> None:
    assert _normalize_sample_bound(None, samples=_samples(), field="min_value") is None


def test_normalize_sample_bound_rejects_bool() -> None:
    with pytest.raises(TypeError, match="min_value must be numeric or a tensor"):
        _normalize_sample_bound(True, samples=_samples(), field="min_value")


def test_normalize_sample_bound_rejects_empty_tensor() -> None:
    with pytest.raises(ValueError, match="max_value must not be empty"):
        _normalize_sample_bound(
            torch.zeros(0),
            samples=_samples(),
            field="max_value",
        )


def test_normalize_sample_bound_rejects_non_numeric() -> None:
    with pytest.raises(TypeError, match="max_value must be numeric or a tensor"):
        _normalize_sample_bound(
            cast(float, "high"),
            samples=_samples(),
            field="max_value",
        )


def test_normalize_sample_bound_rejects_nan() -> None:
    with pytest.raises(ValueError, match="min_value must not contain NaNs"):
        _normalize_sample_bound(float("nan"), samples=_samples(), field="min_value")

    with pytest.raises(ValueError, match="min_value must not contain NaNs"):
        _normalize_sample_bound(
            torch.tensor([0.0, float("nan"), 1.0, 2.0]),
            samples=_samples(),
            field="min_value",
        )


def test_normalize_sample_bound_rejects_non_broadcastable_shape() -> None:
    with pytest.raises(ValueError, match=r"broadcast to sample shape \(2, 3, 4\)"):
        _normalize_sample_bound(
            torch.zeros(5),
            samples=_samples(),
            field="min_value",
        )


def test_normalize_sample_bound_matches_sample_dtype_and_broadcasts() -> None:
    samples = _samples()

    scalar = _normalize_sample_bound(2, samples=samples, field="min_value")
    assert scalar is not None
    assert scalar.dtype == samples.dtype
    assert scalar.device == samples.device
    assert float(scalar.item()) == 2.0

    per_column = _normalize_sample_bound(
        torch.arange(4, dtype=torch.float64),
        samples=samples,
        field="max_value",
    )
    assert per_column is not None
    assert per_column.dtype == samples.dtype
    assert tuple(per_column.shape) == (4,)


def test_sample_constraint_violations_flags_nan_and_bound_breaches() -> None:
    samples = torch.tensor(
        [
            [[0.5, 0.5], [float("nan"), 0.5], [-1.0, 0.5], [0.5, 2.0]],
        ]
    )
    is_bad = _sample_constraint_violations(
        samples,
        min_value=torch.tensor(0.0),
        max_value=torch.tensor(1.0),
    )
    assert is_bad.tolist() == [[False, True, True, True]]

    unbounded = _sample_constraint_violations(samples, min_value=None, max_value=None)
    assert unbounded.tolist() == [[False, True, False, False]]


def test_bound_error_fragment_reports_scalar_value_and_generic_tensor() -> None:
    assert _bound_error_fragment("min_value", torch.tensor(1.5)) == (
        " min_value=1.5 violation."
    )
    assert _bound_error_fragment("max_value", torch.tensor([1.0, 2.0])) == (
        " max_value violation."
    )
