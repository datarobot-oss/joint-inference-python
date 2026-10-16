"""Contract tests for the model compute precision of inference."""

from __future__ import annotations

import pytest
import torch

from jointfm_inference.model.precision import (
    autocast_context,
    resolve_autocast_dtype,
    upcast_reduced_precision,
)

_CPU = torch.device("cpu")


def test_fp32_resolves_to_no_autocast_on_any_device() -> None:
    assert resolve_autocast_dtype("fp32", _CPU, setting="s") is None
    assert resolve_autocast_dtype("fp32", torch.device("mps"), setting="s") is None


def test_bf16_resolves_on_cpu() -> None:
    assert resolve_autocast_dtype("bf16", _CPU, setting="s") is torch.bfloat16


@pytest.mark.parametrize(
    "precision,device,match",
    [
        ("tf32", _CPU, "Expected s to be one of"),
        ("BF16", _CPU, "Expected s to be one of"),
        ("fp16", _CPU, "'fp16' only on CUDA"),
        ("bf16", torch.device("mps"), "only on cpu or cuda"),
    ],
)
def test_unservable_precision_fails_instead_of_degrading(
    precision: str, device: torch.device, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_autocast_dtype(precision, device, setting="s")


def test_autocast_context_enables_only_reduced_precision() -> None:
    with autocast_context("cpu", None):
        assert not torch.is_autocast_enabled("cpu")
    with autocast_context("cpu", torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") is torch.bfloat16


def test_upcast_reduced_precision_casts_only_bf16_and_fp16() -> None:
    tensors = {
        "bf16": torch.ones(2, dtype=torch.bfloat16),
        "fp16": torch.ones(2, dtype=torch.float16),
        "fp64": torch.ones(2, dtype=torch.float64),
        "index": torch.arange(2),
    }

    upcast = upcast_reduced_precision(tensors)

    assert upcast["bf16"].dtype is torch.float32
    assert upcast["fp16"].dtype is torch.float32
    assert upcast["fp64"] is tensors["fp64"]
    assert upcast["index"] is tensors["index"]
