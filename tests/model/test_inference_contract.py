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

"""Tests for the backend-neutral inference contract.

Covers the canonical request invariants, runtime conformance of the JointFM 1
backend and its distribution result to the contract protocols, and exact
equivalence of every result operation with the direct ``PredictorAdapter``
call it wraps.
"""

import pytest
import torch

from jointfm_inference.data.types import CanonicalSample
from jointfm_inference.model.checkpoint import JOINTFM_1_GENERATION
from jointfm_inference.model.inference import (
    INFERENCE_RETURN_MODES,
    CanonicalInferenceRequest,
    ConditionalDistributionResult,
    ConditionRequest,
    InferenceBackend,
    InferenceDistributionResult,
    InferenceRequestError,
    JointFM1DistributionResult,
    PredictorAdapter,
    ServingBackend,
    condition_at_position,
    normalize_head_name,
)
from jointfm_inference.model.inference.adapter import DEFAULT_MEAN_SAMPLES
from jointfm_inference.model.inference.jointfm_1_backend import (
    JOINTFM_1_SERVING_BACKEND,
    JointFM1InferenceBackend,
)

from tests.support.inference import create_student_t_adapter
from tests.support.models import checkpoint_mapping, tiny_model

_STUDENT_T_DF = 6.0


def _canonical_sample(*, query_rows: bool = True) -> CanonicalSample:
    """Build a three-column sample whose last two rows are query rows."""
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    query_row_mask = torch.zeros(5, dtype=torch.bool)
    if query_rows:
        query_row_mask[-2:] = True
        values[query_row_mask] = 0.0
    return CanonicalSample(
        values=values,
        time_positions=torch.arange(5, dtype=torch.float64),
        time_index_mode="continuous_float",
        query_row_mask=query_row_mask,
        metadata={"column_names": ["feature_0", "feature_1", "target"]},
    )


def _student_t_adapter(*, H: int = 2, K: int = 2, D: int = 3) -> PredictorAdapter:
    """Build a batched Student-t mixture adapter with well-separated components."""
    logits = torch.zeros(1, H, K)
    logits[..., 0] = 2.0
    means = torch.stack([torch.full((1, H, D), float(k)) for k in range(K)], dim=2)
    chol = torch.eye(D).view(1, 1, 1, D, D).expand(1, H, K, D, D).clone() * 0.1
    return create_student_t_adapter(
        {
            "logits": logits,
            "means": means,
            "chol": chol,
            "df": torch.full((1, H, K), _STUDENT_T_DF),
        }
    )


def _mounted_backend() -> JointFM1InferenceBackend:
    """Mount a tiny network the way a session does, at full precision."""
    model = tiny_model()
    return JOINTFM_1_SERVING_BACKEND.mount(
        model,
        model.cfg,
        checkpoint_mapping(model),
        device=torch.device("cpu"),
        autocast_dtype=None,
    )


class TestCanonicalInferenceRequest:
    def test_valid_request_derivations(self) -> None:
        request = CanonicalInferenceRequest(sample=_canonical_sample())
        expected_rows = torch.tensor([False, False, False, True, True])
        assert torch.equal(request.query_row_mask, expected_rows)
        assert request.column_count == 3

    def test_rejects_sample_without_query_rows(self) -> None:
        with pytest.raises(ValueError, match="at least one query row"):
            CanonicalInferenceRequest(sample=_canonical_sample(query_rows=False))

    def test_rejects_non_canonical_sample(self) -> None:
        with pytest.raises(TypeError, match="CanonicalSample"):
            CanonicalInferenceRequest(sample={"values": torch.zeros(2, 2)})  # ty: ignore[invalid-argument-type]


class TestContractConformance:
    def test_backend_protocols_are_runtime_checkable(self) -> None:
        assert isinstance(JOINTFM_1_SERVING_BACKEND, ServingBackend)
        assert not isinstance(object(), ServingBackend)
        assert not isinstance(object(), InferenceBackend)

    def test_the_mounted_backend_reports_what_it_serves(self) -> None:
        backend = _mounted_backend()

        assert isinstance(backend, InferenceBackend)
        assert backend.generation == JOINTFM_1_GENERATION
        assert backend.head_name == "studentt"
        assert backend.supported_return_modes == INFERENCE_RETURN_MODES
        assert backend.supports_conditioning is True
        assert backend.metadata_traits == ()

    def test_execute_returns_a_result_that_conditions(self) -> None:
        """``supports_conditioning`` promises a conditional result on every request."""
        result = _mounted_backend().execute(
            CanonicalInferenceRequest(sample=_canonical_sample())
        )

        assert isinstance(result, ConditionalDistributionResult)
        assert result.samples(n_samples=3).shape == (1, 3, 2, 3)

    def test_head_name_drops_head_suffix_and_underscores(self) -> None:
        assert normalize_head_name("StudentTHead") == "studentt"
        assert normalize_head_name("Dummy_OtherHead") == "dummyother"

    def test_request_error_names_the_responsible_field(self) -> None:
        error = InferenceRequestError("too long", field="history_rows")
        assert isinstance(error, ValueError)
        assert error.field == "history_rows"
        assert InferenceRequestError("unattributed").field is None


class TestJointFM1DistributionResult:
    def test_satisfies_the_conditional_result_protocol(self) -> None:
        result = JointFM1DistributionResult(_student_t_adapter(), active_column_count=3)
        assert isinstance(result, InferenceDistributionResult)
        assert isinstance(result, ConditionalDistributionResult)

    def test_rejects_non_adapter(self) -> None:
        with pytest.raises(TypeError, match="PredictorAdapter"):
            JointFM1DistributionResult(object(), active_column_count=1)  # ty: ignore[invalid-argument-type]

    def test_rejects_non_positive_active_column_count(self) -> None:
        with pytest.raises(ValueError, match="active_column_count"):
            JointFM1DistributionResult(_student_t_adapter(), active_column_count=0)

    @pytest.mark.parametrize("n_samples", [None, 50])
    def test_mean_matches_direct_adapter_call(self, n_samples: int | None) -> None:
        adapter = _student_t_adapter()
        result = JointFM1DistributionResult(adapter, active_column_count=3)
        via_contract = result.mean(
            n_samples=n_samples, rng=torch.Generator().manual_seed(11)
        )
        direct = adapter.mean(
            n_samples=DEFAULT_MEAN_SAMPLES if n_samples is None else n_samples,
            rng=torch.Generator().manual_seed(11),
        )
        assert via_contract.shape == (1, 2, 3)
        assert torch.equal(via_contract, direct)

    def test_samples_match_direct_adapter_call(self) -> None:
        adapter = _student_t_adapter()
        result = JointFM1DistributionResult(adapter, active_column_count=3)
        via_contract = result.samples(
            n_samples=50,
            rng=torch.Generator().manual_seed(7),
            min_value=-10.0,
            max_value=10.0,
        )
        direct = adapter.sample(
            n_samples=50,
            rng=torch.Generator().manual_seed(7),
            min_value=-10.0,
            max_value=10.0,
        )
        assert via_contract.shape == (1, 50, 2, 3)
        assert torch.equal(via_contract, direct)

    def test_quantiles_match_direct_adapter_call(self) -> None:
        adapter = _student_t_adapter()
        result = JointFM1DistributionResult(adapter, active_column_count=3)
        q = [0.1, 0.5, 0.9]
        via_contract = result.quantiles(
            q, n_samples=200, rng=torch.Generator().manual_seed(3)
        )
        direct = adapter.quantiles(
            q=q, n_samples=200, rng=torch.Generator().manual_seed(3)
        )
        assert via_contract.shape == (3, 1, 2, 3)
        assert torch.equal(via_contract, direct)

    def test_log_prob_matches_direct_adapter_call(self) -> None:
        adapter = _student_t_adapter()
        result = JointFM1DistributionResult(adapter, active_column_count=3)
        values = torch.zeros(1, 2, 3)
        via_contract = result.log_prob(values)
        direct = adapter.log_prob(values)
        assert via_contract.shape == (1, 2)
        assert torch.equal(via_contract, direct)


def _correlated_adapter() -> PredictorAdapter:
    """Build a two-column Student-t mixture whose components disagree.

    Every column correlates with the other, so conditioning on one moves the
    other and a mistake shows up as a wrong number rather than as a shape that
    happens to match.
    """
    generator = torch.Generator().manual_seed(11)
    chol = torch.tril(torch.randn(1, 2, 3, 2, 2, generator=generator) * 0.4)
    return create_student_t_adapter(
        {
            "logits": torch.randn(1, 2, 3, generator=generator),
            "means": torch.randn(1, 2, 3, 2, generator=generator),
            "chol": chol + torch.eye(2),
            "df": torch.full((1, 2, 3), _STUDENT_T_DF),
        }
    )


class TestJointFM1DistributionResultConditioning:
    def test_condition_answers_the_closed_form_over_the_request_columns(
        self,
    ) -> None:
        adapter = _correlated_adapter()
        result = JointFM1DistributionResult(adapter, active_column_count=2)
        request = ConditionRequest(
            horizon=1, equality_columns=(0,), equality_values=(0.4,)
        )

        read_out = result.condition(
            request,
            n_samples=64,
            rng=torch.Generator().manual_seed(3),
            min_value=torch.tensor([float("-inf"), -0.5]),
        )

        assert read_out.read_out_columns == (1,)
        oracle = condition_at_position(adapter, request, n_samples=64, column_count=2)
        mean = read_out.mean()
        assert mean.shape == (1, 1, 2)
        torch.testing.assert_close(mean, oracle.mean())
        samples = read_out.samples()
        assert samples.shape == (1, 64, 1, 2)
        assert bool((samples[..., 1] >= -0.5).all())
        values = torch.tensor([[[0.4, 0.1]]])
        scores = read_out.log_prob(values)
        assert scores.shape == (1, 1)
        torch.testing.assert_close(scores, oracle.log_prob(values))

    def test_condition_refuses_parameters_of_another_width(self) -> None:
        result = JointFM1DistributionResult(
            _correlated_adapter(), active_column_count=3
        )
        request = ConditionRequest(
            horizon=0, equality_columns=(0,), equality_values=(0.0,)
        )

        with pytest.raises(RuntimeError, match="cover 2 columns.*declares 3"):
            result.condition(request, n_samples=8)
