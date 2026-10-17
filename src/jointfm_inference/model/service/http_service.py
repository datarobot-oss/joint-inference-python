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

"""JSON transport adapter wrapping :class:`JointFMInferenceSession`.

The class validates an external inference request, builds canonical samples,
executes them on the session's serving backend, and serializes responses for
the packaged HTTP API. Every capability it advertises or gates on is read from
that backend, so the transport names no checkpoint generation. Execution
returns the backend-neutral distribution-result operations consumed by the
return-mode branches.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from http import HTTPStatus
from typing import Any, cast

import orjson
import torch

from jointfm_inference.data.adapters import DataFrameSchema, dataframe_to_canonical
from jointfm_inference.data.time_index import build_canonical_query_sample
from jointfm_inference.data.types import TIME_INDEX_MODES, TimeIndexModeLiteral
from jointfm_inference.model.inference import (
    CanonicalInferenceRequest,
    InferenceDistributionResult,
    InferenceRequestError,
    JointFMInferenceSession,
)
from jointfm_inference.model.inference.conditional_query import (
    ConditionRequest,
    ConditionalReadOut,
)
from jointfm_inference.model.inference.contract import ConditionalDistributionResult

from .column_specs import (
    DEFAULT_CALENDAR_ID,
    ColumnSpec,
    _parse_column_specs,
    _resolve_requested_columns,
)
from .conditions import parse_conditions, supported_condition_kinds
from .config import InferenceServiceConfig
from .constants import (
    HEALTH_REQUEST_TYPE,
    PREDICT_REQUEST_TYPE,
    QUERY_MODE_CONDITION,
    QUERY_MODE_FORECAST,
    REQUEST_FIELDS,
    SUPPORTED_REQUEST_TYPES,
)
from .data_frames import _history_frame, _log_prob_values_tensor, _query_rows_frame
from .errors import ServiceError, ServiceErrorCode
from .scoring import (
    _resolve_sample_count,
    _sampling_bound_tensors,
    _torch_generator,
)
from .serialization import (
    _serialize_log_prob_tensor,
    encode_response,
    _slice_forecast_tensor,
    _slice_quantile_tensor,
    _slice_sample_tensor,
)
from .time_conversion import _serialize_query_times
from .validation import (
    _optional_bool,
    _optional_float,
    _optional_int,
    _optional_string,
    _require_quantiles,
    _require_sequence,
    _require_string,
)


class JointFMHTTPService:
    """Thin JSON transport adapter over the packaged inference session."""

    def __init__(
        self,
        config: InferenceServiceConfig,
        session: JointFMInferenceSession,
    ) -> None:
        """Bind validated runtime config and model session to the HTTP transport.

        The served capabilities are read once here from the session's
        serving backend, which is fixed for the session's lifetime.

        Raises:
            ValueError: If ``config.precision`` disagrees with the precision
                the session runs at, so ``/healthz`` cannot misreport it.
        """
        if config.precision != session.precision:
            raise ValueError(
                f"service config precision {config.precision!r} does not match "
                f"the session precision {session.precision!r}"
            )
        self._config = config
        self._session = session
        self._device = torch.device(config.device)
        self._head_name = session.backend.head_name
        self._supported_return_modes = session.backend.supported_return_modes
        # The advertisement and the pre-execution gate read one derived answer
        # rather than a hardcoded list.
        self._supported_condition_kinds = supported_condition_kinds(
            supports_conditioning=session.backend.supports_conditioning
        )
        self._supported_query_modes = (QUERY_MODE_FORECAST,) + (
            (QUERY_MODE_CONDITION,) if self._supported_condition_kinds else ()
        )
        self._serviceable_time_index_modes = (
            frozenset(session.time_index_support.supported_request_time_index_modes)
            & TIME_INDEX_MODES
        )
        # Cap concurrent prediction requests so per-request GPU transients
        # (notably the per-draw Cholesky expansion of the Student-t sampler) cannot
        # stack across threads and OOM the device. ``ThreadingHTTPServer``
        # spawns one thread per request, and PyTorch CUDA ops release the GIL,
        # so without this gate the caching allocator would see N concurrent
        # peaks summed on one device. The bounded semaphore is acquired in
        # ``predict`` and excludes the cheap health short-circuit.
        self._concurrency_semaphore = threading.BoundedSemaphore(
            value=config.max_concurrent_requests,
        )

    def health_payload(self) -> dict[str, Any]:
        """Return readiness metadata for the mounted-checkpoint runtime.

        Includes identity fields, the model ``precision`` every forward runs
        at, the advertised query, return, and time-index modes, and the
        mounted model's ``decoding_strategy``. ``supported_metadata_traits``
        lists the expert-hint traits a column descriptor may carry.
        """
        time_index_support = self._session.time_index_support

        return {
            "status": "ok",
            "schema_version": self._config.schema_version,
            "image_version": self._config.image_version,
            "model_version": self._config.model_version,
            "checkpoint_version": self._config.checkpoint_version,
            "checkpoint_path": self._config.checkpoint_path,
            "device": self._config.device,
            "precision": self._session.precision,
            "generation": self._session.generation,
            "head": self._head_name,
            "decoding_strategy": self._session.decoding_strategy,
            "supported_query_modes": list(self._supported_query_modes),
            "supported_condition_kinds": list(self._supported_condition_kinds),
            "supported_metadata_traits": list(self._session.metadata_traits),
            "supported_return_modes": sorted(self._supported_return_modes),
            "supported_time_index_modes": sorted(self._serviceable_time_index_modes),
            "time_index_encoding": time_index_support.encoding,
            "max_sample_count": self._config.max_sample_count,
            "max_concurrent_requests": self._config.max_concurrent_requests,
            "data_generation": self._session.capacity_envelope.to_payload(),
        }

    def predict(self, request_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return the decoded JSON document :meth:`predict_response_bytes` sends.

        In-process callers therefore see exactly what an HTTP client decodes:
        plain lists of floats, each the shortest decimal of its float32 value.
        """
        return cast(
            dict[str, Any], orjson.loads(self.predict_response_bytes(request_payload))
        )

    def predict_response_bytes(self, request_payload: Mapping[str, Any]) -> bytes:
        """Validate one JSON request and return the encoded forecast response.

        The service accepts the external inference payload, checks schema and
        model versions, validates forecast-only query semantics, resolves the
        requested return mode, converts history rows to the canonical sample
        schema, and delegates prediction to the loaded adapter. Absolute
        datetime requests must name a time column, and requested query rows must
        be future rows appended after the observed history.

        Responses include model/checkpoint identity, head name, diagnostics, and
        the requested forecast payload: means, samples, quantiles, or
        log-probability scores. Sample count and quantile requests are bounded
        by service constants, optional seeds make stochastic sampling
        reproducible, and generated values are clipped to column bounds when
        the canonical metadata declares them.
        Validation failures raise ``ServiceError`` with an HTTP status and field
        context; unexpected adapter failures propagate as internal service
        errors at the API boundary.

        A ``request_type`` of ``"health"`` short-circuits the request: the
        service returns :meth:`health_payload` without inspecting any other
        field. Clients behind a gateway that forwards only the prediction
        route fetch service metadata this way. The branch must run
        before any schema or model version check so callers that do not yet
        know which versions the container pins can still discover them.

        The response is encoded inside the concurrency slot, so the host
        memory of an encoded response is bounded by ``max_concurrent_requests``
        exactly like the device memory of the forecast that produced it.
        """
        request_type = request_payload.get("request_type", PREDICT_REQUEST_TYPE)
        if request_type == HEALTH_REQUEST_TYPE:
            return encode_response(self.health_payload())
        if request_type not in SUPPORTED_REQUEST_TYPES:
            raise ServiceError(
                "VALIDATION_ERROR",
                (
                    "Unsupported request_type: expected one of "
                    f"{sorted(SUPPORTED_REQUEST_TYPES)}, got {request_type!r}"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field="request_type",
            )

        # Acquire the per-container concurrency slot before any version check
        # or forecast work so the cap also bounds the cheap validation path.
        # Acquisition blocks; the upstream gateway timeout decides how long a
        # queued request waits before giving up.
        with self._concurrency_semaphore:
            return encode_response(self._run_forecast_request(request_payload))

    def _run_forecast_request(
        self, request_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run one validated forecast request under the concurrency cap."""
        unknown_fields = sorted(set(map(str, request_payload)) - REQUEST_FIELDS)
        if unknown_fields:
            raise ServiceError(
                "VALIDATION_ERROR",
                (
                    f"unknown request fields {unknown_fields}; "
                    f"expected a subset of {sorted(REQUEST_FIELDS)}"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field=unknown_fields[0],
            )
        schema_version = _require_string(
            request_payload.get("schema_version"),
            field="schema_version",
        )
        if schema_version != self._config.schema_version:
            raise ServiceError(
                "SCHEMA_VERSION_MISMATCH",
                (
                    "Unsupported schema_version: "
                    f"expected {self._config.schema_version!r}, got {schema_version!r}"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field="schema_version",
            )

        # Optional drift guard: callers may omit model_version, in which case
        # they accept whatever this container serves; when supplied (typically
        # echoed back from /healthz or pinned in client config) it must match,
        # so a stale client pinned to a redeployed version still fails fast.
        model_version = _optional_string(
            request_payload.get("model_version"),
            field="model_version",
        )
        if model_version is not None and model_version != self._config.model_version:
            raise ServiceError(
                "MODEL_VERSION_MISMATCH",
                (
                    "Unsupported model_version: "
                    f"expected {self._config.model_version!r}, got {model_version!r}"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field="model_version",
            )

        query_mode = request_payload.get("query_mode", QUERY_MODE_FORECAST)
        if query_mode == QUERY_MODE_CONDITION and not self._supported_condition_kinds:
            raise ServiceError(
                ServiceErrorCode.UNSUPPORTED_HEAD_QUERY_COMBINATION,
                (
                    f"The mounted {self._head_name!r} head on the "
                    f"{self._session.generation!r} generation cannot answer condition "
                    "queries; /healthz advertises the modes it serves"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field="query_mode",
            )
        if query_mode not in self._supported_query_modes:
            raise ServiceError(
                ServiceErrorCode.VALIDATION_ERROR,
                (
                    "Unsupported query_mode: expected one of "
                    f"{list(self._supported_query_modes)}, got {query_mode!r}"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field="query_mode",
            )
        if query_mode != QUERY_MODE_CONDITION and "conditions" in request_payload:
            raise ServiceError(
                ServiceErrorCode.VALIDATION_ERROR,
                "conditions need query_mode='condition'",
                status=HTTPStatus.BAD_REQUEST,
                field="conditions",
            )

        return_mode = _require_string(
            request_payload.get("return_mode", "mean"),
            field="return_mode",
        )
        if return_mode not in self._supported_return_modes:
            raise ServiceError(
                "VALIDATION_ERROR",
                (
                    "Unsupported return_mode: "
                    f"expected one of {sorted(self._supported_return_modes)}, "
                    f"got {return_mode!r}"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field="return_mode",
            )

        time_index_mode = _require_string(
            request_payload.get("time_index_mode"),
            field="time_index_mode",
        )
        if time_index_mode not in self._serviceable_time_index_modes:
            raise ServiceError(
                "VALIDATION_ERROR",
                (
                    "Unsupported time_index_mode: expected one of "
                    f"{sorted(self._serviceable_time_index_modes)}, "
                    f"got {time_index_mode!r}"
                ),
                status=HTTPStatus.BAD_REQUEST,
                field="time_index_mode",
            )

        time_column = _optional_string(
            request_payload.get("time_column"), "time_column"
        )
        if time_index_mode == "absolute_datetime" and time_column is None:
            raise ServiceError(
                "VALIDATION_ERROR",
                "absolute_datetime requests must provide time_column",
                status=HTTPStatus.BAD_REQUEST,
                field="time_column",
            )

        raw_columns = _require_sequence(
            request_payload.get("columns"),
            field="columns",
        )
        column_specs = _parse_column_specs(raw_columns)
        requested_column_names = _resolve_requested_columns(
            column_specs,
            request_payload.get("requested_columns"),
        )
        query_times = _require_sequence(
            request_payload.get("query_times"),
            field="query_times",
        )
        condition_requests: tuple[ConditionRequest, ...] = ()
        if query_mode == QUERY_MODE_CONDITION:
            condition_requests = parse_conditions(
                request_payload,
                column_specs=column_specs,
                query_time_count=len(query_times),
            )
        history_rows = _require_sequence(
            request_payload.get("history_rows"),
            field="history_rows",
        )
        history_frame = _history_frame(
            history_rows,
            column_specs=column_specs,
            time_index_mode=time_index_mode,
            time_column=time_column,
        )

        if _optional_bool(
            request_payload.get("use_local_normalized_time"),
            field="use_local_normalized_time",
            default=False,
        ):
            raise ServiceError(
                "VALIDATION_ERROR",
                "use_local_normalized_time is not served; omit it or pass false",
                status=HTTPStatus.BAD_REQUEST,
                field="use_local_normalized_time",
            )
        calendar_id = _optional_string(
            request_payload.get("calendar_id"), field="calendar_id"
        )
        if calendar_id is not None and calendar_id != DEFAULT_CALENDAR_ID:
            raise ServiceError(
                "VALIDATION_ERROR",
                f"calendar_id is not served; omit it or pass {DEFAULT_CALENDAR_ID!r}",
                status=HTTPStatus.BAD_REQUEST,
                field="calendar_id",
            )

        try:
            schema = DataFrameSchema(
                columns=column_specs,
                time_index_mode=cast(TimeIndexModeLiteral, time_index_mode),
                time_column=time_column,
                time_scale_seconds=_optional_float(
                    request_payload.get("time_scale_seconds"),
                    field="time_scale_seconds",
                ),
                timezone=_optional_string(request_payload.get("timezone"), "timezone"),
            )
            observed_sample = dataframe_to_canonical(history_frame, schema)
        except (TypeError, ValueError) as error:
            raise ServiceError(
                "VALIDATION_ERROR",
                str(error),
                status=HTTPStatus.BAD_REQUEST,
                field="history_rows",
            ) from error

        try:
            query_sample = build_canonical_query_sample(
                observed_sample,
                query_times=query_times,
            )
        except (TypeError, ValueError) as error:
            raise ServiceError(
                "VALIDATION_ERROR",
                str(error),
                status=HTTPStatus.BAD_REQUEST,
                field="query_times",
            ) from error

        # The query builder appends forecast rows strictly after the history.
        horizon_count = len(query_times)
        canonical_request = CanonicalInferenceRequest(sample=query_sample)

        log_prob_values = None
        if return_mode == "log_prob":
            query_rows_frame = _query_rows_frame(
                request_payload.get("query_rows"),
                column_specs=column_specs,
                time_index_mode=time_index_mode,
                time_column=time_column,
                query_times=query_times,
            )
            log_prob_values = _log_prob_values_tensor(
                query_frame=query_rows_frame,
                schema=schema,
                requested_column_names=requested_column_names,
                device=self._device,
            )

        seed = _optional_int(request_payload.get("seed"), field="seed")
        sample_count = _resolve_sample_count(
            request_payload=request_payload,
            config=self._config,
        )
        quantiles: list[float] = (
            _require_quantiles(request_payload.get("quantiles"))
            if return_mode == "quantiles"
            else []
        )
        rng = _torch_generator(seed=seed, device=self._device)

        column_name_to_index = {
            column.name: idx for idx, column in enumerate(column_specs)
        }
        requested_indices = [
            column_name_to_index[name] for name in requested_column_names
        ]

        # All request validation is done; the single stateless full-request
        # model forward runs last, on the backend fixed at construction.
        try:
            result = self._session.execute(canonical_request)
        except InferenceRequestError as error:
            raise ServiceError(
                "VALIDATION_ERROR",
                str(error),
                status=HTTPStatus.BAD_REQUEST,
                field=error.field,
            ) from error

        if condition_requests:
            return self._condition_response(
                result,
                condition_requests,
                return_mode=return_mode,
                quantiles=quantiles,
                query_times=query_times,
                time_index_mode=time_index_mode,
                column_specs=column_specs,
                requested_column_names=requested_column_names,
                requested_indices=requested_indices,
                log_prob_values=log_prob_values,
                sample_count=sample_count,
                rng=rng,
                seed=seed,
                history_row_count=len(history_rows),
                horizon_count=horizon_count,
            )

        tensor = self._return_mode_tensor(
            result,
            return_mode=return_mode,
            quantiles=quantiles,
            column_specs=column_specs,
            log_prob_values=log_prob_values,
            sample_count=sample_count,
            rng=rng,
        )
        return {
            "schema_version": self._config.schema_version,
            "image_version": self._config.image_version,
            "model_version": self._config.model_version,
            "checkpoint_version": self._config.checkpoint_version,
            "head": self._head_name,
            "query_mode": query_mode,
            "return_mode": return_mode,
            "outputs": _outputs_payload(
                tensor,
                return_mode=return_mode,
                quantiles=quantiles,
                query_times=query_times,
                time_index_mode=time_index_mode,
                requested_column_names=requested_column_names,
                requested_indices=requested_indices,
            ),
            "plausibility": None,
            "diagnostics": {
                "history_rows": len(history_rows),
                "horizon_count": horizon_count,
                "seed": seed,
            },
            "errors": [],
        }

    def _return_mode_tensor(
        self,
        result: InferenceDistributionResult,
        *,
        return_mode: str,
        quantiles: Sequence[float],
        column_specs: Sequence[ColumnSpec],
        log_prob_values: torch.Tensor | None,
        sample_count: int,
        rng: torch.Generator | None,
    ) -> torch.Tensor:
        """Answer one return mode over every future position, unconditionally."""
        if return_mode == "mean":
            return result.mean(n_samples=sample_count, rng=rng)
        if return_mode == "samples":
            sample_min_value, sample_max_value = _sampling_bound_tensors(
                column_specs,
                device=self._device,
            )
            return result.samples(
                n_samples=sample_count,
                rng=rng,
                min_value=sample_min_value,
                max_value=sample_max_value,
            )
        if return_mode == "quantiles":
            return result.quantiles(q=quantiles, n_samples=sample_count, rng=rng)
        if return_mode == "log_prob":
            if log_prob_values is None:
                raise RuntimeError("log_prob return mode did not prepare query values")
            return result.log_prob(log_prob_values)
        raise RuntimeError(f"Unhandled return_mode {return_mode!r}")

    def _condition_response(
        self,
        result: InferenceDistributionResult,
        condition_requests: Sequence[ConditionRequest],
        *,
        return_mode: str,
        quantiles: Sequence[float],
        query_times: Sequence[Any],
        time_index_mode: str,
        column_specs: Sequence[ColumnSpec],
        requested_column_names: Sequence[str],
        requested_indices: Sequence[int],
        log_prob_values: torch.Tensor | None,
        sample_count: int,
        rng: torch.Generator | None,
        seed: int | None,
        history_row_count: int,
        horizon_count: int,
    ) -> dict[str, Any]:
        """Answer every future position under the conditions that cover it.

        A covered position carries its conditional read-out; a position no
        condition covers carries the unconditioned answer. That is exact rather
        than a placeholder: the served result emits one joint per position and
        draws positions independently, so a condition at one position leaves
        every other position's distribution unchanged. The response therefore
        has the forecast mode's layout, position for position.

        The two ``plausibility`` numbers sum their per-position values over
        the covered positions, which is again exact because positions are
        independent. Within a position they chain as
        :class:`ConditionalReadOut` describes, so their sum over the whole
        response is the plausibility of the whole condition set.

        Raises:
            ServiceError: with ``VALIDATION_ERROR`` on field ``conditions``
                when the library rejects a resolved request, and on field
                ``query_rows`` when a scored row contradicts the condition
                covering its position.
        """
        covered = {request.horizon for request in condition_requests}
        # The unconditioned answer draws before any conditional does, so an
        # uncovered position carries exactly what the forecast mode answers
        # under the same seed.
        unconditioned = (
            None
            if len(covered) == horizon_count
            else self._return_mode_tensor(
                result,
                return_mode=return_mode,
                quantiles=quantiles,
                column_specs=column_specs,
                log_prob_values=log_prob_values,
                sample_count=sample_count,
                rng=rng,
            )
        )
        minimum, maximum = _sampling_bound_tensors(column_specs, device=self._device)
        # The capability gate at the top of the request path already refused
        # any backend whose result cannot condition, so the narrowing is the
        # gate's conclusion rather than a second guess at it.
        conditional = cast(ConditionalDistributionResult, result)
        read_outs: dict[int, ConditionalReadOut] = {}
        for request in condition_requests:
            try:
                read_outs[request.horizon] = conditional.condition(
                    request,
                    n_samples=sample_count,
                    rng=rng,
                    min_value=minimum,
                    max_value=maximum,
                )
            except ValueError as error:
                raise ServiceError(
                    ServiceErrorCode.VALIDATION_ERROR,
                    str(error),
                    status=HTTPStatus.BAD_REQUEST,
                    field="conditions",
                ) from error

        horizon_axis = _HORIZON_AXIS[return_mode]
        pieces: list[torch.Tensor] = []
        for horizon in range(horizon_count):
            if horizon in read_outs:
                piece = _read_out_tensor(
                    read_outs[horizon],
                    return_mode=return_mode,
                    quantiles=quantiles,
                    log_prob_values=log_prob_values,
                    horizon=horizon,
                )
            else:
                if unconditioned is None:
                    raise RuntimeError(
                        f"position {horizon} is uncovered but no unconditioned "
                        "answer was computed"
                    )
                piece = unconditioned.narrow(horizon_axis, horizon, 1)
            # Both answers cover every request column; keep the projection.
            if return_mode != "log_prob":
                piece = piece[..., list(requested_indices)]
            pieces.append(piece)
        tensor = torch.cat(pieces, dim=horizon_axis)

        return {
            "schema_version": self._config.schema_version,
            "image_version": self._config.image_version,
            "model_version": self._config.model_version,
            "checkpoint_version": self._config.checkpoint_version,
            "head": self._head_name,
            "query_mode": QUERY_MODE_CONDITION,
            "return_mode": return_mode,
            "outputs": _outputs_payload(
                tensor,
                return_mode=return_mode,
                quantiles=quantiles,
                query_times=query_times,
                time_index_mode=time_index_mode,
                requested_column_names=requested_column_names,
                requested_indices=list(range(len(requested_indices))),
            ),
            "plausibility": {
                "equality_log_density": _summed_over_positions(
                    [read_out.equality_log_density for read_out in read_outs.values()]
                ),
                "region_log_probability": _summed_over_positions(
                    [read_out.region_log_probability for read_out in read_outs.values()]
                ),
            },
            "diagnostics": {
                "history_rows": history_row_count,
                "horizon_count": horizon_count,
                "seed": seed,
                "condition_draws": sample_count,
                "interval_estimator": _least_supported_estimator(read_outs.values()),
            },
            "errors": [],
        }


# Axis of the future positions in each return mode's tensor: ``(B, H, D)`` for
# a mean, ``(B, S, H, D)`` for draws, ``(Q, B, H, D)`` for quantiles, and
# ``(B, H)`` for scores.
_HORIZON_AXIS = {"mean": 1, "samples": 2, "quantiles": 2, "log_prob": 1}


def _read_out_tensor(
    read_out: ConditionalReadOut,
    *,
    return_mode: str,
    quantiles: Sequence[float],
    log_prob_values: torch.Tensor | None,
    horizon: int,
) -> torch.Tensor:
    """Answer one return mode at one covered position, one position wide."""
    if return_mode == "mean":
        return read_out.mean()
    if return_mode == "samples":
        return read_out.samples()
    if return_mode == "quantiles":
        return read_out.quantiles(quantiles)
    if return_mode == "log_prob":
        if log_prob_values is None:
            raise RuntimeError("log_prob return mode did not prepare query values")
        try:
            return read_out.log_prob(log_prob_values[:, horizon : horizon + 1])
        except ValueError as error:
            raise ServiceError(
                ServiceErrorCode.VALIDATION_ERROR,
                f"query_rows position {horizon}: {error}",
                status=HTTPStatus.BAD_REQUEST,
                field="query_rows",
            ) from error
    raise RuntimeError(f"Unhandled return_mode {return_mode!r}")


def _outputs_payload(
    tensor: torch.Tensor,
    *,
    return_mode: str,
    quantiles: Sequence[float],
    query_times: Sequence[Any],
    time_index_mode: str,
    requested_column_names: Sequence[str],
    requested_indices: Sequence[int],
) -> dict[str, Any]:
    """Serialize one return mode's tensor into the response ``outputs`` block."""
    outputs: dict[str, Any] = {
        "query_times": _serialize_query_times(
            query_times,
            time_index_mode=time_index_mode,
        ),
        "requested_columns": list(requested_column_names),
        "mean": None,
        "samples": None,
        "quantiles": None,
        "log_prob": None,
    }
    if return_mode == "mean":
        outputs["mean"] = _slice_forecast_tensor(tensor, requested_indices)
    elif return_mode == "samples":
        outputs["samples"] = _slice_sample_tensor(tensor, requested_indices)
    elif return_mode == "quantiles":
        outputs["quantiles"] = _slice_quantile_tensor(
            tensor,
            quantiles=quantiles,
            requested_indices=requested_indices,
        )
    elif return_mode == "log_prob":
        outputs["log_prob"] = _serialize_log_prob_tensor(tensor)
    else:
        raise RuntimeError(f"Unhandled return_mode {return_mode!r}")
    return outputs


def _summed_over_positions(values: Sequence[torch.Tensor | None]) -> float | None:
    """Sum one plausibility number over the positions that report it."""
    reported = [float(value) for value in values if value is not None]
    if not reported:
        return None
    return float(sum(reported))


def _least_supported_estimator(
    read_outs: Iterable[ConditionalReadOut],
) -> dict[str, Any] | None:
    """Describe the estimated position whose weights support its region least.

    Only a multi-column box is estimated; a position with one interval column
    or none reports nothing. Across several estimated positions the smallest
    effective sample size is the one that bounds how far the whole response
    can be trusted, so it is the one reported.
    """
    estimates = [
        (float(read_out.interval_effective_sample_size), read_out.interval_points)
        for read_out in read_outs
        if read_out.interval_points is not None
        and read_out.interval_effective_sample_size is not None
    ]
    if not estimates:
        return None
    effective_sample_size, points = min(estimates)
    return {"points": points, "effective_sample_size": effective_sample_size}
