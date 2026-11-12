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

"""``jointfm-inference profile``: per-stage latency of served forecast requests.

The command loads the checkpoint that the ``JOINTFM_INFERENCE_*`` environment
configures, exactly as ``serve`` does, and answers synthetic forecast requests
in this process through the request path a ``/predict`` call takes, without
the HTTP transport, so network transfer is not part of any figure. Requests
run one at a time, which the stage timers of
:mod:`jointfm_inference.core.timing` need for a per-request split.

Each case is one return mode at one column count, and for ``samples`` one
sample count; the other return modes leave ``n_samples`` to the service
default. A case sends one request per warm-up round, then ``repeats`` requests
with timing off for the end-to-end wall time, then ``repeats`` with timing on
for the stage split. The report gives medians over the repeats, so the
difference between the two wall times is what the timers' device
synchronizations cost.

The synthetic history is a seeded Gaussian random walk per column, as long as
the checkpoint's longest trained history, with as many horizons as its
longest trained horizon unless the options say otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jointfm_inference.core.timing import (
    ENCODE_STAGE,
    ENCODER_STAGE,
    EXECUTE_STAGE,
    FORWARD_STAGE,
    POOL_HEAD_STAGE,
    PREPARE_STAGE,
    READOUT_STAGE,
    REQUEST_STAGE,
    SAMPLE_ASSEMBLE_STAGE,
    SAMPLE_CHOLESKY_STAGE,
    SAMPLE_GAMMA_STAGE,
    SAMPLE_GAUSSIAN_STAGE,
    SERIALIZE_STAGE,
    clear_timings,
    is_enabled,
    set_enabled,
    stage_seconds,
)
from jointfm_inference.model.service.http_service import JointFMHTTPService
from jointfm_inference.model.service.server import create_service_from_env

PROFILE_DESCRIPTION = (
    "Time each stage of served forecast requests on the checkpoint that the "
    "JOINTFM_INFERENCE_* environment configures: the checkpoint is loaded in "
    "this process and answers synthetic requests one at a time, without HTTP, "
    "across the given column counts, sample counts, and return modes. Prints "
    "a Markdown table of median milliseconds per stage."
)

DEFAULT_COLUMN_COUNTS = (1, 10, 50, 100)
DEFAULT_SAMPLE_COUNTS = (100, 1000, 5000)
DEFAULT_RETURN_MODES = ("samples", "mean", "quantiles", "log_prob")
PROFILE_QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 1
DEFAULT_SEED = 0
# Far from zero, so no column's history standardizes against a vanishing level.
RANDOM_WALK_LEVEL = 100.0
TIME_COLUMN = "step"
MILLISECONDS_PER_SECOND = 1000.0
BYTES_PER_MEGABYTE = 1e6

# Table columns in report order: a stage precedes the stages that are its parts.
REPORTED_STAGES: tuple[tuple[str, str], ...] = (
    (REQUEST_STAGE, "request"),
    (EXECUTE_STAGE, "execute"),
    (PREPARE_STAGE, "prepare"),
    (FORWARD_STAGE, "forward"),
    (ENCODER_STAGE, "encoder"),
    (POOL_HEAD_STAGE, "pool+head"),
    (READOUT_STAGE, "read-out"),
    (SAMPLE_CHOLESKY_STAGE, "cholesky"),
    (SAMPLE_GAUSSIAN_STAGE, "gaussian"),
    (SAMPLE_GAMMA_STAGE, "gamma"),
    (SAMPLE_ASSEMBLE_STAGE, "assemble"),
    (SERIALIZE_STAGE, "serialize"),
    (ENCODE_STAGE, "encode"),
)
# The stages a request runs in sequence; the rest of its time is validation
# and the conversion to a canonical request.
REQUEST_PARTS = (EXECUTE_STAGE, READOUT_STAGE, SERIALIZE_STAGE, ENCODE_STAGE)
# Health fields the JSON record keeps; the checkpoint path stays out of it.
RECORDED_HEALTH_FIELDS = (
    "generation",
    "head",
    "checkpoint_version",
    "precision",
    "device",
    "image_version",
    "max_sample_count",
    "data_generation",
)


@dataclass(frozen=True)
class ProfileCase:
    """One timed request shape; ``sample_count`` ``None`` sends no ``n_samples``."""

    return_mode: str
    column_count: int
    sample_count: int | None


@dataclass(frozen=True)
class CaseProfile:
    """The timings of one case, one entry per timed repeat.

    ``stage_seconds`` holds, per stage, the seconds each timed request spent
    in it, summed over the stage's regions (a stage such as the forward may
    run more than once per request).
    """

    case: ProfileCase
    response_bytes: int
    wall_seconds: tuple[float, ...]
    timed_wall_seconds: tuple[float, ...]
    stage_seconds: dict[str, tuple[float, ...]]

    def median_stage_seconds(self, name: str) -> float | None:
        """Return the stage's median seconds, or ``None`` when it never ran."""
        if name not in self.stage_seconds:
            return None
        return statistics.median(self.stage_seconds[name])

    def median_unattributed_seconds(self) -> float:
        """Return the median request time outside :data:`REQUEST_PARTS`."""
        requests = self.stage_seconds[REQUEST_STAGE]
        parts = [self.stage_seconds[name] for name in REQUEST_PARTS]
        return statistics.median(
            [
                request - sum(part[index] for part in parts)
                for index, request in enumerate(requests)
            ]
        )


def _positive_int_list(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers."""
    values = tuple(int(item) for item in text.split(","))
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def _return_mode_list(text: str) -> tuple[str, ...]:
    """Parse a comma-separated list of return modes."""
    return tuple(item.strip() for item in text.split(","))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the profile command's options to ``parser``."""
    parser.add_argument(
        "--columns",
        type=_positive_int_list,
        default=DEFAULT_COLUMN_COUNTS,
        help="Comma-separated column counts (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int_list,
        default=DEFAULT_SAMPLE_COUNTS,
        help="Comma-separated sample counts of the samples mode (default: %(default)s)",
    )
    parser.add_argument(
        "--return-modes",
        type=_return_mode_list,
        default=DEFAULT_RETURN_MODES,
        help="Comma-separated return modes (default: %(default)s)",
    )
    parser.add_argument(
        "--history-rows",
        type=int,
        default=None,
        help="History length in steps (default: the checkpoint's n_input)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        default=None,
        help="Forecast horizons per request (default: the checkpoint's n_output)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Requests per case with timing off, and again with it on",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Untimed requests per case before the timed ones",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed of the synthetic history and of every request's draws",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write every timing, not just the medians, to this JSON file",
    )


def build_profile_payload(
    case: ProfileCase,
    *,
    history_rows: int,
    horizon_count: int,
    schema_version: str,
    seed: int,
) -> dict[str, Any]:
    """Return one ordinal-time forecast request answering ``case``.

    Every column is a numeric random walk and is requested; a ``log_prob``
    request scores the walk's own continuation.
    """
    generator = np.random.default_rng(seed)
    steps = history_rows + horizon_count
    walk = RANDOM_WALK_LEVEL + np.cumsum(
        generator.standard_normal((steps, case.column_count)), axis=0
    )
    names = [f"series_{index}" for index in range(case.column_count)]

    def row(step: int) -> dict[str, Any]:
        """Return the request row of one step of the walk."""
        return {TIME_COLUMN: step, **dict(zip(names, walk[step].tolist()))}

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "return_mode": case.return_mode,
        "time_index_mode": "ordinal",
        "time_column": TIME_COLUMN,
        "columns": [{"name": name, "modality": "numeric"} for name in names],
        "history_rows": [row(step) for step in range(history_rows)],
        "query_times": list(range(history_rows, steps)),
        "seed": seed,
    }
    if case.sample_count is not None:
        payload["n_samples"] = case.sample_count
    if case.return_mode == "quantiles":
        payload["quantiles"] = list(PROFILE_QUANTILES)
    if case.return_mode == "log_prob":
        payload["query_rows"] = [row(step) for step in range(history_rows, steps)]
    return payload


def profile_cases(
    *,
    column_counts: Sequence[int],
    sample_counts: Sequence[int],
    return_modes: Sequence[str],
) -> list[ProfileCase]:
    """Return the grid: every return mode at every column count, samples at every sample count."""
    cases: list[ProfileCase] = []
    for return_mode in return_modes:
        counts: Sequence[int | None] = (
            sample_counts if return_mode == "samples" else (None,)
        )
        for column_count in column_counts:
            cases.extend(
                ProfileCase(return_mode, column_count, sample_count)
                for sample_count in counts
            )
    return cases


def _timed_request(
    service: JointFMHTTPService, payload: Mapping[str, Any]
) -> tuple[float, bytes]:
    """Answer one request and return its wall seconds and response body."""
    start = time.perf_counter()
    body = service.predict_response_bytes(payload)
    return time.perf_counter() - start, body


def profile_case(
    service: JointFMHTTPService,
    case: ProfileCase,
    payload: Mapping[str, Any],
    *,
    repeats: int,
    warmup: int,
) -> CaseProfile:
    """Time ``payload`` with timing off, then with it on; restore the timing state.

    Raises:
        ValueError: when ``repeats`` is not positive or ``warmup`` is negative.
    """
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    was_enabled = is_enabled()
    set_enabled(False)
    try:
        for _ in range(warmup):
            service.predict_response_bytes(payload)
        walls = [_timed_request(service, payload)[0] for _ in range(repeats)]
        set_enabled(True)
        timed_walls: list[float] = []
        per_request: list[dict[str, float]] = []
        response_sizes: set[int] = set()
        for _ in range(repeats):
            clear_timings()
            seconds, body = _timed_request(service, payload)
            timed_walls.append(seconds)
            response_sizes.add(len(body))
            per_request.append(
                {name: sum(values) for name, values in stage_seconds().items()}
            )
    finally:
        clear_timings()
        set_enabled(was_enabled)
    names = sorted({name for request in per_request for name in request})
    return CaseProfile(
        case=case,
        response_bytes=max(response_sizes),
        wall_seconds=tuple(walls),
        timed_wall_seconds=tuple(timed_walls),
        stage_seconds={
            name: tuple(request.get(name, 0.0) for request in per_request)
            for name in names
        },
    )


def _milliseconds(seconds: float | None) -> str:
    """Format seconds as milliseconds, or a dash for a stage that never ran."""
    if seconds is None:
        return "–"
    return f"{seconds * MILLISECONDS_PER_SECOND:.1f}"


def format_report(
    health: Mapping[str, Any], profiles: Sequence[CaseProfile], *, repeats: int
) -> str:
    """Return the run's settings and one Markdown table row per case.

    Only stages that ran in some case get a column; ``other`` is the request
    time outside execution, read-out, serialization, and encoding.
    """
    device = torch.device(health["device"])
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    )
    lines = [
        f"generation {health['generation']}, head {health['head']}, "
        f"checkpoint {health['checkpoint_version']}, precision {health['precision']}",
        f"device {health['device']} ({device_name}), torch {torch.__version__}, "
        f"runner {health['image_version']}, medians of {repeats} requests",
        "",
    ]
    stages = [
        (name, heading)
        for name, heading in REPORTED_STAGES
        if any(name in profile.stage_seconds for profile in profiles)
    ]
    headings = [
        "mode",
        "columns",
        "samples",
        "response (MB)",
        "wall (ms)",
        "timed wall (ms)",
        "other (ms)",
        *(f"{heading} (ms)" for _, heading in stages),
    ]
    lines.append("| " + " | ".join(headings) + " |")
    lines.append("|" + "---|" * len(headings))
    for profile in profiles:
        case = profile.case
        cells = [
            case.return_mode,
            str(case.column_count),
            "–" if case.sample_count is None else str(case.sample_count),
            f"{profile.response_bytes / BYTES_PER_MEGABYTE:.2f}",
            _milliseconds(statistics.median(profile.wall_seconds)),
            _milliseconds(statistics.median(profile.timed_wall_seconds)),
            _milliseconds(profile.median_unattributed_seconds()),
            *(_milliseconds(profile.median_stage_seconds(name)) for name, _ in stages),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _require_within(
    values: Sequence[int], *, limit: int, option: str, limit_name: str
) -> None:
    """Refuse any value above ``limit`` before a single request runs."""
    above = [value for value in values if value > limit]
    if above:
        raise SystemExit(
            f"{option} {above} exceeds the service's {limit_name} of {limit}"
        )


def profile_service(
    service: JointFMHTTPService,
    *,
    column_counts: Sequence[int],
    sample_counts: Sequence[int],
    return_modes: Sequence[str],
    history_rows: int | None,
    horizon_count: int | None,
    repeats: int,
    warmup: int,
    seed: int,
) -> list[CaseProfile]:
    """Profile every case of the grid on ``service``, one request at a time.

    ``history_rows`` and ``horizon_count`` default to the checkpoint's
    ``n_input`` and ``n_output``.

    Raises:
        SystemExit: when a column count exceeds the checkpoint's
            ``max_series``, a sample count the service's
            ``max_sample_count``, or a return mode is not served.
    """
    health = service.health_payload()
    envelope = health["data_generation"]
    _require_within(
        column_counts,
        limit=envelope["max_series"],
        option="--columns",
        limit_name="max_series",
    )
    if "samples" in return_modes:
        _require_within(
            sample_counts,
            limit=health["max_sample_count"],
            option="--samples",
            limit_name="max_sample_count",
        )
    unserved = sorted(set(return_modes) - set(health["supported_return_modes"]))
    if unserved:
        raise SystemExit(f"--return-modes {unserved} are not served")
    history = envelope["n_input"] if history_rows is None else history_rows
    horizons = envelope["n_output"] if horizon_count is None else horizon_count
    return [
        profile_case(
            service,
            case,
            build_profile_payload(
                case,
                history_rows=history,
                horizon_count=horizons,
                schema_version=health["schema_version"],
                seed=seed,
            ),
            repeats=repeats,
            warmup=warmup,
        )
        for case in profile_cases(
            column_counts=column_counts,
            sample_counts=sample_counts,
            return_modes=return_modes,
        )
    ]


def run(arguments: argparse.Namespace) -> None:
    """Load the configured checkpoint, profile the grid, and print the report."""
    service = create_service_from_env()
    profiles = profile_service(
        service,
        column_counts=arguments.columns,
        sample_counts=arguments.samples,
        return_modes=arguments.return_modes,
        history_rows=arguments.history_rows,
        horizon_count=arguments.horizons,
        repeats=arguments.repeats,
        warmup=arguments.warmup,
        seed=arguments.seed,
    )
    health = service.health_payload()
    print(format_report(health, profiles, repeats=arguments.repeats))
    if arguments.output is not None:
        record = {
            "health": {key: health[key] for key in RECORDED_HEALTH_FIELDS},
            "torch": torch.__version__,
            "cases": [asdict(profile) for profile in profiles],
        }
        arguments.output.write_text(json.dumps(record, indent=2) + "\n")


__all__ = [
    "PROFILE_DESCRIPTION",
    "CaseProfile",
    "ProfileCase",
    "add_arguments",
    "build_profile_payload",
    "format_report",
    "profile_case",
    "profile_cases",
    "profile_service",
    "run",
]
