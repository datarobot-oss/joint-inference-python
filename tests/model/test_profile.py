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

"""Stage timing of served requests and the ``profile`` command that reports it."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from jointfm_inference import cli
from jointfm_inference.core import timing
from jointfm_inference.model import profile
from jointfm_inference.model.profile import (
    REPORTED_STAGES,
    ProfileCase,
    build_profile_payload,
    profile_service,
)
from jointfm_inference.model.service import JointFMHTTPService
from tests.support.models import TINY_CAPACITY_ENVELOPE, tiny_model, write_checkpoint
from tests.support.service import (
    SERVICE_MAX_SAMPLE_COUNT,
    clear_service_environment,
    mount_service,
)

_SAMPLE_COUNT = 4
# The tiny checkpoint's widest request and its trained history and horizon.
_MAX_SERIES = int(TINY_CAPACITY_ENVELOPE["max_series"])
_HISTORY_ROWS = int(TINY_CAPACITY_ENVELOPE["n_input"])
_HORIZON_COUNT = int(TINY_CAPACITY_ENVELOPE["n_output"])
_SAMPLER_STAGES = {
    timing.SAMPLE_CHOLESKY_STAGE,
    timing.SAMPLE_GAUSSIAN_STAGE,
    timing.SAMPLE_GAMMA_STAGE,
    timing.SAMPLE_ASSEMBLE_STAGE,
}


@pytest.fixture(autouse=True)
def _timing_off() -> Iterator[None]:
    """Leave every test with timing off and an empty registry."""
    yield
    timing.set_enabled(False)
    timing.clear_timings()


@pytest.fixture
def service() -> JointFMHTTPService:
    return mount_service(tiny_model())


def _payload(return_mode: str, *, sample_count: int | None) -> dict:
    return build_profile_payload(
        ProfileCase(return_mode, _MAX_SERIES, sample_count),
        history_rows=_HISTORY_ROWS,
        horizon_count=_HORIZON_COUNT,
        schema_version="v6",
        seed=3,
    )


def test_a_served_samples_request_times_every_reported_stage(
    service: JointFMHTTPService,
) -> None:
    timing.set_enabled(True)
    service.predict_response_bytes(_payload("samples", sample_count=_SAMPLE_COUNT))

    assert set(timing.stage_seconds()) == {name for name, _ in REPORTED_STAGES}


def test_a_mean_request_draws_nothing_so_no_sampler_stage_runs(
    service: JointFMHTTPService,
) -> None:
    timing.set_enabled(True)
    service.predict_response_bytes(_payload("mean", sample_count=None))

    recorded = set(timing.stage_seconds())
    assert recorded == {name for name, _ in REPORTED_STAGES} - _SAMPLER_STAGES


def test_a_served_request_records_nothing_while_timing_is_off(
    service: JointFMHTTPService,
) -> None:
    service.predict_response_bytes(_payload("samples", sample_count=_SAMPLE_COUNT))
    assert timing.stage_seconds() == {}


def test_profiling_covers_the_grid_and_restores_the_timing_state(
    service: JointFMHTTPService,
) -> None:
    profiles = profile_service(
        service,
        column_counts=(1, _MAX_SERIES),
        sample_counts=(2, _SAMPLE_COUNT),
        return_modes=("samples", "quantiles"),
        history_rows=None,
        horizon_count=None,
        repeats=2,
        warmup=1,
        seed=0,
    )

    assert [
        (p.case.return_mode, p.case.column_count, p.case.sample_count) for p in profiles
    ] == [
        ("samples", 1, 2),
        ("samples", 1, 4),
        ("samples", 4, 2),
        ("samples", 4, 4),
        ("quantiles", 1, None),
        ("quantiles", 4, None),
    ]
    assert not timing.is_enabled()
    assert timing.stage_seconds() == {}
    for case_profile in profiles:
        assert len(case_profile.wall_seconds) == 2
        assert len(case_profile.stage_seconds[timing.REQUEST_STAGE]) == 2
        # The request's parts run one after another inside it.
        assert case_profile.median_unattributed_seconds() >= 0.0
    samples_4 = profiles[3]
    assert samples_4.response_bytes > profiles[1].response_bytes


def test_the_report_has_one_row_per_case_and_a_column_per_stage_that_ran(
    service: JointFMHTTPService,
) -> None:
    profiles = profile_service(
        service,
        column_counts=(2,),
        sample_counts=(_SAMPLE_COUNT,),
        return_modes=("mean", "log_prob"),
        history_rows=None,
        horizon_count=None,
        repeats=1,
        warmup=0,
        seed=0,
    )
    report = profile.format_report(service.health_payload(), profiles, repeats=1)
    table = [line for line in report.splitlines() if line.startswith("|")]

    assert len(table) == 2 + len(profiles)
    assert "cholesky (ms)" not in table[0]
    assert "encoder (ms)" in table[0]
    assert table[2].startswith("| mean | 2 | – |")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"column_counts": (5,)}, "max_series"),
        ({"sample_counts": (SERVICE_MAX_SAMPLE_COUNT + 1,)}, "max_sample_count"),
        ({"return_modes": ("median",)}, "not served"),
    ],
)
def test_profiling_refuses_a_grid_the_service_cannot_answer_before_any_request(
    service: JointFMHTTPService,
    monkeypatch: pytest.MonkeyPatch,
    options: dict,
    message: str,
) -> None:
    def _no_request(payload: object) -> bytes:
        raise AssertionError("no request may run")

    monkeypatch.setattr(service, "predict_response_bytes", _no_request)
    grid = {
        "column_counts": (1,),
        "sample_counts": (1,),
        "return_modes": ("samples",),
    }
    grid.update(options)
    with pytest.raises(SystemExit, match=message):
        profile_service(
            service,
            **grid,
            history_rows=None,
            horizon_count=None,
            repeats=1,
            warmup=0,
            seed=0,
        )


def test_the_profile_command_reports_the_configured_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "jointfm.pt"
    write_checkpoint(checkpoint_path)
    record_path = tmp_path / "profile.json"
    clear_service_environment(monkeypatch)
    monkeypatch.setenv("JOINTFM_INFERENCE_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("JOINTFM_INFERENCE_PRECISION", "fp32")

    cli.main(
        [
            "profile",
            "--columns",
            "2",
            "--samples",
            "3",
            "--return-modes",
            "samples,mean",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--output",
            str(record_path),
        ]
    )

    report = capsys.readouterr().out
    assert "generation jointfm-1" in report
    assert "| samples | 2 | 3 |" in report
    record = json.loads(record_path.read_text())
    assert "checkpoint_path" not in record["health"]
    assert [case["case"]["return_mode"] for case in record["cases"]] == [
        "samples",
        "mean",
    ]
    assert record["cases"][0]["stage_seconds"][timing.SAMPLE_GAMMA_STAGE]


def test_profile_options_parse_comma_lists() -> None:
    parsed = cli.build_parser().parse_args(
        ["profile", "--columns", "1,10", "--return-modes", "samples,mean"]
    )
    assert parsed.columns == (1, 10)
    assert parsed.samples == profile.DEFAULT_SAMPLE_COUNTS
    assert parsed.return_modes == ("samples", "mean")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["profile", "--columns", "0,10"])
