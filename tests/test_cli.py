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

"""The ``jointfm-inference`` command."""

from __future__ import annotations

import pytest

from jointfm_inference import cli
from jointfm_inference.model import probe
from jointfm_inference.model.service import server


def test_serve_runs_the_service(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def _main() -> None:
        events.append("served")

    monkeypatch.setattr(server, "main", _main)
    cli.main(["serve"])
    assert events == ["served"]


def test_serve_takes_no_options() -> None:
    """Every service setting is an environment variable, never a flag."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["serve", "--port", "9000"])
    assert exc_info.value.code == 2


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert exc_info.value.code == 2


def test_probe_passes_its_parsed_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _run(arguments) -> None:
        seen.append(arguments.service_url)

    monkeypatch.setattr(probe, "run", _run)
    # The handler is bound when the parser is built, so build it after patching.
    cli.main(["probe", "--service-url", "http://a:1", "--service-url", "http://b:2"])
    assert seen == [["http://a:1", "http://b:2"]]
