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
