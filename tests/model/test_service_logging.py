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

"""The service process's log: where its records go and what a server failure leaves there."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

import jointfm_inference.model.service.server as service_server
from jointfm_inference.model.service.server import (
    LOG_FILE_COUNT,
    LOG_FILE_NAME,
    configure_logging,
)
from tests.support.http_handler import (
    HANDLER_LOGGER,
    INTERNAL_DETAIL,
    failing_handler,
    internal_error_reference,
)

# A file size a few records outgrow, so a test rolls the log over quickly.
SMALL_LOG_FILE_BYTES = 256
ROLLOVER_RECORDS = 64


# pytest adds and removes its own capture handlers around each test phase.
PYTEST_LOGGING_MODULE = "_pytest.logging"


@pytest.fixture
def restored_root_logger() -> Iterator[logging.Logger]:
    """Remove and close the handlers a test adds to the root logger, and restore its level."""
    root = logging.getLogger()
    saved_handlers, saved_level = set(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        if (
            handler not in saved_handlers
            and type(handler).__module__ != PYTEST_LOGGING_MODULE
        ):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def test_a_server_failure_leaves_its_traceback_under_its_reference_in_the_log(
    restored_root_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Stderr and the log file hold the traceback the caller's reference names."""
    log_dir = tmp_path / "logs"
    configure_logging(str(log_dir))

    handler = failing_handler("/predict", "predict_response_bytes")
    handler.do_POST()

    reference = internal_error_reference(handler)
    for log_text in (capsys.readouterr().err, (log_dir / LOG_FILE_NAME).read_text()):
        assert f"ERROR {HANDLER_LOGGER}:" in log_text
        assert reference in log_text
        assert "Traceback (most recent call last)" in log_text
        assert f"RuntimeError: {INTERNAL_DETAIL}" in log_text


def test_without_a_log_dir_the_log_reaches_stderr_from_info_up(
    restored_root_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The service's lifecycle records are ``INFO``, and a container collects stderr."""
    configure_logging(None)
    service_logger = logging.getLogger(service_server.__name__)

    service_logger.debug("not shown")
    service_logger.info("shown")

    err = capsys.readouterr().err
    assert f"INFO {service_server.__name__}: shown" in err
    assert "not shown" not in err


def test_the_log_file_rolls_over_and_keeps_a_bounded_number_of_files(
    restored_root_logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The service's logs never outgrow ``LOG_FILE_COUNT`` files of bounded size."""
    monkeypatch.setattr(service_server, "LOG_FILE_MAX_BYTES", SMALL_LOG_FILE_BYTES)
    configure_logging(str(tmp_path))
    service_logger = logging.getLogger(service_server.__name__)

    for index in range(ROLLOVER_RECORDS):
        service_logger.info("record %d", index)

    rolled = [f"{LOG_FILE_NAME}.{n}" for n in range(1, LOG_FILE_COUNT)]
    assert sorted(path.name for path in tmp_path.iterdir()) == [LOG_FILE_NAME, *rolled]
    assert f"record {ROLLOVER_RECORDS - 1}" in (tmp_path / LOG_FILE_NAME).read_text()
