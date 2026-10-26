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

"""The ``jointfm-inference`` command: ``serve`` and ``probe``.

``serve`` takes no options: every setting is a ``JOINTFM_INFERENCE_*``
environment variable (``jointfm_inference.model.service.config``), so a local
process and a container read the same configuration file.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from jointfm_inference.model import probe
from jointfm_inference.model.service import server

SERVE_DESCRIPTION = (
    "Serve one JointFM checkpoint over HTTP. Configured entirely through "
    "JOINTFM_INFERENCE_* environment variables; JOINTFM_INFERENCE_CHECKPOINT_PATH "
    "is required, and JOINTFM_INFERENCE_CHECKPOINT_VERSION defaults to the start "
    "of the checkpoint's SHA-256 digest."
)


def _serve(arguments: argparse.Namespace) -> None:
    """Run the HTTP service until the process is stopped."""
    del arguments
    server.main()


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the ``jointfm-inference`` command and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="jointfm-inference",
        description="Serve JointFM checkpoints and probe running services.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser(
        "serve", help="Serve one checkpoint over HTTP", description=SERVE_DESCRIPTION
    )
    serve_parser.set_defaults(handler=_serve)
    probe_parser = commands.add_parser(
        "probe",
        help="Check health and every return mode of running services",
        description=probe.PROBE_DESCRIPTION,
    )
    probe.add_arguments(probe_parser)
    probe_parser.set_defaults(handler=probe.run)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse ``argv`` and run the selected subcommand."""
    arguments = build_parser().parse_args(argv)
    arguments.handler(arguments)
