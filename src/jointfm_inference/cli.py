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

"""The ``jointfm-inference`` command: ``serve``, ``probe``, ``profile``, ``schema``.

``serve`` takes no options: every setting is a ``JOINTFM_INFERENCE_*``
environment variable (``jointfm_inference.model.service.config``), so a local
process and a container read the same configuration file. ``profile`` loads
the checkpoint from the same settings. ``schema`` writes the OpenAPI document
of the HTTP contract, which needs no checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from jointfm_inference.model import probe, profile
from jointfm_inference.model.service import server, wire_schema

SERVE_DESCRIPTION = (
    "Serve one JointFM checkpoint over HTTP. Configured entirely through "
    "JOINTFM_INFERENCE_* environment variables; JOINTFM_INFERENCE_CHECKPOINT_PATH "
    "is required, and JOINTFM_INFERENCE_CHECKPOINT_VERSION defaults to the start "
    "of the checkpoint's SHA-256 digest."
)
SCHEMA_DESCRIPTION = (
    "Write the OpenAPI 3.1 document of the service's HTTP contract, generated "
    "from the service's own code. With --check, write nothing and fail when the "
    "file at --output differs from the document."
)


def _serve(arguments: argparse.Namespace) -> None:
    """Run the HTTP service until the process is stopped."""
    del arguments
    server.main()


def _schema(arguments: argparse.Namespace) -> None:
    """Write the OpenAPI document to ``--output``, or check that file with ``--check``.

    Raises:
        SystemExit: under ``--check``, when the file is missing or differs
            from the document the code generates.
    """
    output = Path(arguments.output)
    document = wire_schema.render_openapi()
    if arguments.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != document:
            raise SystemExit(
                f"{output} is not the OpenAPI document the code generates; "
                f"rewrite it with: jointfm-inference schema --output {output}"
            )
        print(f"{output} matches the generated OpenAPI document")
        return
    output.write_text(document, encoding="utf-8")
    print(f"wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the ``jointfm-inference`` command and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="jointfm-inference",
        description=(
            "Serve JointFM checkpoints, probe running services, profile "
            "served-request latency, and write the service's OpenAPI document."
        ),
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
    profile_parser = commands.add_parser(
        "profile",
        help="Time each stage of served requests on the configured checkpoint",
        description=profile.PROFILE_DESCRIPTION,
    )
    profile.add_arguments(profile_parser)
    profile_parser.set_defaults(handler=profile.run)
    schema_parser = commands.add_parser(
        "schema",
        help="Write the OpenAPI document of the HTTP contract",
        description=SCHEMA_DESCRIPTION,
    )
    schema_parser.add_argument(
        "--output",
        default=wire_schema.OPENAPI_FILE_NAME,
        help="The file to write or check (default: %(default)s)",
    )
    schema_parser.add_argument(
        "--check",
        action="store_true",
        help="Write nothing; fail when the file differs from the document",
    )
    schema_parser.set_defaults(handler=_schema)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse ``argv`` and run the selected subcommand."""
    arguments = build_parser().parse_args(argv)
    arguments.handler(arguments)
