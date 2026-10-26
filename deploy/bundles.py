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

"""Print the compute bundles the configured cluster offers Workloads.

Run by ``task deploy:bundles``, which loads the DataRobot credentials from
``.env``. Any bundle ID it prints is a valid ``JOINTFM_DEPLOY_BUNDLE``.
"""

from __future__ import annotations

import os

from stack import (
    DATAROBOT_API_TOKEN_ENV,
    DATAROBOT_ENDPOINT_ENV,
    DEFAULT_DEPLOY_BUNDLE,
    fetch_workload_bundles,
)

GIB = 1024**3


def main() -> None:
    """Print one row per bundle, CPU bundles first and GPUs by size."""
    bundles = fetch_workload_bundles(
        os.environ[DATAROBOT_ENDPOINT_ENV], os.environ[DATAROBOT_API_TOKEN_ENV]
    )
    rows = [("ID", "GPU", "VRAM", "CPU", "RAM", "")]
    for bundle in sorted(
        bundles, key=lambda b: (b.gpu, b.gpu_memory_bytes, b.memory_bytes)
    ):
        gpu = f"{bundle.gpu} x {bundle.gpu_type}" if bundle.gpu else "-"
        vram = f"{bundle.gpu_memory_bytes / GIB:g} GiB" if bundle.gpu else "-"
        rows.append(
            (
                bundle.id,
                gpu,
                vram,
                f"{bundle.cpu:g}",
                f"{bundle.memory_bytes / GIB:g} GiB",
                "default" if bundle.id == DEFAULT_DEPLOY_BUNDLE else "",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


if __name__ == "__main__":
    main()
