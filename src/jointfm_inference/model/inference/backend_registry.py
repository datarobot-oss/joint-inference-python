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

"""The inference backend registry: checkpoint generation to serving backend.

A distribution registers the serving backend of one generation as an entry
point of :data:`INFERENCE_BACKEND_ENTRY_POINT_GROUP`, named by the
generation's checkpoint generation marker and pointing at a
:class:`~jointfm_inference.model.inference.contract.ServingBackend` object. The registry
loads only the entry point a checkpoint names, so serving one generation never
imports another generation's backend, and no caller has to register anything
at startup: installing the distribution is the registration.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Final

from .contract import ServingBackend

INFERENCE_BACKEND_ENTRY_POINT_GROUP: Final = "jointfm_inference.backends"


class UnregisteredGenerationError(LookupError):
    """No installed distribution registers a serving backend for a generation."""


def registered_generations() -> tuple[str, ...]:
    """Return the generations with a registered serving backend, sorted."""
    return tuple(
        sorted(
            entry_point.name
            for entry_point in entry_points(group=INFERENCE_BACKEND_ENTRY_POINT_GROUP)
        )
    )


def resolve_serving_backend(generation: str) -> ServingBackend:
    """Return the serving backend registered for ``generation``.

    Raises:
        UnregisteredGenerationError: when no installed distribution registers
            ``generation``; there is no fallback to another backend.
        RuntimeError: when several distributions register ``generation``,
            since the registry cannot tell which one is meant.
        TypeError: when the registered object is not a ``ServingBackend``.
        ValueError: when the registered backend reports another generation
            than the entry point it is registered under.
    """
    matches = entry_points(group=INFERENCE_BACKEND_ENTRY_POINT_GROUP, name=generation)
    if not matches:
        raise UnregisteredGenerationError(
            f"no serving backend is registered for checkpoint generation "
            f"{generation!r}; registered generations: {list(registered_generations())}. "
            f"A distribution registers one under the "
            f"{INFERENCE_BACKEND_ENTRY_POINT_GROUP!r} entry-point group."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"several serving backends are registered for checkpoint generation "
            f"{generation!r}: {sorted(entry_point.value for entry_point in matches)}"
        )
    (entry_point,) = matches
    backend = entry_point.load()
    if not isinstance(backend, ServingBackend):
        raise TypeError(
            f"entry point {entry_point.value!r} registered for {generation!r} is "
            f"not a ServingBackend, got {type(backend).__name__}"
        )
    if backend.generation != generation:
        raise ValueError(
            f"entry point {entry_point.value!r} is registered for {generation!r} "
            f"but serves {backend.generation!r}"
        )
    return backend


__all__ = [
    "INFERENCE_BACKEND_ENTRY_POINT_GROUP",
    "UnregisteredGenerationError",
    "registered_generations",
    "resolve_serving_backend",
]
