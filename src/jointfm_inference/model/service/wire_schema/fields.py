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

"""Building blocks of the wire schema: described values, fields, and objects.

Every object on the wire is a tuple of :class:`WireField` entries, and every
enumerated vocabulary a tuple of :class:`WireValue` entries; the helpers here
turn them into the JSON Schema fragments of the OpenAPI document. JSON Schema
is the vocabulary OpenAPI 3.1 uses for payloads
(https://json-schema.org/draft/2020-12).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

COMPONENTS_PREFIX = "#/components/schemas/"


@dataclass(frozen=True)
class WireValue:
    """One value of an enumerated wire vocabulary, with what it means."""

    value: str
    description: str


@dataclass(frozen=True)
class WireField:
    """One field of a JSON object on the wire.

    ``schema`` is the JSON Schema of the field's value, without its
    description; ``required`` says whether every object of its kind carries
    the field.
    """

    name: str
    schema: Mapping[str, Any]
    description: str
    required: bool = False


def described_values(
    descriptions: Mapping[str, str], *, owned: Collection[str], vocabulary: str
) -> tuple[WireValue, ...]:
    """Pair every value of a vocabulary the code owns with its description.

    The tuple keeps the order of ``descriptions``, which is the order the
    document lists the values in.

    Raises:
        ValueError: when the described values differ from ``owned``: an
            undescribed value would reach the wire without documentation, and
            a described value the code does not own would be documented but
            refused.
    """
    missing = sorted(set(owned) - set(descriptions))
    surplus = sorted(set(descriptions) - set(owned))
    if missing or surplus:
        raise ValueError(
            f"the {vocabulary} descriptions must cover exactly the values the "
            f"code serves: undescribed {missing}, unserved {surplus}"
        )
    return tuple(WireValue(value, text) for value, text in descriptions.items())


def values_of(values: Sequence[WireValue]) -> tuple[str, ...]:
    """Return the plain values of one vocabulary, in document order."""
    return tuple(value.value for value in values)


def field_names(fields: Sequence[WireField]) -> frozenset[str]:
    """Return the names of one object's fields."""
    return frozenset(field.name for field in fields)


def ref(component: str) -> dict[str, str]:
    """Return a reference to one schema under ``components.schemas``."""
    return {"$ref": f"{COMPONENTS_PREFIX}{component}"}


def nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a schema that admits ``schema`` or JSON ``null``."""
    return {"anyOf": [dict(schema), {"type": "null"}]}


def enum_schema(values: Sequence[WireValue]) -> dict[str, Any]:
    """Return the string schema whose values are exactly ``values``."""
    return {"type": "string", "enum": list(values_of(values))}


def vocabulary_text(intro: str, values: Sequence[WireValue]) -> str:
    """Return ``intro`` followed by one Markdown bullet per value and its meaning."""
    bullets = "\n".join(f"- `{value.value}`: {value.description}" for value in values)
    return f"{intro}\n\n{bullets}"


def object_schema(
    fields: Sequence[WireField],
    *,
    description: str,
    title: str | None = None,
) -> dict[str, Any]:
    """Return the schema of a JSON object holding exactly ``fields``.

    A key outside ``fields`` fails validation, as the service refuses it.
    """
    schema: dict[str, Any] = {} if title is None else {"title": title}
    schema.update(
        {
            "type": "object",
            "description": description,
            "properties": {
                field.name: {**field.schema, "description": field.description}
                for field in fields
            },
            "required": [field.name for field in fields if field.required],
            "additionalProperties": False,
        }
    )
    return schema
