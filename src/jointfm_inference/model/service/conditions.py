"""Parsing the transport's condition list into resolved per-position requests.

The wire form names columns by name and the emission positions each condition
covers by index into the request's ``query_times``; the library works in
column indices of the head's emitted joint, one emission position at a time.
This module is the only place that translation happens, and it is where every
field rule of the condition envelope is enforced.

The resolution groups the conditions by position: every position some
condition covers becomes one :class:`ConditionRequest`, whose mask is uniform
across that position's leading axes by construction. A column carries at most
one condition per position, so two conditions on the same column must cover
disjoint positions.
"""

from __future__ import annotations

from collections import defaultdict
from http import HTTPStatus
from typing import Any, Mapping, Sequence

from jointfm_inference.model.inference.conditional_query import ConditionRequest

from .column_specs import ColumnSpec
from .constants import (
    CONDITION_KIND_EQUALITY,
    SUPPORTED_CONDITION_KINDS,
)
from .errors import ServiceError, ServiceErrorCode

_CONDITIONS_FIELD = "conditions"
_POSITIONS_KEY = "query_time_indices"
_EQUALITY_KEYS = frozenset({"column", "kind", "value", _POSITIONS_KEY})
_INTERVAL_KEYS = frozenset({"column", "kind", "lower", "upper", _POSITIONS_KEY})


def supported_condition_kinds(*, supports_conditioning: bool) -> tuple[str, ...]:
    """Return the condition kinds the mounted deployment can answer.

    Derived rather than configured, so a deployment cannot advertise a
    capability its checkpoint does not have. Both kinds arrive together for a
    backend that can condition at all: the equality step is the head's closed
    form and the interval step runs on the reduced mixture that step leaves,
    so no head has one without the other.

    Whether the mounted model conditions at all is not decided here: it is the
    serving backend's ``supports_conditioning``, so a request cannot be
    accepted by a guard the advertisement disagrees with.
    """
    return SUPPORTED_CONDITION_KINDS if supports_conditioning else ()


def _reject(message: str, *, field: str = _CONDITIONS_FIELD) -> ServiceError:
    """Build one pre-execution rejection for a malformed condition list."""
    return ServiceError(
        ServiceErrorCode.VALIDATION_ERROR,
        message,
        status=HTTPStatus.BAD_REQUEST,
        field=field,
    )


def _require_finite(raw: Any, *, label: str) -> float:
    """Coerce one bound or pinned value, rejecting anything unusable."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _reject(f"{label} must be a number")
    value = float(raw)
    if value != value or abs(value) == float("inf"):
        raise _reject(f"{label} must be finite")
    return value


def _parse_bound(entry: Mapping[str, Any], key: str, sign: float) -> float:
    """Read one interval bound, ``null`` meaning the side is open."""
    if key not in entry:
        raise _reject(f"an interval condition must carry {key!r}, null when open")
    raw = entry[key]
    if raw is None:
        return sign * float("inf")
    return _require_finite(raw, label=f"interval bound {key!r}")


def _parse_positions(
    entry: Mapping[str, Any], *, label: str, query_time_count: int
) -> tuple[int, ...]:
    """Read the positions one condition covers, ``null`` meaning every position."""
    if _POSITIONS_KEY not in entry:
        raise _reject(
            f"{label} must carry {_POSITIONS_KEY!r}, null for every requested "
            "future position"
        )
    raw = entry[_POSITIONS_KEY]
    if raw is None:
        return tuple(range(query_time_count))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise _reject(f"{label}.{_POSITIONS_KEY} must be a list or null")
    if not raw:
        raise _reject(
            f"{label}.{_POSITIONS_KEY} must name at least one position; null "
            "covers every requested future position"
        )
    for index in raw:
        if isinstance(index, bool) or not isinstance(index, int):
            raise _reject(f"{label}.{_POSITIONS_KEY} must hold integers")
        if not 0 <= index < query_time_count:
            raise _reject(
                f"{label}.{_POSITIONS_KEY} entry {index} is outside the "
                f"{query_time_count} requested future positions"
            )
    if len(set(raw)) != len(raw):
        raise _reject(f"{label}.{_POSITIONS_KEY} names a position more than once")
    return tuple(raw)


def parse_conditions(
    payload: Mapping[str, Any],
    *,
    column_specs: Sequence[ColumnSpec],
    query_time_count: int,
) -> tuple[ConditionRequest, ...]:
    """Resolve the request's ``conditions`` list into one request per position.

    Each condition names the positions it covers by index into ``query_times``,
    ``null`` covering all of them. Conditions on different columns may share a
    position, which is how one position mixes a pin and a band; two conditions
    on the same column must cover disjoint positions, because a column carries
    at most one condition per position. The list says what the request
    conditions, never what it reads back: the read-out projection is
    ``requested_columns`` alone, which may name a conditioned column of either
    kind like any other.

    Args:
        payload: The whole request payload; the list is read from
            ``conditions``.
        column_specs: Declared columns, in the order the head emits them.
        query_time_count: Number of future positions the request asks for.

    Returns:
        One :class:`ConditionRequest` per covered position, in column indices,
        ordered by position. A position no condition covers has none.

    Raises:
        ServiceError: with ``VALIDATION_ERROR`` for every malformed or
            self-contradictory list, before any model execution.
    """
    entries = payload.get(_CONDITIONS_FIELD)
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise _reject("condition queries must carry a conditions list")
    if not entries:
        raise _reject("conditions must carry at least one condition")

    column_index = {spec.name: index for index, spec in enumerate(column_specs)}
    equality: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)
    interval: defaultdict[int, list[tuple[int, float, float]]] = defaultdict(list)
    claimed: dict[tuple[str, int], int] = {}

    for position, entry in enumerate(entries):
        label = f"conditions[{position}]"
        if not isinstance(entry, Mapping):
            raise _reject(f"{label} must be an object")
        name = entry.get("column")
        if not isinstance(name, str):
            raise _reject(f"{label}.column must be a column name")
        if name not in column_index:
            raise _reject(f"{label} references unknown column {name!r}")

        kind = entry.get("kind")
        if kind not in SUPPORTED_CONDITION_KINDS:
            raise _reject(
                f"{label}.kind must be one of {list(SUPPORTED_CONDITION_KINDS)}, "
                f"got {kind!r}"
            )

        expected_keys = (
            _EQUALITY_KEYS if kind == CONDITION_KIND_EQUALITY else _INTERVAL_KEYS
        )
        surplus = set(entry) - expected_keys
        if surplus:
            raise _reject(
                f"{label} carries fields {sorted(surplus)} a {kind} condition has no use for"
            )

        horizons = _parse_positions(
            entry, label=label, query_time_count=query_time_count
        )
        overlap = [horizon for horizon in horizons if (name, horizon) in claimed]
        if overlap:
            raise _reject(
                f"column {name!r} carries more than one condition at "
                f"{_POSITIONS_KEY} {overlap}: {label} overlaps "
                f"conditions[{claimed[(name, overlap[0])]}]"
            )
        for horizon in horizons:
            claimed[(name, horizon)] = position

        if kind == CONDITION_KIND_EQUALITY:
            if "value" not in entry:
                raise _reject(f"{label} must carry 'value'")
            value = _require_finite(entry["value"], label=f"{label}.value")
            for horizon in horizons:
                equality[horizon].append((column_index[name], value))
            continue

        lower = _parse_bound(entry, "lower", -1.0)
        upper = _parse_bound(entry, "upper", 1.0)
        if lower >= upper:
            raise _reject(f"{label} needs lower < upper, got [{lower}, {upper}]")
        if abs(lower) == float("inf") and abs(upper) == float("inf"):
            raise _reject(f"{label} is open on both sides, which conditions nothing")
        for horizon in horizons:
            interval[horizon].append((column_index[name], lower, upper))

    requests: list[ConditionRequest] = []
    for horizon in sorted(set(equality) | set(interval)):
        pins = equality[horizon]
        bands = interval[horizon]
        if len(pins) + len(bands) >= len(column_specs):
            where = f" at {_POSITIONS_KEY} position {horizon}"
            if bands:
                raise _reject(
                    "a request must leave at least one column unconditioned"
                    f"{where}: what it reads out is the conditional distribution "
                    "of the columns it does not condition"
                )
            raise _reject(
                f"pinning every column{where} leaves nothing to read out; the "
                "joint density of those values is what the log_prob return mode "
                "answers"
            )
        try:
            requests.append(
                ConditionRequest(
                    horizon=horizon,
                    equality_columns=tuple(column for column, _ in pins),
                    equality_values=tuple(value for _, value in pins),
                    interval_columns=tuple(column for column, _, _ in bands),
                    interval_lower=tuple(lower for _, lower, _ in bands),
                    interval_upper=tuple(upper for _, _, upper in bands),
                )
            )
        except ValueError as error:
            raise _reject(str(error)) from error
    return tuple(requests)


__all__ = ["parse_conditions", "supported_condition_kinds"]
