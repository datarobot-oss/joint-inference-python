"""Answering one conditioning request at one emission position.

This module owns the served ``condition`` query mode's algebra, one layer below
the transport: it takes a mixture head's emitted parameters, the columns a
request pins or bounds, and returns an object whose operations mirror the
public return modes.

The two condition kinds compose in a fixed order, and both steps run through
the same closed form:

1. The equality columns are conditioned first, exactly, which keeps the mixture
   in its family. The normalizer of the resulting unnormalized component
   weights is the density the model assigns to the pinned values.
2. The interval columns are then confined on that reduced mixture by the
   distribution-function walk in :mod:`.interval_conditioning`. Each draw of
   the interval block is conditioned on *again*, by the same equality
   conditional, which turns the answer into one pooled mixture over the
   read-out columns: a draw contributes its own conditional, with weight
   ``1/S``. Every read-out then reuses the head's own sampler and scorer on
   that pooled mixture rather than a second implementation of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from jointfm_inference.model.jointfm_1.head import AUTO_STANDARDIZATION_METADATA_KEYS
from jointfm_inference.model.linalg import robust_cholesky

from .adapter import PredictorAdapter
from .conditioning import conditioned_log_density
from .interval_conditioning import effective_sample_size, walk_box

# Quasi-random points per component behind a multi-column box probability.
# The walk's error falls roughly as the reciprocal of this count, and its
# working set is one emission position wide, so the default buys about three
# significant digits on a deep-tail region at negligible cost.
DEFAULT_INTERVAL_POINTS = 8192


def _mixture_shapes(components: int, width: int) -> Dict[str, tuple[int, ...]]:
    """Trailing shapes of one conditioned mixture parameter mapping, per name.

    An unexpected key raises here instead of being dropped silently.
    """
    return {
        "logits": (components,),
        "means": (components, width),
        "chol": (components, width, width),
        "df": (components,),
    }


def _reshape_mixture(
    params: Dict[str, Tensor],
    leading: tuple[int, ...],
    *,
    components: int,
    width: int,
) -> Dict[str, Tensor]:
    """Reshape a mixture parameter mapping onto new leading axes."""
    shapes = _mixture_shapes(components, width)
    return {
        name: value.reshape(*leading, *shapes[name]) for name, value in params.items()
    }


def _expand_mixture(
    params: Dict[str, Tensor],
    leading: tuple[int, ...],
    *,
    components: int,
    width: int,
) -> Dict[str, Tensor]:
    """Broadcast a mixture parameter mapping onto new leading axes."""
    shapes = _mixture_shapes(components, width)
    return {
        name: value.expand(*leading, *shapes[name]) for name, value in params.items()
    }


@dataclass(frozen=True)
class ConditionRequest:
    """One conditioning request, resolved to column indices.

    Columns are indices into the request's columns -- the leading positions of
    the head's emitted joint at one emission position -- whatever their
    canonical role, because the head emits a joint over all of them. Bounds
    may be infinite on one side; a column carries at most one condition of
    either kind.

    Attributes:
        horizon: Index of the emission position every condition refers to.
        equality_columns: Columns pinned to a value.
        equality_values: The pinned values, positionally aligned.
        interval_columns: Columns confined to a range.
        interval_lower: Lower bounds, ``-inf`` for an open side.
        interval_upper: Upper bounds, ``+inf`` for an open side.
    """

    horizon: int
    equality_columns: tuple[int, ...] = ()
    equality_values: tuple[float, ...] = ()
    interval_columns: tuple[int, ...] = ()
    interval_lower: tuple[float, ...] = ()
    interval_upper: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Reject a request that is malformed independently of any head."""
        if self.horizon < 0:
            raise ValueError(f"horizon must not be negative, got {self.horizon}")
        if len(self.equality_columns) != len(self.equality_values):
            raise ValueError(
                "equality_columns and equality_values must have the same length"
            )
        if (
            not len(self.interval_columns)
            == len(self.interval_lower)
            == len(self.interval_upper)
        ):
            raise ValueError(
                "interval_columns, interval_lower and interval_upper must have "
                "the same length"
            )
        if not self.equality_columns and not self.interval_columns:
            raise ValueError("a conditioning request must carry at least one condition")
        conditioned = self.equality_columns + self.interval_columns
        if len(set(conditioned)) != len(conditioned):
            raise ValueError("a column carries at most one condition")
        if any(column < 0 for column in conditioned):
            raise ValueError("condition columns must not be negative")
        for value in self.equality_values:
            if not float(value) == float(value) or abs(float(value)) == float("inf"):
                raise ValueError("an equality condition needs a finite value")
        for lower, upper in zip(self.interval_lower, self.interval_upper):
            if not lower < upper:
                raise ValueError(
                    f"an interval needs lower < upper, got [{lower}, {upper}]"
                )
            if abs(lower) == float("inf") and abs(upper) == float("inf"):
                raise ValueError("an interval open on both sides is not a condition")

    @property
    def conditioned_columns(self) -> tuple[int, ...]:
        """Every column this request conditions, in ascending order."""
        return tuple(sorted(self.equality_columns + self.interval_columns))

    def read_out_columns(self, column_count: int) -> tuple[int, ...]:
        """Columns of the position whose conditional distribution is described."""
        conditioned = set(self.conditioned_columns)
        return tuple(
            column for column in range(column_count) if column not in conditioned
        )


class ConditionalReadOut:
    """The conditional at one emission position, as the return modes see it.

    Operations return tensors over every column of the request's position in
    the neutral result-shape contract with a single emission position on the horizon axis,
    so the response builder's column slicing applies unchanged. A conditioned
    column carries what the request said about it: its pinned value, or the
    truncated draw the interval step produced.

    The draw budget is bound at construction, because the interval draws are
    made once and every read-out is a view of the same conditional object.

    Two accounting numbers ride along with the read-out, and they chain rather
    than describe separate things. ``equality_log_density`` is the joint density
    of *every* pinned value at once, under the position's own mixture.
    ``region_log_probability`` is the joint probability of the *whole* box, under
    the mixture the equality step has already reduced -- so on a mixed request it
    is the probability of the box **given the pins**, not the box's own. Their
    sum is therefore the plausibility of the whole condition set, exactly and
    with no independence assumption, and it is a density times a probability:
    unit-carrying, so only differences between comparable requests mean anything.
    Neither number is a product over columns, and a caller must not rebuild
    either one by multiplying per-column marginals.
    """

    def __init__(
        self,
        *,
        pooled: PredictorAdapter,
        sample_source: PredictorAdapter,
        sample_count: int,
        one_sample_per_draw: bool,
        sample_bounds: tuple[Tensor | float | None, Tensor | float | None],
        read_out_columns: tuple[int, ...],
        column_count: int,
        request: ConditionRequest,
        interval_draws: Tensor | None,
        equality_log_density: Tensor | None,
        region_log_probability: Tensor | None,
        interval_effective_sample_size: Tensor | None,
        interval_points: int | None,
        rng: torch.Generator | None,
    ) -> None:
        """Bind the pooled conditional mixture and its accounting.

        Draws are taken when ``samples`` asks for them, not here, so a ``mean``
        or ``log_prob`` read-out pays for no sampling at all.
        """
        self._pooled = pooled
        self._sample_source = sample_source
        self._sample_count = sample_count
        self._one_sample_per_draw = one_sample_per_draw
        self._sample_bounds = sample_bounds
        self._rng = rng
        self._read_out_columns = read_out_columns
        self._column_count = column_count
        self._request = request
        self._interval_draws = interval_draws
        self.equality_log_density = equality_log_density
        self.region_log_probability = region_log_probability
        self.interval_effective_sample_size = interval_effective_sample_size
        self.interval_points = interval_points

    @property
    def read_out_columns(self) -> tuple[int, ...]:
        """Columns whose conditional distribution this read-out describes."""
        return self._read_out_columns

    @property
    def sample_count(self) -> int:
        """Draws behind every read-out of this conditional."""
        return self._sample_count

    def _draw_read_out(self) -> Tensor:
        """Draw ``(B, S, 1, r)`` read-out values from the conditional.

        With interval conditions the source carries one conditional per
        truncated draw on its leading axis, so one sample each pairs a read-out
        with the draw it was conditioned on; without them the conditional is a
        single mixture and the draws come from it directly.
        """
        minimum, maximum = self._sample_bounds
        if not self._one_sample_per_draw:
            return self._sample_source.sample(
                n_samples=self._sample_count,
                rng=self._rng,
                min_value=minimum,
                max_value=maximum,
            )
        width = len(self._read_out_columns)
        drawn = self._sample_source.sample(
            n_samples=1, rng=self._rng, min_value=minimum, max_value=maximum
        )
        return drawn.reshape(-1, self._sample_count, 1, width)

    def _full_width(self, read_out_values: Tensor) -> Tensor:
        """Place read-out values into a tensor carrying every column of the position."""
        shape = list(read_out_values.shape)
        shape[-1] = self._column_count
        full = torch.zeros(
            shape, dtype=read_out_values.dtype, device=read_out_values.device
        )
        full[..., list(self._read_out_columns)] = read_out_values
        return full

    def mean(self) -> Tensor:
        """Return the ``(B, 1, D)`` conditional mean.

        Exact for the read-out columns: the pooled mixture's mean is a weighted
        sum of component means, so no sampling enters, and with interval
        conditions it is the average of exact conditional means over the draws.
        A Student-t component's mean is its location only above one degree of
        freedom, which the head guarantees by flooring the emitted value;
        conditioning raises the degrees of freedom further, so the conditional
        mean is defined wherever the unconditioned one is.
        """
        weights = F.softmax(self._pooled.params["logits"], dim=-1).unsqueeze(-1)
        read_out_mean = (weights * self._pooled.params["means"]).sum(dim=-2)
        full = self._full_width(read_out_mean)
        for column, value in zip(
            self._request.equality_columns, self._request.equality_values
        ):
            full[..., column] = value
        if self._interval_draws is not None:
            drawn_mean = self._interval_draws.mean(dim=1).to(full.dtype)
            for position, column in enumerate(self._request.interval_columns):
                full[..., column] = drawn_mean[..., position].unsqueeze(-1)
        return full

    def samples(self) -> Tensor:
        """Return ``(B, S, 1, D)`` draws from the conditional.

        With interval conditions each row pairs one truncated draw of the
        interval columns with a read-out drawn from the equality conditional at
        that same draw, so the rows are exact draws from the joint conditional
        rather than independently summarized marginals.
        """
        full = self._full_width(self._draw_read_out())
        for column, value in zip(
            self._request.equality_columns, self._request.equality_values
        ):
            full[..., column] = value
        if self._interval_draws is not None:
            drawn = self._interval_draws.to(full.dtype).unsqueeze(2)
            for position, column in enumerate(self._request.interval_columns):
                full[..., column] = drawn[..., position]
        return full

    def quantiles(self, q: Sequence[float]) -> Tensor:
        """Return ``(Q, B, 1, D)`` marginal quantiles of the conditional."""
        read_out_quantiles = self._pooled.quantiles(
            q=q, n_samples=self.sample_count, rng=self._rng
        )
        full = self._full_width(read_out_quantiles)
        for column, value in zip(
            self._request.equality_columns, self._request.equality_values
        ):
            full[..., column] = value
        if self._interval_draws is not None:
            probabilities = torch.tensor(
                list(q), dtype=self._interval_draws.dtype, device=full.device
            )
            drawn_quantiles = torch.quantile(self._interval_draws, probabilities, dim=1)
            for position, column in enumerate(self._request.interval_columns):
                full[..., column] = drawn_quantiles[..., position].unsqueeze(-1)
        return full

    def log_prob(self, values: Tensor) -> Tensor:
        """Score full-width ``(B, 1, D)`` values under the conditional.

        Only the read-out columns are scored, because the conditioned columns
        are what the request fixed rather than what it asks about. Values that
        contradict the request — a pinned column given a different value, or a
        bounded column outside its range — are rejected instead of scored,
        since they ask for the density of an event the request excluded.
        """
        if values.shape[-1] != self._column_count:
            raise ValueError(
                f"log_prob values must carry all {self._column_count} columns, got "
                f"{int(values.shape[-1])}"
            )
        for column, value in zip(
            self._request.equality_columns, self._request.equality_values
        ):
            supplied = values[..., column]
            if not torch.allclose(
                supplied, torch.full_like(supplied, float(value)), atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    f"log_prob values contradict the equality condition on column "
                    f"{column}: the request pinned {value}"
                )
        for position, column in enumerate(self._request.interval_columns):
            supplied = values[..., column]
            lower = self._request.interval_lower[position]
            upper = self._request.interval_upper[position]
            if bool(((supplied < lower) | (supplied > upper)).any().item()):
                raise ValueError(
                    f"log_prob values fall outside the interval condition on column "
                    f"{column}: the request confined it to [{lower}, {upper}]"
                )
        return self._pooled.log_prob(values[..., list(self._read_out_columns)])


# The parameter keys a conditioned position may carry; anything else is a
# defect rather than a layout to guess at.
_PARAMETER_KEYS = frozenset(
    {"logits", "df", "means", "lowrank_diag", "lowrank_U", "chol"}
)

# The emitted mapping also carries the standardization statistics of the history
# window. They are one window-wide pair, not a parameter of the joint at an
# emission position, and the head emits its parameters in data space, so the
# conditioned mixture drops them.
_NON_PARAMETER_KEYS = frozenset(AUTO_STANDARDIZATION_METADATA_KEYS)


def _position_params(
    adapter: PredictorAdapter, horizon: int, column_count: int
) -> Dict[str, Tensor]:
    """Slice every emitted parameter to one position.

    The standardization metadata next to the parameters is dropped.

    Raises:
        RuntimeError: when the emitted width is not ``column_count``, which is
            a defect of the backend rather than of the request.
        ValueError: when ``horizon`` is outside the emitted positions, or when
            a key is not a
            known parameter.
    """
    logits = adapter.params["logits"]
    horizon_count = int(logits.shape[1])
    if not 0 <= horizon < horizon_count:
        raise ValueError(
            f"horizon {horizon} is out of range for {horizon_count} emitted positions"
        )
    emitted_width = int(adapter.params["means"].shape[-1])
    if emitted_width != column_count:
        raise RuntimeError(
            f"the parameters cover {emitted_width} columns, the request declares "
            f"{column_count}"
        )
    sliced: Dict[str, Tensor] = {}
    for name, value in adapter.params.items():
        if name in _NON_PARAMETER_KEYS:
            continue
        if name not in _PARAMETER_KEYS:
            raise ValueError(f"emitted parameter {name!r} is not a mixture parameter")
        sliced[name] = value[:, horizon : horizon + 1]
    return sliced


def _read_out_bound(
    bound: Tensor | float | None, read_out_columns: tuple[int, ...]
) -> Tensor | float | None:
    """Narrow a full-width per-column sampling bound to the read-out columns."""
    if bound is None or not isinstance(bound, Tensor):
        return bound
    if bound.ndim == 0:
        return bound
    return bound[..., list(read_out_columns)]


def condition_at_position(
    adapter: PredictorAdapter,
    request: ConditionRequest,
    *,
    n_samples: int,
    interval_points: int = DEFAULT_INTERVAL_POINTS,
    rng: torch.Generator | None = None,
    min_value: Tensor | float | None = None,
    max_value: Tensor | float | None = None,
    column_count: int,
) -> ConditionalReadOut:
    """Answer one conditioning request against the head's emitted joint.

    Args:
        adapter: Batched adapter over the executed request's emitted
            parameters, shaped ``(B, H, ...)``.
        request: The conditions, resolved to column indices of one position.
        n_samples: Draw budget; it is both the interval step's resampled draw
            count and the sample count a ``samples`` read-out returns.
        interval_points: Quasi-random points per component behind a
            multi-column box probability. The walk never runs on fewer points
            than ``n_samples``, and the read-out reports the pooled count over
            all components, so its effective sample size is a share of it.
        rng: Optional generator making the answer reproducible.
        min_value: Optional per-column lower rejection bounds from the
            caller's column metadata, one per request column, applied to the
            drawn read-out columns exactly as the unconditional sampling path
            applies them.
        max_value: Optional per-column upper rejection bounds, likewise.
        column_count: Number of columns the request declares, which the
            emitted parameters must match.

    Returns:
        A :class:`ConditionalReadOut` over the conditioned position.

    Raises:
        ValueError: when the request names a column the position does not have,
            leaves no read-out column, or asks for a non-positive draw budget.
        RuntimeError: when the emitted parameters do not cover exactly
            ``column_count`` columns.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    position = PredictorAdapter(
        adapter.head, _position_params(adapter, request.horizon, column_count)
    )
    for column in request.conditioned_columns:
        if column >= column_count:
            raise ValueError(
                f"condition names column {column}, but the conditioned position "
                f"has {column_count} columns"
            )
    read_out_columns = request.read_out_columns(column_count)
    if not read_out_columns:
        if request.interval_columns:
            raise ValueError(
                "a request must leave at least one column unconditioned: what it "
                "reads out is the conditional distribution of the columns it does "
                "not condition"
            )
        raise ValueError(
            "a request pinning every column of the position leaves nothing to read "
            "out; the joint density of those values is what the log_prob return "
            "mode answers"
        )

    logits = position.params["logits"]
    batch_count, _, component_count = logits.shape
    device = logits.device
    dtype = logits.dtype

    equality_mask = torch.zeros(column_count, dtype=torch.bool, device=device)
    equality_values = torch.zeros(
        batch_count, 1, column_count, dtype=dtype, device=device
    )
    if request.equality_columns:
        pinned = torch.tensor(request.equality_values, dtype=dtype, device=device)
        equality_mask[list(request.equality_columns)] = True
        equality_values[..., list(request.equality_columns)] = pinned

    reduced = position.condition(equality_values, equality_mask)
    equality_log_density = (
        conditioned_log_density(reduced.params["logits"])
        if request.equality_columns
        else None
    )

    surviving = [
        column
        for column in range(column_count)
        if column not in set(request.equality_columns)
    ]
    read_out_local = [surviving.index(column) for column in read_out_columns]

    reduced_df = reduced.params["df"]

    if not request.interval_columns:
        pooled_params = dict(reduced.params)
        pooled_params["logits"] = F.log_softmax(reduced.params["logits"], dim=-1)
        pooled_params["means"] = reduced.params["means"][..., read_out_local]
        pooled_params["chol"] = reduced.params["chol"][..., read_out_local, :][
            ..., :, read_out_local
        ]
        pooled = PredictorAdapter(adapter.head, pooled_params)
        return ConditionalReadOut(
            pooled=pooled,
            sample_source=pooled,
            sample_count=n_samples,
            one_sample_per_draw=False,
            sample_bounds=(
                _read_out_bound(min_value, read_out_columns),
                _read_out_bound(max_value, read_out_columns),
            ),
            rng=rng,
            read_out_columns=read_out_columns,
            column_count=column_count,
            request=request,
            interval_draws=None,
            equality_log_density=equality_log_density,
            region_log_probability=None,
            interval_effective_sample_size=None,
            interval_points=None,
        )

    interval_local = [surviving.index(column) for column in request.interval_columns]
    block_index = torch.tensor(interval_local, dtype=torch.long, device=device)
    reduced_chol = reduced.params["chol"]
    # The scale matrix of the interval columns alone, which is a marginal of the
    # reduced mixture: a Student-t mixture marginalizes by dropping rows and
    # columns and keeps the joint's degrees of freedom unchanged.
    scale = reduced_chol @ reduced_chol.transpose(-1, -2)
    block_scale = scale.index_select(-1, block_index).index_select(-2, block_index)
    lower = torch.tensor(request.interval_lower, dtype=torch.float64, device=device)
    upper = torch.tensor(request.interval_upper, dtype=torch.float64, device=device)
    # A point is reused whenever the resampling draws it twice, so the walk
    # never runs on fewer points than the draw budget it has to feed.
    walk_points = max(interval_points, n_samples)
    walk = walk_box(
        reduced.params["means"].index_select(-1, block_index),
        robust_cholesky(block_scale),
        lower,
        upper,
        points=walk_points,
        df=reduced_df,
        rng=rng,
    )

    # The region's probability weighs each component's box probability by that
    # component's posterior weight after the equality step.
    component_log_weights = F.log_softmax(reduced.params["logits"], dim=-1)
    region_log_probability = torch.logsumexp(
        component_log_weights + walk.log_box_probability.to(dtype), dim=-1
    )

    # One weighted set over components and points: resampling it gives draws
    # from the truncated mixture, every one inside the box.
    pooled_log_weights = (
        component_log_weights.to(torch.float64).unsqueeze(-1) + walk.log_weights
    ).reshape(batch_count, -1)
    draw_index = torch.multinomial(
        F.softmax(pooled_log_weights, dim=-1),
        n_samples,
        replacement=True,
        generator=rng,
    )
    block_width = len(interval_local)
    flat_draws = walk.draws.reshape(batch_count, -1, block_width)
    interval_draws = torch.gather(
        flat_draws, 1, draw_index.unsqueeze(-1).expand(-1, -1, block_width)
    )
    # A single interval column needs no estimation: its box probability is one
    # CDF difference and its draws are inverse CDFs, so every weight inside a
    # component is the same and there is no estimator to characterize.
    interval_sample_size = (
        None
        if walk.is_exact
        else torch.stack([effective_sample_size(row) for row in pooled_log_weights])
    )
    pooled_points = None if walk.is_exact else walk_points * component_count

    # Conditioning the reduced mixture on each draw rides the emission-position
    # axis, which the conditioning core treats as one more independent leading
    # axis; the mask is the same interval block for every draw.
    draw_values = torch.zeros(
        batch_count, n_samples, len(surviving), dtype=dtype, device=device
    )
    draw_values[..., interval_local] = interval_draws.to(dtype)
    interval_mask = torch.zeros(len(surviving), dtype=torch.bool, device=device)
    interval_mask[interval_local] = True
    expanded = _expand_mixture(
        reduced.params,
        (batch_count, n_samples),
        components=component_count,
        width=len(surviving),
    )
    per_draw = PredictorAdapter(adapter.head, expanded).condition(
        draw_values, interval_mask
    )

    read_out_width = len(read_out_columns)
    normalized = dict(per_draw.params)
    normalized["logits"] = F.log_softmax(per_draw.params["logits"], dim=-1) - torch.log(
        torch.tensor(float(n_samples), dtype=dtype, device=device)
    )
    pooled_params = _reshape_mixture(
        normalized,
        (batch_count, 1),
        components=n_samples * component_count,
        width=read_out_width,
    )
    pooled = PredictorAdapter(adapter.head, pooled_params)

    folded = _reshape_mixture(
        per_draw.params,
        (batch_count * n_samples, 1),
        components=component_count,
        width=read_out_width,
    )
    return ConditionalReadOut(
        pooled=pooled,
        sample_source=PredictorAdapter(adapter.head, folded),
        sample_count=n_samples,
        one_sample_per_draw=True,
        sample_bounds=(
            _read_out_bound(min_value, read_out_columns),
            _read_out_bound(max_value, read_out_columns),
        ),
        rng=rng,
        read_out_columns=read_out_columns,
        column_count=column_count,
        request=request,
        interval_draws=interval_draws,
        equality_log_density=equality_log_density,
        region_log_probability=region_log_probability,
        interval_effective_sample_size=interval_sample_size,
        interval_points=pooled_points,
    )


__all__ = [
    "DEFAULT_INTERVAL_POINTS",
    "ConditionRequest",
    "ConditionalReadOut",
    "condition_at_position",
]
