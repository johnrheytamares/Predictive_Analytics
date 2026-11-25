"""Operational post-processing variants. Nonnegative integers + booking/guest consistency."""
from __future__ import annotations

from typing import Sequence


PARTY = 7.0


def clamp_int(value: float) -> int:
    return int(max(0, round(float(value))))


def floor_int(value: float) -> int:
    return int(max(0, int(float(value) // 1)))


def ensure_consistency(bookings: list[int], guests: list[int]) -> tuple[list[int], list[int]]:
    """guests >= bookings; if guests>0 then bookings>=1."""
    out_b, out_g = [], []
    for b, g in zip(bookings, guests):
        b = max(0, int(b))
        g = max(0, int(g))
        if g > 0 and b < 1:
            b = 1
        g = max(g, b)
        out_b.append(b)
        out_g.append(g)
    return out_b, out_g


def round_pair(bookings: Sequence[float], guests: Sequence[float]) -> tuple[list[int], list[int]]:
    return [clamp_int(b) for b in bookings], [clamp_int(g) for g in guests]


def aggregate_preserving_round(values: Sequence[float]) -> list[int]:
    """Largest-remainder rounding so integers nearly preserve the sum."""
    vals = [max(0.0, float(v)) for v in values]
    floors = [int(v // 1) for v in vals]
    target = int(round(sum(vals)))
    remainder = target - sum(floors)
    order = sorted(range(len(vals)), key=lambda i: (vals[i] - floors[i], -i), reverse=True)
    out = floors[:]
    for i in range(max(0, remainder)):
        out[order[i % len(out)]] += 1
    # If overshot because target < sum floors (shouldn't with nonnegative), clip later via consistency.
    if remainder < 0:
        order_down = sorted(range(len(vals)), key=lambda i: (vals[i] - floors[i], i))
        need = -remainder
        for i in order_down:
            if need <= 0:
                break
            take = min(out[i], need)
            out[i] -= take
            need -= take
    return [max(0, x) for x in out]


def apply_ops(
    bookings: Sequence[float],
    guests: Sequence[float],
    *,
    mode: str,
    party_size: float = PARTY,
    book_min_threshold: float = 0.0,
) -> tuple[list[int], list[int]]:
    """
    Modes:
      raw_round                 - independent rounding only (may violate consistency; fixed after)
      consistency_only          - round then guests>=bookings and book-min if guests>0
      book_min_only             - round + booking minimum if rounded guests>0 (no party floor)
      party_floor_only          - round + guests>=max(g, b*party, b); no forced book-min beyond consistency
      variant_b                 - production Variant B
      variant_a                 - production Variant A
      no_book_min_party         - round + party floor + guests>=bookings; never raise bookings for guests
      soft_book_min             - book-min only if raw bookings >= threshold OR rounded guests suggest demand
      aggregate_round           - aggregate-preserving round on each series, then consistency
      aggregate_round_party     - aggregate round + party floor + consistency
      floor_round               - floor instead of round, then consistency
      ceil_book_round_guest     - ceil bookings, round guests, then consistency
    """
    party = max(1.0, float(party_size))
    raw_b = [float(x) for x in bookings]
    raw_g = [float(x) for x in guests]

    if mode == "raw_round":
        b, g = round_pair(raw_b, raw_g)
        return ensure_consistency(b, g)

    if mode == "consistency_only":
        b, g = round_pair(raw_b, raw_g)
        return ensure_consistency(b, g)

    if mode == "book_min_only":
        b, g = round_pair(raw_b, raw_g)
        for i in range(len(b)):
            if g[i] > 0 and b[i] < 1:
                b[i] = 1
            g[i] = max(g[i], b[i])
        return b, g

    if mode == "party_floor_only":
        b, g = round_pair(raw_b, raw_g)
        for i in range(len(b)):
            g[i] = max(g[i], clamp_int(b[i] * party), b[i])
            if g[i] > 0 and b[i] < 1:
                # Consistency only: party floor on zero bookings with positive guests still needs a booking.
                b[i] = 1
                g[i] = max(g[i], clamp_int(b[i] * party), b[i])
        return b, g

    if mode == "variant_b":
        b, g = round_pair(raw_b, raw_g)
        for i in range(len(b)):
            if g[i] > 0 and b[i] < 1:
                b[i] = 1
            g[i] = max(g[i], b[i])
        return b, g

    if mode == "variant_a":
        b, g = round_pair(raw_b, raw_g)
        for i in range(len(b)):
            if g[i] > 0 and b[i] < 1:
                b[i] = 1
            g[i] = max(g[i], clamp_int(b[i] * party), b[i])
        return b, g

    if mode == "no_book_min_party":
        # Never invent a booking solely because guests rounded up; if guests>0 after rules, still consistent.
        b, g = round_pair(raw_b, raw_g)
        for i in range(len(b)):
            g[i] = max(g[i], clamp_int(b[i] * party), b[i])
            if g[i] > 0 and b[i] < 1:
                # Drop guest-only days instead of inventing bookings (operational alternative).
                g[i] = 0
        return b, g

    if mode == "soft_book_min":
        b, g = round_pair(raw_b, raw_g)
        thr = float(book_min_threshold)
        for i in range(len(b)):
            if g[i] > 0 and b[i] < 1 and raw_b[i] >= thr:
                b[i] = 1
            elif g[i] > 0 and b[i] < 1 and raw_b[i] < thr:
                g[i] = 0  # suppress guest-only false demand rather than invent booking
            g[i] = max(g[i], b[i])
        return b, g

    if mode == "soft_book_min_party":
        b, g = apply_ops(raw_b, raw_g, mode="soft_book_min", party_size=party, book_min_threshold=book_min_threshold)
        for i in range(len(b)):
            if b[i] > 0:
                g[i] = max(g[i], clamp_int(b[i] * party), b[i])
        return b, g

    if mode == "aggregate_round":
        b = aggregate_preserving_round(raw_b)
        g = aggregate_preserving_round(raw_g)
        return ensure_consistency(b, g)

    if mode == "aggregate_round_party":
        b = aggregate_preserving_round(raw_b)
        g = aggregate_preserving_round(raw_g)
        for i in range(len(b)):
            g[i] = max(g[i], clamp_int(b[i] * party), b[i])
            if g[i] > 0 and b[i] < 1:
                g[i] = 0
        return ensure_consistency(b, g)

    if mode == "floor_round":
        b = [floor_int(x) for x in raw_b]
        g = [floor_int(x) for x in raw_g]
        return ensure_consistency(b, g)

    if mode == "ceil_book_round_guest":
        b = [int(max(0, -(-float(x) // 1))) for x in raw_b]  # ceil for nonnegative
        g = [clamp_int(x) for x in raw_g]
        return ensure_consistency(b, g)

    raise ValueError(mode)


# Fixed ablation catalog (no holdout-specific choices).
ABLATIONS = [
    "consistency_only",
    "book_min_only",
    "party_floor_only",
    "variant_b",
    "variant_a",
    "no_book_min_party",
    "soft_book_min",
    "soft_book_min_party",
    "aggregate_round",
    "aggregate_round_party",
    "floor_round",
    "ceil_book_round_guest",
]

# Candidates eligible for past-only selection among operationally valid policies.
SELECTABLE = [
    dict(mode="consistency_only"),
    dict(mode="variant_b"),
    dict(mode="variant_a"),
    dict(mode="no_book_min_party"),
    dict(mode="soft_book_min", book_min_threshold=0.0),
    dict(mode="soft_book_min", book_min_threshold=0.5),
    dict(mode="soft_book_min_party", book_min_threshold=0.5),
    dict(mode="aggregate_round"),
    dict(mode="aggregate_round_party"),
    dict(mode="floor_round"),
]
