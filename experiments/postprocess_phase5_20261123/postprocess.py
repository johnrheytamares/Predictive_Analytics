"""Operational post-processing variants for Phase 5. No model training."""
from __future__ import annotations

from statistics import median
from typing import Sequence


PRODUCTION_PARTY_SIZE = 7.0  # forecast_models_metadata.json typical_guests_per_booking


def clamp_int(value: float) -> int:
    return int(max(0, round(float(value))))


def apply_variant_a(
    bookings: Sequence[float],
    guests: Sequence[float],
    party_size: float = PRODUCTION_PARTY_SIZE,
) -> tuple[list[int], list[int]]:
    """Existing production rules from predict_real.generate_forecast."""
    out_b, out_g = [], []
    party = max(1.0, float(party_size))
    for raw_b, raw_g in zip(bookings, guests):
        b, g = clamp_int(raw_b), clamp_int(raw_g)
        if g > 0 and b < 1:
            b = 1
        g = max(g, clamp_int(b * party), b)
        out_b.append(b)
        out_g.append(g)
    return out_b, out_g


def apply_variant_b(
    bookings: Sequence[float],
    guests: Sequence[float],
) -> tuple[list[int], list[int]]:
    """Rounding/clamp + booking minimum + guests>=bookings; no party-size floor."""
    out_b, out_g = [], []
    for raw_b, raw_g in zip(bookings, guests):
        b, g = clamp_int(raw_b), clamp_int(raw_g)
        if g > 0 and b < 1:
            b = 1
        g = max(g, b)
        out_b.append(b)
        out_g.append(g)
    return out_b, out_g


def guest_floor_changes(
    bookings: Sequence[float],
    guests: Sequence[float],
    party_size: float = PRODUCTION_PARTY_SIZE,
) -> dict:
    """Isolate days where the party-size floor changes guests relative to variant B."""
    a_b, a_g = apply_variant_a(bookings, guests, party_size)
    b_b, b_g = apply_variant_b(bookings, guests)
    changed = [i for i, (ga, gb) in enumerate(zip(a_g, b_g)) if ga != gb]
    return {
        "changed_days": len(changed),
        "guest_total_increase": sum(a_g) - sum(b_g),
        "booking_total_delta": sum(a_b) - sum(b_b),
        "changed_indexes": changed,
    }


def party_size_from_history(history_rows: list[tuple[float, float]]) -> float:
    """Cutoff-local median guests/bookings on nonzero booking days (adaptive recipe)."""
    values = [g / b for b, g in history_rows if b > 0]
    return round(max(1.0, median(values)), 2) if values else 1.0
