"""Chronological evaluation windows shared with prior offline experiments."""
from __future__ import annotations

from datetime import date, timedelta

SNAPSHOT_START = date(2025, 12, 1)
SNAPSHOT_END = date(2026, 11, 21)
SNAPSHOT_DAYS = 356
SNAPSHOT_BOOKINGS = 519
SNAPSHOT_GUESTS = 6669
RECORDED_INPUT_SHA256 = "0af4b7c020769453ef748f2302ca8176c9ced891e77d86db4f33ec5eb2a77cf1"

FIRST_WEEK_START = date(2026, 4, 6)
DEVELOPMENT_WEEKS = 24
RESERVED_WEEKS = 8
TOTAL_EVAL_WEEKS = DEVELOPMENT_WEEKS + RESERVED_WEEKS  # 32
MINIMUM_HISTORY_DAYS = 120


def week_starts(count: int = TOTAL_EVAL_WEEKS) -> list[date]:
    return [FIRST_WEEK_START + timedelta(weeks=i) for i in range(count)]


def development_cutoffs() -> list[date]:
    return week_starts(DEVELOPMENT_WEEKS)


def reserved_cutoffs() -> list[date]:
    return week_starts(TOTAL_EVAL_WEEKS)[DEVELOPMENT_WEEKS:]


def period_label(cutoff: date) -> str:
    if cutoff < FIRST_WEEK_START + timedelta(weeks=DEVELOPMENT_WEEKS):
        return "development"
    return "reserved"


def period_bounds() -> dict:
    last_dev = FIRST_WEEK_START + timedelta(weeks=DEVELOPMENT_WEEKS - 1)
    last_res = FIRST_WEEK_START + timedelta(weeks=TOTAL_EVAL_WEEKS - 1)
    return {
        "development": {
            "weeks": DEVELOPMENT_WEEKS,
            "first_cutoff": FIRST_WEEK_START.isoformat(),
            "last_cutoff": last_dev.isoformat(),
            "inclusive_end": (last_dev + timedelta(days=6)).isoformat(),
            "note": "2026-04-06 through 2026-09-20",
        },
        "reserved": {
            "weeks": RESERVED_WEEKS,
            "first_cutoff": (FIRST_WEEK_START + timedelta(weeks=DEVELOPMENT_WEEKS)).isoformat(),
            "last_cutoff": last_res.isoformat(),
            "inclusive_end": (last_res + timedelta(days=6)).isoformat(),
            "note": "2026-09-21 through 2026-11-15; previously inspected, not untouched validation",
        },
    }
