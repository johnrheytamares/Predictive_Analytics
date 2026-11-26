# ReserVision v3.2 RC — `guests_per_booking_sumratio` (isolated)

Shadow release candidate only. **Does not replace production v3.1.**

## What changed

| Component | v3.1 (rollback / production) | v3.2 RC |
|---|---|---|
| Bookings | Two-Stage + floor_round | **Identical** booking integers |
| Guests | Robust Median | Past-84-day Σg/Σb × bookings |
| Demographics | Period-preserving mix | Period-preserving mix |
| Holidays / party floors | None | None |

Booking integers stay identical because floor_round consistency still uses the v3.1 robust-median guest path as an auxiliary input; **served guest totals and intervals are sum-ratio only**.

## Build / verify

```bash
python prepare_rc.py
python -m unittest test_rc.py -v
```

## Rollback

Keep `production_v31` pointed at `../reserVision_v3_1_two_stage_rc_20261124`.
This RC stays `production_approved=false`.
