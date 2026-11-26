"""Focused v3.2 production adapter tests; no fitting, sync, database or deployment."""
from pathlib import Path
import asyncio, json, sys, unittest
from urllib.parse import urlencode
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from production_v32 import api, runtime

async def asgi(app, path, params=None, headers=None, method='GET', body=None):
    payload = b'' if body is None else (body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode())
    messages = []
    async def receive():
        return {'type': 'http.request', 'body': payload, 'more_body': False}
    async def send(m):
        messages.append(m)
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if body is not None and not any(k == b'content-type' for k, _ in hdrs):
        hdrs.append((b'content-type', b'application/json'))
    await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method,
               'scheme': 'http', 'path': path, 'raw_path': path.encode(),
               'query_string': urlencode(params or {}).encode(), 'headers': hdrs,
               'client': ('127.0.0.1', 1), 'server': ('127.0.0.1', 8002), 'root_path': ''}, receive, send)
    head = next(m for m in messages if m['type'] == 'http.response.start')
    raw = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
    parsed = json.loads(raw) if raw else {}
    return head['status'], parsed, dict(head['headers'])

def get(path, params=None, headers=None):
    return asyncio.run(asgi(api.app, path, params, headers))

def query(start, days):
    return dict(start_date=start, end_date=str((pd.Timestamp(start) + pd.Timedelta(days=days - 1)).date()))


class ProductionV32ServingTests(unittest.TestCase):
    def test_readiness_identity_approval_disabled(self):
        status, body, _ = get('/health')
        self.assertEqual(status, 200)
        self.assertTrue(body['model_ready'])
        self.assertEqual(body['metadata']['model_version'], '3.2')
        self.assertFalse(body['metadata']['shadow_serving_only'])
        self.assertTrue(body['production_approved'])
        self.assertTrue(body['metadata']['production_approved'])
        self.assertEqual(body['metadata']['targets']['bookings']['selected_model'], 'two_stage')
        self.assertEqual(body['metadata']['targets']['guests']['selected_model'], 'guests_per_booking_sumratio')
        self.assertEqual(body['metadata']['postprocess']['selected_mode'], 'floor_round')
        guests = body['weekly_performance']['guests']['operational_floor_round']
        bookings = body['weekly_performance']['bookings']['operational_floor_round']
        self.assertAlmostEqual(bookings['mae'], 2.54, places=2)
        self.assertAlmostEqual(bookings['rmse'], 3.87, places=2)
        self.assertAlmostEqual(guests['mae'], 29.18, places=2)
        self.assertAlmostEqual(guests['rmse'], 40.94, places=2)
        self.assertIn('not a validated confidence level', guests['interval_disclaimer'].lower())
        self.assertFalse(body['lifecycle']['activation_disabled'])
        self.assertEqual(body['lifecycle']['rollback_commit'], runtime.ROLLBACK_COMMIT)

    def test_sample_jan_feb_2027(self):
        status, body, _ = get('/forecast', dict(start_date='2027-01-01', end_date='2027-02-08'))
        self.assertEqual(status, 200)
        self.assertEqual(body['model_version'], '3.2')
        self.assertEqual(sum(r['predicted_bookings'] for r in body['forecast']), 12)
        self.assertEqual(sum(r['predicted_total_guests'] for r in body['forecast']), 108)
        self.assertTrue(body['model']['production_approved'])
        self.assertFalse(body['model']['shadow_serving_only'])
        # Period-preserving mix: aggregate equals period allocator
        from predict_real import estimate_guest_mix, GUEST_MIX_CATEGORIES
        agg = {k: 0 for k in GUEST_MIX_CATEGORIES}
        for row in body['forecast']:
            mix = row['estimated_guest_mix']
            self.assertIsNotNone(mix)
            self.assertEqual(sum(mix.values()), row['predicted_total_guests'])
            for k, v in mix.items():
                agg[k] += v
        total = sum(r['predicted_total_guests'] for r in body['forecast'])
        self.assertEqual(agg, estimate_guest_mix(total, body['model_metadata']))

    def test_horizons_7_30_90(self):
        start = str(pd.Timestamp.now(tz='Asia/Manila').date())
        for days in (7, 30, 90):
            status, body, _ = get('/forecast', query(start, days))
            self.assertEqual(status, 200, msg=f'{days}d failed: {body}')
            self.assertEqual(len(body['forecast']), days)
            self.assertEqual(body['model_version'], '3.2')
            self.assertFalse(body['model']['runtime_optimization'])
            for r in body['forecast']:
                b, g = r['predicted_bookings'], r['predicted_total_guests']
                self.assertIs(type(b), int)
                self.assertIs(type(g), int)
                self.assertGreaterEqual(b, 0)
                self.assertGreaterEqual(g, b)
                if g > 0:
                    self.assertGreaterEqual(b, 1)
                # Sum-ratio intervals present
                self.assertLessEqual(r['guests_lower'], g)
                self.assertGreaterEqual(r['guests_upper'], g)

    def test_validate_release_gates(self):
        runtime.validate_release()
        self.assertTrue(runtime.PRODUCTION_APPROVED)


if __name__ == '__main__':
    unittest.main()
