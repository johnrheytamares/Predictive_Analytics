import copy
import unittest
from datetime import date,timedelta
import analyze

class AnalysisTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for t in ('bookings','guests'):
            for i in range(32):
                cutoff=date(2026,4,6)+timedelta(weeks=i)
                for j in range(7):
                    rows.append(dict(method='weekday_baseline',target=t,cutoff=cutoff.isoformat(),
                        date=(cutoff+timedelta(days=j)).isoformat(),horizon_offset=j,
                        actual=1.,predicted=1.,error=0.,period='development' if i<24 else 'reserved'))
        return rows
    def test_alignment_accepts_exact_match(self):
        r=self.fixture();self.assertEqual(len(analyze.validate(r)),448)
    def test_contradictory_actual_rejected(self):
        r=self.fixture();other=copy.deepcopy(r[0]);other.update(method='other',actual=2.,error=-1.)
        with self.assertRaises(AssertionError):analyze.validate(r+[other])
    def test_duplicate_rejected(self):
        r=self.fixture()
        with self.assertRaises(AssertionError):analyze.validate(r+[r[0]])
    def test_horizon_rejected(self):
        r=self.fixture();r[0]['horizon_offset']=1
        with self.assertRaises(AssertionError):analyze.validate(r)
    def test_whole_missing_week_rejected(self):
        r=self.fixture();other=[dict(x,method='other') for x in r if x['cutoff']!='2026-04-06']
        with self.assertRaises(AssertionError):analyze.validate(r+other)
    def test_zero_wape_undefined_but_errors_retained(self):
        m=analyze.metrics([dict(actual=0,predicted=3)])
        self.assertIsNone(m['wape']);self.assertEqual(m['mae'],3)
    def test_weekly_cancellation_not_daily_improvement(self):
        rows=[dict(method='m',target='bookings',period='development',cutoff='2026-04-06',actual=1,predicted=p) for p in (0,2)]
        w=analyze.weekly(rows)[0]
        self.assertEqual(w['absolute_error'],0);self.assertEqual(w['daily_absolute_error'],2)

if __name__=='__main__':unittest.main()
