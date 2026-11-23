import unittest
from datetime import date,timedelta
from unittest.mock import patch
import sys
import experiment as e


class AdaptiveExperimentTests(unittest.TestCase):
    def setUp(self):
        self.cutoff=date(2026,4,6)
        self.data={self.cutoff-timedelta(days=i):(2,10) for i in range(1,127)}

    def test_all_candidates_ignore_future_actuals_and_party_size(self):
        altered=dict(self.data)
        for i in range(56):
            altered[self.cutoff+timedelta(days=i)]=(99999,999999)
        for method in e.METHODS:
            for target in (0,1):
                self.assertEqual(e.predict(self.data,self.cutoff,target,method),e.predict(altered,self.cutoff,target,method))
        self.assertEqual(e.party_size(self.data,self.cutoff),e.party_size(altered,self.cutoff))

    def test_adaptive_tie_retains_four_week(self):
        self.assertEqual(e.predict(self.data,self.cutoff,0,'adaptive')[1],'current_4week')

    def test_trend_is_bounded_and_zero_history_stays_zero(self):
        data=dict(self.data)
        for i in range(1,15):data[self.cutoff-timedelta(days=i)]=(20,100)
        self.assertEqual(e.predict(data,self.cutoff,0,'trend_adjusted')[1],1.25)
        zeros={d:(0,0) for d in data}
        self.assertEqual(e.predict(zeros,self.cutoff,1,'trend_adjusted')[0],[0]*7)

    def test_robust_median_keeps_zeros_and_does_not_change_data(self):
        data=dict(self.data)
        for lag,g in ((4,0),(3,0),(2,10),(1,100)):
            data[self.cutoff-timedelta(weeks=lag)]=(1,g)
        before=dict(data)
        result=e.predict(data,self.cutoff,1,'robust_guests')[0]
        self.assertEqual(result[0],5)
        self.assertEqual(data,before)

    def test_operational_processing_matches_production_adjustments(self):
        sys.path.insert(0,str(e.ROOT))
        import predict_real as serving
        import pandas as pd
        bookings=[.5,1.5,-1,0,2.5,0,3]
        guests=[1,0,-2,0,10,8,0]
        rows=lambda values:[{'prediction':serving.clamp_int(v),'lower':0,'upper':100} for v in values]
        with patch.object(serving,'get_model_bundles',return_value=({},{})), \
             patch.object(serving,'predict_series',side_effect=[rows(bookings),rows(guests)]):
            result=serving.generate_forecast(pd.Timestamp('2026-04-06'),pd.Timestamp('2026-04-12'),{'typical_guests_per_booking':7.25})
        expected=[[r['predicted_bookings'] for r in result['forecast']], [r['predicted_total_guests'] for r in result['forecast']]]
        self.assertEqual(e.operational(bookings,guests,7.25),expected)


if __name__=='__main__':unittest.main()
