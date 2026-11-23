import unittest
from datetime import date,timedelta
from experiment import transform,inconsistency,prior

class AblationTests(unittest.TestCase):
 def test_B_E_equivalence(self):
  raw=[[0,.5,1.5,2.5,-1],[1,2,0,2,-1]]
  self.assertEqual(transform(raw,7.25,'B'),transform(raw,7.25,'E'))
 def test_C_and_D_do_not_silently_repair_guests_below_bookings(self):
  for v in ('C','D'):
   self.assertEqual(inconsistency(transform([[1],[0]],7,v))['guests_below_bookings'],1)
 def test_booking_minimum_and_guest_floor_are_separate(self):
  self.assertEqual(transform([[0],[2]],7,'C'),[[0],[2]])
  self.assertEqual(transform([[0],[2]],7,'D'),[[1],[2]])
  self.assertEqual(transform([[0],[2]],7,'B'),[[1],[7]])
 def test_rounding_matches_python_ties(self):
  self.assertEqual(transform([[.5,1.5,2.5],[0,0,0]],7,'C')[0],[0,2,2])
 def test_future_actuals_do_not_change_raw_predictions_or_floor(self):
  cutoff=date(2026,4,6)
  data={cutoff-timedelta(days=i):(1,4) for i in range(1,127)}
  altered=dict(data);altered[cutoff]=(1000,10000)
  for method in ('current_4week','robust_guests'):
   for target in (0,1):self.assertEqual(prior.predict(data,cutoff,target,method),prior.predict(altered,cutoff,target,method))
  self.assertEqual(prior.party_size(data,cutoff),prior.party_size(altered,cutoff))

if __name__=='__main__':unittest.main()
