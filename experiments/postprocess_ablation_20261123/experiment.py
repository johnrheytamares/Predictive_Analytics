"""Isolated post-processing ablation; no fitting, production edits or database access."""
import argparse
import csv
from datetime import date,timedelta
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
spec=importlib.util.spec_from_file_location('adaptive_experiment',HERE.parent/'adaptive_20261123'/'experiment.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
VARIANTS=('A','B','C','D','E')
PLAN={
 'A':'Raw nonnegative model expectations; no operational adjustment',
 'B':'Existing operational: Python round/clamp, positive-guests booking minimum, historical party-size floor and guests>=bookings',
 'C':'Python round and nonnegative clamp only; no consistency repair',
 'D':'C plus booking>=1 when rounded guests>0; no guest floor or silent guest consistency repair',
 'E':'C plus existing positive-guests booking rule, then historical party-size floor and guests>=bookings; equivalent to B',
 'configurations':{'current_pair':'four-week bookings + four-week guests','robust_pair':'four-week bookings + median-of-four-matching-weekdays guests'},
 'periods':{'development':'2026-04-06 through 2026-09-20 (24 weeks)','reserved':'2026-09-21 through 2026-11-15 (8 weeks)','overall':'all 32 weeks'},
 'limitations':['Reserved period has been inspected previously; NOT untouched independent validation',
                'All source imports/zeros retained. Input availability reconstructed by arrival date, not original booking creation time',
                'Refreshed recipe comparison, not replay of a frozen live artifact',
                'Low-demand thresholds remain <=5 bookings/week and <=35 guests/week',
                'Error contributions are sequential and order-dependent; not independent causal effects'],
}


def transform(raw,party,variant):
 if variant=='A':return [list(raw[0]),list(raw[1])]
 if variant=='B':return prior.operational(*raw,party)
 result=[[],[]]
 for b,g in zip(*raw):
  b,g=int(max(0,round(float(b)))),int(max(0,round(float(g))))
  if variant in ('D','E') and g>0 and b<1:b=1
  if variant=='E':g=max(g,int(max(0,round(b*party))),b)
  result[0].append(b);result[1].append(g)
 return result


def inconsistency(values):
 return {'positive_guests_zero_bookings':sum(g>0 and b==0 for b,g in zip(*values)),
         'positive_bookings_zero_guests':sum(b>0 and g==0 for b,g in zip(*values)),
         'guests_below_bookings':sum(g<b for b,g in zip(*values))}


def evaluate(data):
 rows=[];summaries=[];contributions=[];frequencies=[];quality=[]
 reference=json.loads((HERE.parent/'adaptive_20261123'/'results'/'results.json').read_text())
 for configuration,method in (('current_pair','current_4week'),('robust_pair','robust_guests')):
  weeks=[]
  for i in range(32):
   cutoff=date(2026,4,6)+timedelta(weeks=i)
   raw=[prior.predict(data,cutoff,t,method)[0] for t in (0,1)]
   actual=[[data[cutoff+timedelta(days=j)][t] for j in range(7)] for t in (0,1)]
   party=prior.party_size(data,cutoff)
   variants={v:transform(raw,party,v) for v in VARIANTS}
   if variants['B']!=variants['E']:raise ValueError('B and E must reproduce the same operational rule')
   weeks.append({'cutoff':cutoff.isoformat(),'actual':actual,'party':party,'variants':variants})
   for variant,values in variants.items():
    for t,target in enumerate(('bookings','guests')):
     rows.append({'configuration':configuration,'variant':variant,'target':target,'start':cutoff.isoformat(),
                  'actual':sum(actual[t]),'predicted':sum(values[t]),'error':sum(values[t])-sum(actual[t]),
                  'absolute_error':abs(sum(values[t])-sum(actual[t]))})
  for period,subset in (('development',weeks[:24]),('reserved',weeks[24:]),('overall',weeks)):
   for variant in VARIANTS:
    for t,target in enumerate(('bookings','guests')):
     m=prior.summarize([(w['actual'][t],w['variants'][variant][t]) for w in subset],target)
     summaries.append({'configuration':configuration,'variant':variant,'period':period,'target':target,**m})
     if variant=='B' and period!='overall':
      expected=reference['development' if period=='development' else 'holdout'][method]['operational'][target]
      for scale in ('weekly','daily'):
       for field in ('wape','mae','rmse','net_error','absolute_error'):
        if abs(m[scale][field]-expected[scale][field])>1e-8:raise ValueError('Operational reproduction failed')
     changed=sum(abs(a-b)>1e-9 for w in subset for a,b in zip(w['variants']['A'][t],w['variants'][variant][t]))
     frequencies.append({'configuration':configuration,'variant':variant,'period':period,'target':target,
                         'changed_from_raw_days':changed,'days':len(subset)*7})
    counts=[inconsistency(w['variants'][variant]) for w in subset]
    quality.append({'configuration':configuration,'variant':variant,'period':period,
                    **{key:sum(c[key] for c in counts) for key in counts[0]}})
   for old,new,name in (('A','C','round_clamp'),('C','D','booking_minimum'),('D','B','guest_floor'),('B','E','equivalence')):
    for t,target in enumerate(('bookings','guests')):
     old_week=[sum(w['variants'][old][t]) for w in subset];new_week=[sum(w['variants'][new][t]) for w in subset]
     actual_week=[sum(w['actual'][t]) for w in subset]
     old_day=[p for w in subset for p in w['variants'][old][t]];new_day=[p for w in subset for p in w['variants'][new][t]]
     actual_day=[a for w in subset for a in w['actual'][t]]
     delta=lambda a,p,q:sum(abs(y-x)-abs(z-x) for x,y,z in zip(a,q,p))
     contributions.append({'configuration':configuration,'period':period,'adjustment':name,'target':target,
       'weekly_absolute_error_change':delta(actual_week,old_week,new_week),
       'daily_absolute_error_change':delta(actual_day,old_day,new_day),
       'prediction_total_change':sum(new_day)-sum(old_day),
       'changed_days':sum(abs(p-q)>1e-9 for p,q in zip(old_day,new_day))})
 return {'summaries':summaries,'contributions':contributions,'frequencies':frequencies,'inconsistencies':quality,'weekly':rows}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
 args=parser.parse_args()
 if args.output.exists() or args.output.resolve().parent!=HERE:raise ValueError('Use a new isolated output directory')
 args.output.mkdir()
 (args.output/'protocol.json').write_text(json.dumps(PLAN,indent=2))
 before=prior.base.serving_hashes(ROOT)
 result=evaluate(prior.base.read_export(args.input))
 if before!=prior.base.serving_hashes(ROOT):raise ValueError('Serving files changed')
 result['validation']={'operational_reproduction':True,'B_equals_E':True,'serving_hashes_unchanged':True,
                       'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'serving_hashes':before}
 with (args.output/'results.json').open('x') as h:json.dump(result,h,indent=2,allow_nan=False)
 with (args.output/'weekly_errors.csv').open('x',newline='') as h:
  writer=csv.DictWriter(h,fieldnames=list(result['weekly'][0]));writer.writeheader();writer.writerows(result['weekly'])
 for r in result['summaries']:
  if r['variant']=='E':continue
  print('METRICS',json.dumps({k:r[k] for k in ('configuration','variant','period','target')}|{
   'weekly_wape':r['weekly']['wape'],'daily_wape':r['daily']['wape'],'weekly_mae':r['weekly']['mae'],
   'weekly_rmse':r['weekly']['rmse'],'weekly_bias':r['weekly']['net_error']/r['weeks'],
   'low_mae':r['low_demand']['mae'] if r['low_demand'] else None,'zero_forecast':r['zero_week_false_positive_total']}))
 print('VALIDATION',json.dumps(result['validation']))


if __name__=='__main__':main()
