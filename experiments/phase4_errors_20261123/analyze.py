"""Phase 4: saved raw predictions only; standard library, no fitting or serving imports."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / 'multimodel_compare_20261123' / 'results'

def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def metrics(rows):
    n = len(rows)
    actual = sum(r['actual'] for r in rows)
    error = [r['predicted']-r['actual'] for r in rows]
    absolute = sum(abs(e) for e in error)
    return dict(n=n, actual=actual, predicted=sum(r['predicted'] for r in rows),
                absolute_error=absolute, wape=100*absolute/actual if actual else None,
                mae=absolute/n if n else None, rmse=math.sqrt(sum(e*e for e in error)/n) if n else None,
                bias=sum(error)/n if n else None, over=sum(e>1e-9 for e in error),
                under=sum(e < -1e-9 for e in error), equal=sum(abs(e)<=1e-9 for e in error),
                over_amount=sum(max(0,e) for e in error), under_amount=sum(max(0,-e) for e in error))

def key(r):
    return r['target'], r['cutoff'], r['date']

def validate(rows):
    reference = {key(r):r for r in rows if r['method']=='weekday_baseline'}
    assert len(reference)==448, 'Baseline must have 224 days per target'
    groups = defaultdict(list)
    seen=set()
    for r in rows:
        identity=(r['method'],*key(r))
        assert identity not in seen, 'Duplicate prediction'
        seen.add(identity)
        offset=(date.fromisoformat(r['date'])-date.fromisoformat(r['cutoff'])).days
        assert offset==int(r['horizon_offset']) and 0<=offset<7, 'Horizon mismatch'
        index=(date.fromisoformat(r['cutoff'])-date(2026,4,6)).days
        assert index%7==0 and 0<=index<224, 'Cutoff mismatch'
        assert r['period']==('development' if index<168 else 'reserved'), 'Period mismatch'
        assert math.isfinite(r['predicted']) and math.isfinite(r['actual']) and r['actual']>=0
        assert abs(r['actual']-reference[key(r)]['actual'])<1e-9, 'Actual demand mismatch'
        assert abs((r['predicted']-r['actual'])-float(r['error']))<1e-8, 'Saved error mismatch'
        groups[(r['method'],r['target'],r['cutoff'])].append(offset)
    assert all(sorted(v)==list(range(7)) for v in groups.values()), 'Incomplete horizon'
    for method,target,period in {(r['method'],r['target'],r['period']) for r in rows}:
        keys={key(r) for r in rows if (r['method'],r['target'],r['period'])==(method,target,period)}
        expected={key(r) for r in reference.values() if r['target']==target and r['period']==period}
        assert keys==expected, 'Missing forecast dates or cutoffs within saved period'
    return reference

def weekly(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['method'],r['target'],r['period'],r['cutoff'])].append(r)
    out=[]
    for (method,target,period,cutoff),values in groups.items():
        a=sum(r['actual'] for r in values); p=sum(r['predicted'] for r in values)
        daily_ae=sum(abs(r['predicted']-r['actual']) for r in values)
        out.append(dict(method=method,target=target,period=period,cutoff=cutoff,actual=a,predicted=p,
                        error=p-a,absolute_error=abs(p-a),daily_absolute_error=daily_ae,
                        within_week_cancellation=daily_ae-abs(p-a)))
    return out

def write_csv(path, rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def digest(paths):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def main():
    out=HERE/'results'
    if out.exists(): raise ValueError('Refusing to overwrite existing output')
    inputs=[SOURCE/'daily_predictions.csv',SOURCE/'weekly_predictions.csv',SOURCE/'results.json',
            SOURCE/'arima/daily_predictions.csv',SOURCE/'arima/weekly_predictions.csv',SOURCE/'arima/results.json']
    protected=list(ROOT.glob('*.py'))+list(ROOT.glob('*.joblib'))+list(ROOT.glob('*metadata.json'))
    before=digest(inputs+protected)
    rows=read(inputs[0])+read(inputs[3])
    for r in rows:
        r['actual']=float(r['actual']);r['predicted']=float(r['predicted'])
    reference=validate(rows)
    # Family-specific Ridge choice is descriptive, selected using development only.
    ridge={}
    for target in ('bookings','guests'):
        candidates=sorted({r['method'] for r in rows if r['method'].startswith('ridge_alpha_')})
        ridge[target]=min(candidates,key=lambda m:tuple(metrics(weekly([r for r in rows if r['method']==m and r['target']==target and r['period']=='development']))[k] for k in ('wape','mae','rmse')))
    arima=json.loads(inputs[5].read_text())['selection']
    print('ARIMA_SELECTION',json.dumps(arima))
    # Frozen Phase 3 target-specific ARIMA choices.
    arima_choice={'bookings':'arima_211','guests':'arima_111'}
    assert all(arima[t]['method']==m for t,m in arima_choice.items()), 'Phase 3 selection changed'
    selected=[r for r in rows if r['method'] in ('weekday_baseline','random_forest') or
              r['method']==ridge[r['target']] or r['method']==arima_choice[r['target']]]
    weeks=weekly(selected)
    # Independently reconcile saved weekly totals, without re-running predictors.
    saved=read(inputs[1])+read(inputs[4]); saved_map={(r['method'],r['target'],r['cutoff']):r for r in saved}
    for w in weeks:
        s=saved_map[(w['method'],w['target'],w['cutoff'])]
        for field in ('actual','predicted'):
            assert abs(float(s[field])-w[field])<1e-7, 'Weekly aggregation mismatch'
    summary=[]; strata=[]; concentration=[]; top=[]
    for method,target,period in sorted({(r['method'],r['target'],r['period']) for r in selected}):
        days=[r for r in selected if (r['method'],r['target'],r['period'])==(method,target,period)]
        ws=[r for r in weeks if (r['method'],r['target'],r['period'])==(method,target,period)]
        ident=dict(method=method,target=target,period=period)
        for scale,values in [('daily',days),('weekly',ws)]: summary.append({**ident,'scale':scale,**metrics(values)})
        masks={'zero':lambda r:r['actual']==0,
               'low_positive':lambda r:0<r['actual']<=(1 if target=='bookings' else 10),
               'middle':lambda r:(1 if target=='bookings' else 10)<r['actual']<(3 if target=='bookings' else 30),
               'high':lambda r:r['actual']>=(3 if target=='bookings' else 30),
               'weekday':lambda r:date.fromisoformat(r['date']).weekday()<5,
               'weekend':lambda r:date.fromisoformat(r['date']).weekday()>=5}
        for label,mask in masks.items(): strata.append({**ident,'stratum':label,**metrics([r for r in days if mask(r)])})
        ordered=sorted(ws,key=lambda r:r['absolute_error'],reverse=True)
        total=sum(r['absolute_error'] for r in ws)
        for rank,w in enumerate(ordered[:10],1): top.append({**w,'rank':rank})
        concentration.append({**ident,'top3_weekly_error_share_pct':100*sum(r['absolute_error'] for r in ordered[:3])/total if total else None,
                              'weekly_absolute_error':total,'daily_absolute_error':sum(w['daily_absolute_error'] for w in ws),
                              'within_week_cancellation':sum(w['within_week_cancellation'] for w in ws),
                              'low_week_error_share_pct':100*sum(w['absolute_error'] for w in ws if w['actual']<=(5 if target=='bookings' else 35))/total if total else None,
                              'high_week_error_share_pct':100*sum(w['absolute_error'] for w in ws if w['actual']>=(20 if target=='bookings' else 150))/total if total else None})
    assert before==digest(inputs+protected), 'Protected files changed'
    out.mkdir()
    protocol=dict(phase=4,raw_only=True,selection=dict(ridge=ridge,arima=arima_choice),
      limitations=['Ridge has no saved reserved predictions; no reserved Ridge comparison possible',
                    'Reserved weeks previously examined, not independent validation',
                    'Outcome strata are descriptive, not inputs or tuned selection criteria',
                    'Alignment checks do not independently prove original model training was leakage-free'],
      day_strata='zero=0; low-positive bookings1 / guests1-10; high bookings>=3 / guests>=30; middle remainder',
      week_strata='low bookings<=5 / guests<=35; high bookings>=20 / guests>=150; zeros retained',
      validation=dict(input_rows=len(rows),selected_rows=len(selected),baseline_keys=len(reference),alignment=True,
                      saved_weekly_totals_match=True,protected_hashes_unchanged=True),hashes=before)
    (out/'validation.json').write_text(json.dumps(protocol,indent=2))
    for name,table in [('metrics',summary),('daily_strata',strata),('weekly_errors',weeks),('top10_weeks',top),('concentration',concentration)]:
        write_csv(out/(name+'.csv'),table)
    (out/'analysis.json').write_text(json.dumps(dict(metrics=summary,strata=strata,concentration=concentration),indent=2,allow_nan=False))
    for r in summary:
        print('METRIC',json.dumps(r))
    for r in concentration: print('CONCENTRATION',json.dumps(r))

if __name__=='__main__': main()
