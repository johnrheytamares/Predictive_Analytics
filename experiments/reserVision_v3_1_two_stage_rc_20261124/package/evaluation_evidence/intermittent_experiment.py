"""Isolated bookings experiment; no database access or serving artifact writes."""
from pathlib import Path
import sys, json, hashlib, itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_forecasts import load_history, metrics, weekday_forecast

OUT = Path(__file__).resolve().parent / 'results'
SOURCE = ROOT / 'experiments/multimodel_dataset_20261123/prepared/data/daily_demand.csv'
GRIDS = {
    'optimized_weekday': [dict(weeks=w, decay=d, shrink=s) for w,d,s in itertools.product((2,4,8,12),(1.,.7),(0.,.5))],
    'two_stage': [dict(prob_days=p, size_days=z, shrink=s) for p,z,s in itertools.product((14,28,56),(28,84,168),(0.,.5,1.))],
    'tsb': [dict(alpha=a,beta=b) for a,b in itertools.product((.05,.1,.3,.5),(.02,.05,.1,.3)) if b<=a],
}

def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir())
            if p.is_file() and (p.suffix in ('.py','.joblib') or p.name.endswith('metadata.json'))}

def predict(h, dates, method, cfg=None):
    cfg = cfg or {}
    assert len(h) and h.index.max() < min(dates)
    if method == 'current_weekday':
        return weekday_forecast(h, dates)
    if method == 'zero_reference':
        return np.zeros(len(dates))
    if method == 'tsb':
        # Initialize from the first observation only. Update p on every day,
        # including zeros; size only on observed positive demand. No future
        # probability decay: unobserved future days are not known zeros.
        p, size = float(h.iloc[0]>0), float(h.iloc[0])
        seen = size>0
        for y in h.iloc[1:]:
            p += cfg['beta'] * (float(y>0)-p)
            if y>0:
                size = size + cfg['alpha']*(y-size) if seen else float(y)
                seen = True
        return np.full(len(dates),p*size)
    result=[]
    for day in dates:
        if method == 'optimized_weekday':
            recent=h.tail(cfg['weeks']*7)
            same=recent[recent.index.dayofweek==day.dayofweek]
            weights=cfg['decay']**np.arange(len(same)-1,-1,-1)
            v=float(np.average(same,weights=weights))
            result.append((1-cfg['shrink'])*v+cfg['shrink']*recent.mean())
        elif method == 'two_stage':
            # Hurdle: probability of any demand times conditional positive size.
            # Zero observations remain in the probability denominator.
            ph=h.tail(cfg['prob_days']); zh=h.tail(cfg['size_days'])
            same=ph[ph.index.dayofweek==day.dayofweek]
            prob=(1-cfg['shrink'])*(same>0).mean()+cfg['shrink']*(ph>0).mean()
            positive=zh[zh>0]
            size=float(positive.mean()) if len(positive) else 0.
            result.append(prob*size)
        else:
            raise ValueError(method)
    return np.asarray(result,float)

def tune(h, method, horizon):
    # Four non-overlapping inner origins ending strictly before outer cutoff.
    # Same forecast horizon as the outer evaluation; no outer labels accessible.
    starts=[len(h)-horizon*k for k in range(4,0,-1)]
    assert min(starts)>=120
    candidates=[]
    for cfg in GRIDS[method]:
        a=[]; p=[]
        for start in starts:
            test=h.iloc[start:start+horizon]
            a.extend(test.values)
            p.extend(predict(h.iloc[:start],test.index,method,cfg))
        candidates.append((metrics(a,p)['mae'],json.dumps(cfg,sort_keys=True),cfg))
    return min(candidates,key=lambda x:(x[0],x[1]))[2]

def summary(frame):
    rows=[]
    for (protocol,method),g in frame.groupby(['protocol','method']):
        daily=metrics(g.actual,g.predicted)
        weekly=g.groupby(['origin','week'])[['actual','predicted']].sum()
        rows.append(dict(protocol=protocol,method=method,daily=daily,
                         weekly=metrics(weekly.actual,weekly.predicted)))
    return rows

def profile(h):
    positive=h[h>0]; runs=[]; run=0
    for y in h:
        if y==0: run+=1
        elif run: runs.append(run);run=0
    if run:runs.append(run)
    return dict(days=len(h),bookings=float(h.sum()),zero_days=int((h==0).sum()),
                zero_pct=float(100*(h==0).mean()),average_inter_demand_interval=len(h)/len(positive),
                positive_size_cv_squared=float((positive.std(ddof=0)/positive.mean())**2),
                longest_zero_run=max(runs,default=0),mean_positive_size=float(positive.mean()))

def checks(h):
    dates=pd.date_range(h.index.max()+pd.Timedelta(days=1),periods=7)
    for method,grid in GRIDS.items():
        cfg=grid[0]
        assert np.isfinite(predict(h,dates,method,cfg)).all()
        assert (predict(h*0,dates,method,cfg)==0).all()
        assert np.allclose(predict(h*0+2,dates,method,cfg),2)
    # TSB probability must decline following observed zeros.
    cfg=dict(alpha=.1,beta=.1)
    a=predict(h*0+2,dates,'tsb',cfg)[0]
    hz=h*0+2; hz.iloc[-14:]=0
    assert predict(hz,dates,'tsb',cfg)[0]<a

def main():
    if OUT.exists():raise ValueError('Results directory exists; refusing overwrite')
    before=hashes(); data=load_history(SOURCE)
    h=data.set_index('demand_date').bookings.astype(float)
    assert len(h)==356 and h.sum()==519
    checks(h)
    records=[]; selections=[]
    # Retain the existing 32 weekly origins. First four are warm-up because
    # nested tuning requires >=120 history days at every inner origin.
    origins=pd.date_range('2026-05-04','2026-11-09',freq='7D')
    protocols=[('rolling_7day',origins,7),
               ('original_holdout_28day',[h.index[-28]],28)]
    for protocol,starts,horizon in protocols:
        for origin in starts:
            train=h[h.index<origin]; test=h.loc[origin:origin+pd.Timedelta(days=horizon-1)]
            assert len(test)==horizon
            for method in ['current_weekday',*GRIDS,'zero_reference']:
                cfg=tune(train,method,horizon) if method in GRIDS else {}
                pred=predict(train,test.index,method,cfg)
                selections.append(dict(protocol=protocol,origin=str(origin.date()),method=method,parameters=cfg))
                for i,(day,y,p) in enumerate(zip(test.index,test.values,pred)):
                    records.append(dict(protocol=protocol,origin=str(origin.date()),date=str(day.date()),
                                        week=i//7,method=method,actual=y,predicted=p))
    frame=pd.DataFrame(records); results=summary(frame)
    original=next(r for r in results if r['protocol']=='original_holdout_28day' and r['method']=='current_weekday')
    assert abs(original['daily']['wape']-203.125)<1e-8
    # Explicit error decomposition explains denominator, zero-day errors, bias.
    g=frame[(frame.protocol=='original_holdout_28day')&(frame.method=='current_weekday')]
    cause=dict(actual_total=float(g.actual.sum()),prediction_total=float(g.predicted.sum()),
               absolute_error=float(abs(g.predicted-g.actual).sum()),
               zero_day_absolute_error=float(g.loc[g.actual==0,'predicted'].sum()),
               positive_day_absolute_error=float(abs(g.loc[g.actual>0,'predicted']-g.loc[g.actual>0,'actual']).sum()))
    after=hashes();assert before==after
    report=dict(dataset_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source=str(SOURCE),provenance='Existing artifact-reconstructed historical export; totals match recorded snapshot; source register not independently audited',
        protocol='Fixed origin per window, expanding history; nested past-only tuning minimizes pooled daily MAE (equivalent to WAPE with shared denominator); raw expected counts',
        limitations=['Previously inspected retrospective periods, not untouched prospective validation',
                     '28-day holdout overlaps rolling evaluation; never pool protocols',
                     'Weekly scores sum seven forecast days within each origin; signed bias is predicted minus actual',
                     'First four prior weekly origins omitted for all methods to guarantee 120-day inner training minimum',
                     'No gaps filled, no zero filtering, no rounding or operational adjustments'],
        profiles=dict(full=profile(h),holdout=profile(h.iloc[-28:]),earlier=profile(h.iloc[:-28])),
        wape_cause=cause,results=results,production_hashes_unchanged=before==after,production_hashes=before,
        selections=selections)
    OUT.mkdir()
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    frame.to_csv(OUT/'daily_predictions.csv',index=False)
    flat=[dict(protocol=r['protocol'],method=r['method'],**{f'{scale}_{k}':v for scale in ('daily','weekly') for k,v in r[scale].items()}) for r in results]
    pd.DataFrame(flat).to_csv(OUT/'comparison.csv',index=False)
    print(json.dumps({k:report[k] for k in ('profiles','wape_cause','results','production_hashes_unchanged')},indent=2))

if __name__=='__main__':main()
