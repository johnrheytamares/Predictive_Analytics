"""Read-only ASGI, loopback HTTP and frozen-release regression checks."""
import asyncio, copy, hashlib, json, os, socket, sys, threading, time, unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import verify_shadow_fastapi as legacy
import shadow_api
import shadow_predict as sp
import api as canonical_api

async def request(app,path,params=None,headers=None):
    messages=[]
    async def receive():return {'type':'http.request','body':b'','more_body':False}
    async def send(message):messages.append(message)
    await app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'GET',
        'scheme':'http','path':path,'raw_path':path.encode(),'query_string':urlencode(params or {}).encode(),
        'headers':[(k.lower().encode(),v.encode()) for k,v in (headers or {}).items()],
        'client':('127.0.0.1',1),'server':('127.0.0.1',8002),'root_path':''},receive,send)
    head=next(m for m in messages if m['type']=='http.response.start')
    body=json.loads(b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body'))
    return head['status'],body,dict(head['headers'])

def get(path,params=None,headers=None,app=None):
    if headers is None:
        key=os.getenv('PREDICTION_API_KEY','').strip()
        headers={'X-API-Key':key} if key else {}
    return asyncio.run(request(app or shadow_api.app,path,params,headers))

def params(days=7,start='2026-11-22',**extra):
    return dict(start_date=start,end_date=str((pd.Timestamp(start)+pd.Timedelta(days=days-1)).date()),**extra)

class HardenedServingTests(unittest.TestCase):
    def test_frozen_metadata_unchanged_and_complete_separate_weekly_scores(self):
        status,health,_=get('/health')
        self.assertEqual(status,200);self.assertTrue(health['model_ready'])
        self.assertEqual(health['metadata'],json.loads((HERE/'artifacts/forecast_models_metadata.json').read_text()))
        for target in ('bookings','guests'):
            block=health['weekly_performance'][target]
            for mode in ('raw','operational_floor_round','variant_a'):
                self.assertTrue({'wape','mae','rmse','bias','n','actual_total'}<=block[mode].keys())
                self.assertEqual(block[mode]['n'],28)
        self.assertAlmostEqual(health['weekly_performance']['guests']['operational_floor_round']['wape'],38.78914405010438)
        self.assertFalse(health['weekly_performance']['long_horizon_accuracy_evaluated'])

    def test_no_runtime_tuning_and_horizon_prefix_stability(self):
        rows=[]
        with patch.object(sp.intermittent,'tune',side_effect=AssertionError('Serving must not optimize')):
            for days in (7,30,90):
                status,body,_=get('/forecast',params(days))
                self.assertEqual(status,200)
                self.assertEqual(body['model']['two_stage_parameters'],dict(prob_days=14,size_days=84,shrink=0.))
                self.assertFalse(body['model']['runtime_optimization'])
                rows.append(body['forecast'])
        self.assertEqual(rows[0],rows[1][:7]);self.assertEqual(rows[1],rows[2][:30])

    def test_all_28_frozen_origins_replay_without_optimization(self):
        frame=pd.read_csv(HERE/'package/postprocess_evidence/daily_predictions.csv')
        frame=frame[(frame.model=='two_stage')&(frame.ops=='floor_round')]
        with patch.object(sp.intermittent,'tune',side_effect=AssertionError('No optimization')):
            for origin,g in frame.groupby('origin'):
                status,body,_=get('/forecast',params(start=origin,allow_historical='true'))
                self.assertEqual(status,200,origin)
                np.testing.assert_array_equal([r['predicted_bookings'] for r in body['forecast']],g.pred_bookings)
                np.testing.assert_array_equal([r['predicted_total_guests'] for r in body['forecast']],g.pred_guests)

    def test_authentication_contract(self):
        with patch.dict(os.environ,{'PREDICTION_API_KEY':'local-test-fixture'}):
            self.assertEqual(get('/forecast',params(),headers={})[0],401)
            self.assertEqual(get('/forecast',params(),headers={'X-API-Key':'wrong'})[0],401)
            self.assertEqual(get('/forecast',params(),headers={'X-API-Key':'local-test-fixture'})[0],200)
            self.assertEqual(get('/health',headers={})[0],200)

    def test_invalid_requests_and_frozen_historical_policy(self):
        for query,code in [({},None),({'start_date':'bad','end_date':'2026-11-28'},'INVALID_DATE'),
                ({'start_date':'2026-11-28','end_date':'2026-11-22'},'INVALID_RANGE'),
                (params(91),'RANGE_TOO_LARGE'),(params(start='2026-11-09'),'RANGE_OVERLAPS_HISTORY'),
                (params(start='2026-11-10',allow_historical='true'),'FROZEN_PARAMETERS_UNAVAILABLE')]:
            status,body,headers=get('/forecast',query)
            self.assertIn(status,(400,422));self.assertIn('detail',body)
            if code:self.assertEqual(headers[b'x-error-code'].decode(),code)

    def test_configuration_mismatch_fails_closed(self):
        bookings,guests=sp.get_bundles()
        bad=dict(bookings,model_type='weekday_baseline')
        with patch.object(sp,'get_bundles',return_value=(bad,guests)):
            status,body,_=get('/forecast',params())
            self.assertEqual(status,422);self.assertEqual(body['code'],'MODEL_CONFIGURATION_MISMATCH')
            self.assertFalse(get('/health')[1]['model_ready'])

    def test_integrity_failure_simulated_without_modifying_frozen_files(self):
        original=Path.read_bytes
        target=(HERE/'artifacts/guests_model.joblib').resolve()
        def corrupted(path):return b'bad-test-bytes' if path.resolve()==target else original(path)
        with patch.object(Path,'read_bytes',corrupted):
            self.assertEqual(get('/forecast',params())[0],422)
            self.assertFalse(get('/health')[1]['model_ready'])
        self.assertTrue(get('/health')[1]['model_ready'])

    def test_floor_round_zero_and_consistency_edges(self):
        b,g=sp.apply_floor_round(np.array([0.,.9,1.2,.1]),np.array([0.,.9,.8,2.9]))
        self.assertEqual(b,[0,0,1,1]);self.assertEqual(g,[0,0,1,2])

    def test_missing_guest_weekday_history_has_no_fallback(self):
        bookings,guests=sp.get_bundles()
        bad=copy.deepcopy(guests)
        bad['baseline_history']=[r for r in bad['baseline_history'] if pd.Timestamp(r['ds']).dayofweek!=0]
        with patch.object(sp,'get_bundles',return_value=(bookings,bad)):
            status,body,_=get('/forecast',params())
            self.assertEqual(status,422);self.assertEqual(body['code'],'ROBUST_RANGE_UNSUPPORTED')
            self.assertIn('unsupported_dates',body);self.assertIn('policy',body)

    def test_canonical_api_outputs_and_parameter_contract_unchanged(self):
        before=get('/forecast',params(),app=canonical_api.app)
        self.assertEqual(before[0],200)
        get('/forecast',params(90))
        after=get('/forecast',params(),app=canonical_api.app)
        for body in (before[1],after[1]):body.pop('generated_at',None)
        self.assertEqual(before,after)
        root=canonical_api.app.openapi()['paths'];shadow=shadow_api.app.openapi()['paths']
        for path in ('/health','/forecast'):
            source={p['name']:p['schema'] for p in root[path]['get'].get('parameters',[])}
            dest={p['name']:p['schema'] for p in shadow[path]['get'].get('parameters',[])}
            for name,schema in source.items():self.assertEqual(schema,dest[name])

    def test_real_loopback_http_startup_and_shutdown(self):
        import uvicorn
        sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
        port=sock.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(shadow_api.app,host='127.0.0.1',port=port,log_level='critical',access_log=False))
        thread=threading.Thread(target=lambda:asyncio.run(server.serve(sockets=[sock])),daemon=True)
        thread.start()
        try:
            deadline=time.monotonic()+10
            while not server.started and thread.is_alive() and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(server.started)
            key=os.getenv('PREDICTION_API_KEY','').strip()
            headers={'X-API-Key':key} if key else {}
            for path in ('/health','/forecast?'+urlencode(params(90))):
                with urlopen(Request(f'http://127.0.0.1:{port}'+path,headers=headers),timeout=10) as response:
                    body=json.load(response)
                    self.assertEqual(response.status,200)
                    if path=='/health':self.assertTrue(body['model_ready'])
                    else:self.assertEqual(len(body['forecast']),90)
        finally:
            server.should_exit=True;thread.join(timeout=10);sock.close()
        self.assertFalse(thread.is_alive())

    def test_artifact_preparation_refuses_existing_frozen_models(self):
        import prepare_shadow_serving
        with self.assertRaises(SystemExit):prepare_shadow_serving.main()
        sp.validate_integrity()


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(legacy.ShadowFastAPITests),
                            unittest.defaultTestLoader.loadTestsFromTestCase(HardenedServingTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    integrity=sp.validate_integrity()
    report=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        protected_files=len(integrity['protected_files']),protected_files_unchanged=True,
        weekly_performance=sp.weekly_performance(),responses=legacy.ShadowFastAPITests.snapshots,
        local_loopback_test_passed=not any('test_real_loopback' in str(t) for t,_ in result.failures+result.errors),
        historical_frozen_origins_replayed=28,no_runtime_optimization=True,
        prospective_validation='not_run',production_replaced=False)
    out=HERE/'integration_validation';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    sys.exit(not result.wasSuccessful())
