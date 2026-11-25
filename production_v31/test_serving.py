"""Focused production adapter tests; no fitting, sync, database or deployment."""
from pathlib import Path
import asyncio, importlib.util, json, os, shutil, socket, subprocess, sys, tempfile, threading, time, unittest
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from production_v31 import api, history_refresh, runtime
from production_v31.runtime import engine

spec=importlib.util.spec_from_file_location('existing_shadow_regression',runtime.RC/'verify_shadow_fastapi.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

async def asgi(app,path,params=None,headers=None,method='GET',body=None):
    payload=b'' if body is None else (body if isinstance(body,(bytes,bytearray)) else json.dumps(body).encode())
    messages=[]
    async def receive():return {'type':'http.request','body':payload,'more_body':False}
    async def send(m):messages.append(m)
    hdrs=[(k.lower().encode(),v.encode()) for k,v in (headers or {}).items()]
    if body is not None and not any(k==b'content-type' for k,_ in hdrs):
        hdrs.append((b'content-type',b'application/json'))
    await app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':method,'scheme':'http',
        'path':path,'raw_path':path.encode(),'query_string':urlencode(params or {}).encode(),
        'headers':hdrs,'client':('127.0.0.1',1),'server':('127.0.0.1',8001),'root_path':''},receive,send)
    head=next(m for m in messages if m['type']=='http.response.start')
    raw=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
    parsed=json.loads(raw) if raw else {}
    return head['status'],parsed,dict(head['headers'])

def get(path,params=None,headers=None):return asyncio.run(asgi(api.app,path,params,headers))
def post(path,body=None,headers=None):return asyncio.run(asgi(api.app,path,headers=headers,method='POST',body=body))
def query(start,days):return dict(start_date=start,end_date=str((pd.Timestamp(start)+pd.Timedelta(days=days-1)).date()))

def frozen_demand_rows():
    bookings,guests=runtime._original_get_bundles(use_cache=False)
    b={row['ds']:float(row['y']) for row in bookings['baseline_history']}
    g={row['ds']:float(row['y']) for row in guests['baseline_history']}
    return [{'demand_date':ds,'bookings':b[ds],'guests':g[ds]} for ds in sorted(b)]

def extended_demand_rows(extra_days=3,bookings_value=2.0,guests_value=4.0):
    rows=frozen_demand_rows()
    last=pd.Timestamp(rows[-1]['demand_date'])
    for offset in range(1,extra_days+1):
        rows.append({
            'demand_date':str((last+pd.Timedelta(days=offset)).date()),
            'bookings':float(bookings_value),
            'guests':float(guests_value),
        })
    return rows

class ProductionServingTests(unittest.TestCase):
    def test_readiness_identity_metrics_and_no_claim_of_approval(self):
        status,body,_=get('/health')
        self.assertEqual(status,200);self.assertEqual(body['service'],'predictive-analytics')
        self.assertTrue(body['model_ready']);self.assertEqual(body['metadata']['model_version'],'3.1')
        self.assertFalse(body['metadata']['shadow_serving_only']);self.assertTrue(body['production_approved'])
        self.assertFalse(body['lifecycle']['prospective_accuracy_validated'])
        for t in ('bookings','guests'):
            for key in ('wape','mae','rmse','bias'):
                self.assertEqual(body['metadata']['targets'][t]['selected_metrics'][key],
                                 body['weekly_performance'][t]['operational_floor_round'][key])
        self.assertFalse(body['lifecycle']['prospective_accuracy_validated'])
        self.assertFalse(body['lifecycle']['dataset_sync_refreshes_served_history'])
        self.assertTrue(body['lifecycle']['history_refresh_supported'])

    def test_current_and_far_future_7_30_90_day_horizons_without_tuning(self):
        starts=[str(pd.Timestamp.now(tz='Asia/Manila').date()),'2027-06-01']
        with patch.object(engine.intermittent,'tune',side_effect=AssertionError('No optimization')):
            for start in starts:
                for days in (7,30,90):
                    status,body,_=get('/forecast',query(start,days))
                    self.assertEqual(status,200);self.assertEqual(len(body['forecast']),days)
                    self.assertEqual(body['model_version'],'3.1')
                    self.assertEqual(body['forecast_capabilities']['cutoff'],start)
                    self.assertFalse(body['model']['runtime_optimization'])
                    for r in body['forecast']:
                        b,g=r['predicted_bookings'],r['predicted_total_guests']
                        self.assertIs(type(b),int);self.assertIs(type(g),int)
                        self.assertGreaterEqual(b,0);self.assertGreaterEqual(g,b)
                        if g>0:self.assertGreaterEqual(b,1)

    def test_frozen_forward_predictions_are_identical_to_previous_api(self):
        expected=json.loads((runtime.RC/'verification_shadow_fastapi.json').read_text())['responses']
        for days in (7,30,90):
            status,body,_=get('/forecast',dict(**query('2026-11-22',days),compare_variant_a='true'))
            self.assertEqual(status,200)
            self.assertEqual(sum(r['predicted_bookings'] for r in body['forecast']),expected[f'range_{days}']['bookings'])
            self.assertEqual(sum(r['predicted_total_guests'] for r in body['forecast']),expected[f'range_{days}']['guests'])
            self.assertEqual(body['postprocess_comparison'],expected[f'variant_a_comparison_{days}d'])

    def test_api_schema_and_error_compatibility(self):
        import api as canonical
        before=canonical.app.openapi()['paths'];after=api.app.openapi()['paths']
        for name in ('start_date','end_date','allow_historical','X-API-Key'):
            a=next(p for p in before['/forecast']['get']['parameters'] if p['name']==name)
            b=next(p for p in after['/forecast']['get']['parameters'] if p['name']==name)
            self.assertEqual(a,b)
        for p,code in [({'start_date':'bad','end_date':'2026-11-28'},'INVALID_DATE'),
            (query('2026-11-22',91),'RANGE_TOO_LARGE'),
            (dict(start_date='2026-11-28',end_date='2026-11-22'),'INVALID_RANGE')]:
            status,body,headers=get('/forecast',p)
            self.assertEqual(status,400);self.assertIn('detail',body)
            self.assertEqual(headers[b'x-error-code'].decode(),code)
        status,body,_=get('/forecast',query('2026-11-22',7))
        self.assertTrue(legacy.PRODUCTION_FORECAST_KEYS<=body.keys())
        self.assertTrue(legacy.PRODUCTION_ROW_KEYS<=body['forecast'][0].keys())

    def test_shared_key_gate_and_public_health(self):
        with patch.dict(os.environ,{'PREDICTION_API_KEY':'test-fixture-only'}):
            self.assertEqual(get('/health')[0],200)
            self.assertEqual(get('/forecast',query('2026-11-22',7))[0],401)
            self.assertEqual(get('/forecast',query('2026-11-22',7),{'X-API-Key':'wrong'})[0],401)
            self.assertEqual(get('/forecast',query('2026-11-22',7),{'X-API-Key':'test-fixture-only'})[0],200)

    def test_no_v3_fallback_for_wrong_artifact_directory(self):
        with patch.object(engine,'ARTIFACTS',ROOT):
            self.assertFalse(get('/health')[1]['model_ready'])
            status,body,_=get('/forecast',query('2026-11-22',7))
            self.assertEqual(status,422);self.assertEqual(body['code'],'MODEL_CONFIGURATION_MISMATCH')

    def test_missing_frozen_artifact_fails_closed_without_disk_mutation(self):
        original=Path.is_file
        target=runtime.RC/'artifacts/bookings_model.joblib'
        with patch.object(Path,'is_file',lambda p:False if p==target else original(p)):
            self.assertFalse(get('/health')[1]['model_ready'])
            self.assertEqual(get('/forecast',query('2026-11-22',7))[0],422)
        self.assertTrue(get('/health')[1]['model_ready'])

    def test_relocatable_image_file_layout_and_cors_environment(self):
        # Copy the exact manifest dependency set into a clean temporary /app-like
        # tree, without any other repository experiments or canonical trainer.
        manifest=json.loads(runtime.MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)
            # Reproduce actual COPY destinations rather than assuming the image
            # contains every manifest file. This also catches missing COPY inputs.
            for line in (ROOT/'Dockerfile.v31').read_text().splitlines():
                if not line.startswith('COPY '):continue
                parts=line.split()[1:];targetdir=destination/parts[-1]
                targetdir.mkdir(parents=True,exist_ok=True)
                for name in parts[:-1]:
                    source=ROOT/name
                    if source.is_dir():shutil.copytree(source,targetdir,dirs_exist_ok=True)
                    else:shutil.copy2(source,targetdir/source.name)
            self.assertTrue((ROOT/'Dockerfile.v31.dockerignore').exists())
            self.assertIn('!experiments/reserVision_v3_1_two_stage_rc_20261124/**',
                          (ROOT/'Dockerfile.v31.dockerignore').read_text())
            self.assertTrue((destination/'experiments/reserVision_v3_1_two_stage_rc_20261124/package/postprocess_evidence/daily_predictions.csv').is_file())
            code="""import asyncio,json
from production_v31 import runtime,api
runtime.validate_release()
assert runtime.health()['model_ready']
assert not (runtime.ROOT/'train_real_model.py').exists()
messages=[]
async def receive():return {'type':'http.request','body':b'','more_body':False}
async def send(m):messages.append(m)
asyncio.run(api.app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'GET','scheme':'http','path':'/health','raw_path':b'/health','query_string':b'','headers':[(b'origin',b'https://ui.example.test')],'client':('127.0.0.1',1),'server':('127.0.0.1',8001),'root_path':''},receive,send))
headers=dict(messages[0]['headers'])
assert headers[b'access-control-allow-origin']==b'https://ui.example.test'
print('Relocation and CORS passed')
"""
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CORS_ALLOW_ORIGINS':'https://ui.example.test'}
            env.pop('PREDICTION_ARTIFACT_DIR',None)
            result=subprocess.run([sys.executable,'-B','-c',code],cwd=destination,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_real_http_lifecycle_and_readiness_healthcheck(self):
        import uvicorn
        sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
        port=sock.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(api.app,log_level='critical',access_log=False))
        thread=threading.Thread(target=lambda:asyncio.run(server.serve(sockets=[sock])),daemon=True);thread.start()
        try:
            deadline=time.monotonic()+10
            while not server.started and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(server.started)
            with urlopen(f'http://127.0.0.1:{port}/health',timeout=10) as r:self.assertTrue(json.load(r)['model_ready'])
            key=os.getenv('PREDICTION_API_KEY','').strip()
            with urlopen(Request(f'http://127.0.0.1:{port}/forecast?'+urlencode(query('2027-06-01',90)),
                         headers={'X-API-Key':key} if key else {}),timeout=10) as r:
                self.assertEqual(len(json.load(r)['forecast']),90)
            result=subprocess.run([sys.executable,'-B','-m','production_v31.healthcheck'],cwd=ROOT,
                env={**os.environ,'PORT':str(port)},capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
        finally:
            server.should_exit=True;thread.join(timeout=10);sock.close()
        self.assertFalse(thread.is_alive())

    def test_existing_sync_contract_is_not_misrepresented_as_history_refresh(self):
        # Source verification only: never call sync (it writes the DB).
        source=(ROOT.parent/'CAPSTONE_BACKEND/reservision-backend/services/predictionTrainingService.js').read_text()
        self.assertIn('skipped_local_training: true',source)
        self.assertIn('syncBookingDemandDaily({ includeConfirmedFallback })',source)
        life=runtime.health()['lifecycle']
        self.assertFalse(life['dataset_sync_refreshes_served_history'])
        self.assertTrue(life['history_refresh_supported'])
        self.assertFalse(life['model_fitted'])
        self.assertFalse(life['automatic_retraining'])
        # Node Sync Training Data still does not POST /history/refresh.
        self.assertIn('lifecycle?.dataset_sync_refreshes_served_history === true',source)
        self.assertNotIn('/history/refresh',source)

class HistoryRefreshTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir=tempfile.TemporaryDirectory()
        self.history_dir=Path(self._tmpdir.name)/'runtime_history'
        self._env=patch.dict(os.environ,{'PREDICTION_HISTORY_DIR':str(self.history_dir)})
        self._env.start()
        history_refresh.clear_overlay_cache()
        engine.clear_cache()
        if self.history_dir.exists():
            shutil.rmtree(self.history_dir)

    def tearDown(self):
        history_refresh.clear_overlay_cache()
        engine.clear_cache()
        self._env.stop()
        self._tmpdir.cleanup()

    def test_successful_refresh_then_forecast_preserves_champions_and_metrics(self):
        before=get('/health')[1]
        frozen_metrics=before['weekly_performance']
        rows=extended_demand_rows(extra_days=5)
        status,body,_=post('/history/refresh',{'rows':rows,'source':'booking_demand_daily'})
        self.assertEqual(status,200,body)
        self.assertTrue(body['success']);self.assertTrue(body['history_refreshed'])
        self.assertFalse(body['model_retrained']);self.assertFalse(body['model_fitted'])
        self.assertFalse(body['database_synchronized']);self.assertFalse(body['hyperparameters_changed'])
        self.assertTrue(body['evaluation_metrics_unchanged'])
        self.assertEqual(body['evaluation_scope'],'retrospective_rolling_28_origins')
        self.assertEqual(body['history_as_of'],rows[-1]['demand_date'])
        self.assertEqual(body['training_days'],len(rows))
        health=get('/health')[1]
        self.assertTrue(health['model_ready'])
        self.assertTrue(health['lifecycle']['served_history_refreshed'])
        self.assertTrue(health['lifecycle']['history_refresh_supported'])
        self.assertFalse(health['lifecycle']['dataset_sync_refreshes_served_history'])
        self.assertFalse(health['lifecycle']['evaluation_metrics_refreshed'])
        self.assertFalse(health['lifecycle']['prospective_accuracy_validated'])
        self.assertEqual(health['weekly_performance'],frozen_metrics)
        self.assertEqual(health['metadata']['targets']['bookings']['selected_model'],'two_stage')
        self.assertEqual(health['metadata']['targets']['guests']['selected_model'],'robust_median_guests')
        self.assertEqual(health['metadata']['postprocess']['selected_mode'],'floor_round')
        # Subsequent forecast uses refreshed cutoff without claiming new validation.
        start=str((pd.Timestamp(rows[-1]['demand_date'])+pd.Timedelta(days=1)).date())
        status,forecast,_=get('/forecast',query(start,7))
        self.assertEqual(status,200,forecast)
        self.assertEqual(forecast['model_version'],'3.1')
        self.assertTrue(forecast['model']['served_history_refreshed'])
        self.assertTrue(forecast['model']['frozen_evaluation_metrics'])
        self.assertEqual(forecast['model']['targets']['bookings']['selected_model'],'two_stage')
        self.assertEqual(forecast['model']['targets']['guests']['selected_model'],'robust_median_guests')
        self.assertEqual(len(forecast['forecast']),7)
        for row in forecast['forecast']:
            self.assertIs(type(row['predicted_bookings']),int)
            self.assertIs(type(row['predicted_total_guests']),int)
            self.assertGreaterEqual(row['predicted_total_guests'],row['predicted_bookings'])

    def test_invalid_history_is_rejected_without_activation(self):
        rows=extended_demand_rows(extra_days=2)
        rows[10]['bookings']=999  # Alters a frozen prefix day.
        status,body,_=post('/history/refresh',{'rows':rows})
        self.assertEqual(status,422,body)
        self.assertEqual(body['code'],'INVALID_HISTORY')
        self.assertFalse((self.history_dir/'active_history.json').exists())
        health=get('/health')[1]
        self.assertTrue(health['model_ready'])
        self.assertFalse(health['lifecycle']['served_history_refreshed'])
        self.assertEqual(health['metadata']['date_to'],'2026-11-21')

    def test_failed_refresh_restores_previous_working_snapshot(self):
        first=extended_demand_rows(extra_days=3)
        status,body,_=post('/history/refresh',{'rows':first})
        self.assertEqual(status,200,body)
        active_before=json.loads((self.history_dir/'active_history.json').read_text(encoding='utf-8'))
        second=extended_demand_rows(extra_days=6)
        with patch.object(runtime,'_validate_overlay_for_serving',
                          side_effect=history_refresh.HistoryRefreshError(
                              'Simulated apply failure.',code='HISTORY_REFRESH_FAILED')):
            status,body,_=post('/history/refresh',{'rows':second})
        self.assertEqual(status,422,body)
        self.assertEqual(body['code'],'HISTORY_REFRESH_FAILED')
        active_after=json.loads((self.history_dir/'active_history.json').read_text(encoding='utf-8'))
        self.assertEqual(active_after['date_to'],active_before['date_to'])
        self.assertEqual(active_after['training_days'],active_before['training_days'])
        health=get('/health')[1]
        self.assertTrue(health['model_ready'])
        self.assertEqual(health['metadata']['date_to'],first[-1]['demand_date'])
        status,forecast,_=get('/forecast',query(str((pd.Timestamp(first[-1]['demand_date'])+pd.Timedelta(days=1)).date()),7))
        self.assertEqual(status,200)
        self.assertEqual(forecast['model_version'],'3.1')

    def test_history_refresh_requires_api_key_when_configured(self):
        with patch.dict(os.environ,{'PREDICTION_API_KEY':'test-fixture-only','PREDICTION_HISTORY_DIR':str(self.history_dir)}):
            self.assertEqual(post('/history/refresh',{'rows':extended_demand_rows()})[0],401)
            self.assertEqual(post('/history/refresh',{'rows':extended_demand_rows()},{'X-API-Key':'wrong'})[0],401)
            status,body,_=post('/history/refresh',{'rows':extended_demand_rows()},{'X-API-Key':'test-fixture-only'})
            self.assertEqual(status,200,body)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ProductionServingTests),
                            unittest.defaultTestLoader.loadTestsFromTestCase(HistoryRefreshTests),
                            unittest.defaultTestLoader.loadTestsFromTestCase(legacy.ShadowFastAPITests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    runtime.validate_release()
    output=ROOT/'production_v31/validation';output.mkdir(exist_ok=True)
    (output/'results.json').write_text(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),passed=result.wasSuccessful(),production_and_frozen_hashes_unchanged=True,
        performance=engine.weekly_performance(),docker_image_built=False,docker_daemon_available=False,
        prospective_accuracy_validated=False,history_refresh_supported=True),indent=2),encoding='utf-8')
    sys.exit(not result.wasSuccessful())
