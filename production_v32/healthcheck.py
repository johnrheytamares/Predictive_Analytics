import json, os
from urllib.request import urlopen
if __name__=='__main__':
    with urlopen('http://127.0.0.1:'+str(int(os.getenv('PORT','8002')))+'/health',timeout=5) as response:
        state=json.load(response)
    # HTTP 200 alone is liveness, not readiness. Never mark a broken mount healthy.
    meta=state.get('metadata') or {}
    if (state.get('model_ready') is not True
            or meta.get('model_version')!='3.2'
            or meta.get('production_approved') is not True
            or state.get('production_approved') is not True):
        raise SystemExit(1)
