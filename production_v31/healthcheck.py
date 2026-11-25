import json, os
from urllib.request import urlopen
if __name__=='__main__':
    with urlopen('http://127.0.0.1:'+str(int(os.getenv('PORT','8001')))+'/health',timeout=5) as response:
        state=json.load(response)
    # HTTP 200 alone is liveness, not readiness. Never mark a broken mount healthy.
    if state.get('model_ready') is not True or state.get('metadata',{}).get('model_version')!='3.1':
        raise SystemExit(1)
