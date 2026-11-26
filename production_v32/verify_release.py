"""Portable build/readiness check; never writes models or prepares a release."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from production_v32 import runtime
if __name__=='__main__':
    runtime.validate_release()
    result=runtime.health()
    if not result['model_ready']:raise SystemExit('V3.2 model is not ready; fallback disabled')
    if result.get('production_approved') is not True:
        raise SystemExit('V3.2 production_approved must be true after authorization')
    print('Frozen v3.2 integrity and readiness passed; production-approved; no training or fallback.')
