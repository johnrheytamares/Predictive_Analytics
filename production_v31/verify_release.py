"""Portable build/readiness check; never writes models or prepares a release."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from production_v31 import runtime
if __name__=='__main__':
    runtime.validate_release()
    result=runtime.health()
    if not result['model_ready']:raise SystemExit('V3.1 model is not ready; fallback disabled')
    print('Frozen v3.1 integrity and readiness passed; no training or fallback.')
