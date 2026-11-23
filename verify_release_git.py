"""Read-only verification of the existing RC; never prepares models or writes reports."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RC = "experiments/robust_release_candidate_20261124"
BASELINE = "51730a2179da9bbc0bf8c492f7b3c6a2a5201d2c"

def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(root), *args], cwd=root
    )

def sha(data):
    return hashlib.sha256(data).hexdigest()

def same_text(a, b):
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")

def main():
    manifest = json.loads((ROOT / RC / "PACKAGE_MANIFEST.json").read_bytes())
    expected = {RC + "/" + row["relative_path"].replace("\\", "/"): row["sha256"]
                for row in manifest["package_files"]}
    expected.update(manifest["serving_code_sha256"])
    failures = []
    for path, digest in expected.items():
        local = (ROOT / path).read_bytes()
        indexed = git(ROOT, "show", ":" + path)
        ok = sha(indexed) == digest
        print(f"{'PASS' if ok else 'FAIL'} staged {path}"
              + (" (line endings only)" if not ok and sha(local) == digest and same_text(local, indexed) else ""))
        if sha(local) != digest or not ok:
            failures.append(path)
    script = "reservision-backend/experiments/robust_release_candidate_20261124/verify-rc-local.mjs"
    backend = ROOT.parent / "CAPSTONE_BACKEND"
    ok = sha(git(backend, "show", ":" + script)) == manifest["related_verification_script"]["sha256"]
    print(f"{'PASS' if ok else 'FAIL'} staged Node verification script")
    if not ok:
        failures.append(script)
    # Historical Git blobs can have LF while the verified Windows files have CRLF.
    for name, digest in manifest["canonical_sha256"].items():
        local = (ROOT / name).read_bytes()
        baseline = git(ROOT, "show", BASELINE + ":" + name)
        ok = sha(local) == digest and (baseline == local or
             (name.endswith(".json") and same_text(baseline, local)))
        print(f"{'PASS' if ok else 'FAIL'} baseline {name}; git_sha256={sha(baseline)}; "
              f"exact_local_bytes={baseline == local}")
        if not ok:
            failures.append("baseline " + name)
    print("PASS" if not failures else "NOT READY: restage only the reviewed byte-preserved files and rerun")
    return bool(failures)

if __name__ == "__main__":
    sys.exit(main())
