"""A fresh checkout must retain the bytes bound into the scientific run plans."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_numerical_source_bytes_match_every_run_plan():
    plans = list((ROOT / 'results').glob('*/plan.json'))
    assert len(plans) >= 4
    for path in plans:
        plan = json.loads(path.read_text())
        for source, digest in plan.get('source_sha256', {}).items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, (path, source)


def test_saved_verification_artifact_bytes_match():
    audits = list((ROOT / 'results').glob('*/verification.json'))
    assert audits
    for path in audits:
        evidence = json.loads(path.read_text())
        for artifact, digest in evidence.get('artifact_sha256', {}).items():
            assert hashlib.sha256((path.parent/artifact).read_bytes()).hexdigest() == digest, (path, artifact)
