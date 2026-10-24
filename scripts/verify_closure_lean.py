"""Kernel-check conditional closure and its rational counterexample; preserve outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PINNED_ARCHIVE_SHA256 = '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
PINNED_COMPILER_SHA256 = 'bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38'
PINNED_COMMIT = 'eb6cafb7290e8bd5f2937522d8d804f58f02bafc'
THEOREMS = (
    'no_predictor_from_collision', 'no_factor_from_collision',
    'toy_initial_moments', 'toy_left_next_moments',
    'toy_right_next_moments', 'toy_next_moments_differ', 'toy_no_moment_factor',
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean', type=Path, default=Path('/opt/cmb-lean-4.35.0/bin/lean'))
    parser.add_argument('--archive', type=Path,
                        default=Path('/opt/cmb-lean-downloads/lean-4.35.0-linux.tar.zst'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing proof evidence; choose a separate versioned output')
    source = ROOT / 'formal/OperationalClosure.lean'
    text = source.read_text(encoding='utf-8')
    if re.search(r'\b(sorry|admit)\b|^\s*(axiom|opaque)\s', text, re.MULTILINE):
        raise ValueError('proof source contains a forbidden unproved declaration')
    names = re.findall(r'^theorem\s+(\w+)', text, re.MULTILINE)
    if tuple(names) != THEOREMS:
        raise ValueError('expected the seven conditional/finite-example theorem declarations')
    archive_hash, compiler_hash = sha(args.archive), sha(args.lean)
    if archive_hash != PINNED_ARCHIVE_SHA256 or compiler_hash != PINNED_COMPILER_SHA256:
        raise ValueError('compiler/archive bytes differ from the pinned official release evidence')
    version = subprocess.check_output([str(args.lean), '--version'], text=True).strip()
    if 'version 4.35.0' not in version or PINNED_COMMIT not in version:
        raise ValueError('Lean version/commit differs from the pinned compiler')
    start = time.perf_counter()
    compiled = subprocess.run([str(args.lean), str(source)], cwd=ROOT, text=True,
                              capture_output=True, timeout=120, check=False)
    assumptions = {}
    for name in names:
        pattern = (rf"'CmbCst\.{re.escape(name)}' "
                   r'(?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)')
        match = re.search(pattern, compiled.stdout)
        if match is not None:
            assumptions[name] = ([value.strip() for value in match.group(1).split(',')]
                                 if match.group(1) else [])
    allowed = {'propext', 'Classical.choice', 'Quot.sound'}
    passed = (compiled.returncode == 0 and len(assumptions) == len(THEOREMS)
              and all(set(values) <= allowed for values in assumptions.values())
              and 'sorryAx' not in compiled.stdout + compiled.stderr)
    rotation_source = ROOT / 'formal/PolarizationGeometry.lean'
    evidence = dict(
        status='passed' if passed else 'failed',
        verified_utc=datetime.now(timezone.utc).isoformat(),
        version=version, compiler_sha256=compiler_hash, archive_sha256=archive_hash,
        compiler_archive_url=('https://github.com/leanprover/lean4/releases/download/'
                              'v4.35.0/lean-4.35.0-linux.tar.zst'),
        source_sha256=sha(source), verifier_sha256=sha(Path(__file__)),
        theorem_count=len(names), theorems=names, theorem_axioms=assumptions,
        exit_code=compiled.returncode, stdout=compiled.stdout, stderr=compiled.stderr,
        seconds=time.perf_counter() - start,
        supporting_rotation_source_sha256=sha(rotation_source),
        supporting_rotation_theorems=['CmbCst.polarization_amplitude_invariant',
                                      'CmbCst.corotated_measurement_invariant'],
        scope=('Conditional typed-function no-factor criterion and an exact rational four-node '
               'counterexample only. Existing rotation identities are referenced unchanged. '
               'This does not verify Float64 feature calculations, an optimizer, sky physics, '
               'universal closure of a repaired representation, or mathematical novelty.'),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=evidence['status'], theorem_count=len(names),
                         seconds=evidence['seconds'], axioms=assumptions), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
