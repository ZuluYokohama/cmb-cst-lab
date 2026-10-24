"""Kernel-check the saved V2 bin witness; bind measured evidence without replay.

The proof concerns exact integer tables and a conditional factor-map theorem.
This verifier binds the measured tables, arrays, coefficients and frozen code by
SHA-256. It never computes sky features, transforms, searches or quantization.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = '7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d'
COMPILER_SHA = 'bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38'
LEAN_COMMIT = 'eb6cafb7290e8bd5f2937522d8d804f58f02bafc'
NAMESPACE = 'CmbCst.ClosureFibersV2'
THEOREMS = ('recorded_table_lengths', 'recorded_current_bins_equal',
            'recorded_primary_future_bins', 'recorded_primary_values_differ',
            'recorded_future_bins_differ', 'conditional_recorded_no_factor')
TABLES = ('currentReferenceBins', 'currentCandidateBins', 'futureReferenceBins', 'futureCandidateBins')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(text, name):
    match = re.search(rf'def\s+{re.escape(name)}\s*:\s*List Int\s*:=\s*(\[[^\]]*\])', text)
    if match is None:
        raise ValueError('missing exact Lean bin table: '+name)
    value = ast.literal_eval(match.group(1))
    if len(value) != 44 or any(type(x) is not int for x in value):
        raise ValueError('44 exact integer bins required: '+name)
    return value


def relative(path):
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run(command, cwd, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            text=True, check=False, timeout=120)
    return dict(command=command, cwd=str(cwd), exit_code=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lean', type=Path, default=Path('/opt/cmb-lean-4.35.0/bin/lean'))
    parser.add_argument('--archive', type=Path, default=Path('/opt/cmb-lean-downloads/lean-4.35.0-linux.tar.zst'))
    parser.add_argument('--run-dir', type=Path, default=ROOT/'results/closure_fibers_v2')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'data/evidence/closure_v2_first_witness_lean.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing formal evidence; use a separate versioned output')
    formal = ROOT/'formal'
    source = formal/'ClosureFibersV2Witness.lean'
    dependency = formal/'OperationalClosure.lean'
    source_text = source.read_text(encoding='utf-8')
    if re.search(r'\b(sorry|admit)\b|^\s*(axiom|opaque)\s', source_text, re.MULTILINE):
        raise ValueError('new proof contains an unproved declaration')
    if tuple(re.findall(r'^theorem\s+(\w+)', source_text, re.MULTILINE)) != THEOREMS:
        raise ValueError('unexpected witness theorem inventory')
    if not source_text.startswith('import OperationalClosure\n'):
        raise ValueError('specialize the unchanged imported proof, not a copied theorem')
    exact_tables = {name: table(source_text, name) for name in TABLES}
    out = args.run_dir
    case_dir = out/'certificates_v2/tile_24_source_baseline'
    case_path, row_path = case_dir/'case.json', case_dir/'rows/start_0056.json'
    case, row, plan = read(case_path), read(row_path), read(out/'plan.json')
    measured_tables = dict(currentReferenceBins=[0]*44, currentCandidateBins=row['input_bins']['bins'],
                           futureReferenceBins=row['future_bins'][0]['reference_bins'],
                           futureCandidateBins=row['future_bins'][0]['candidate_bins'])
    if exact_tables != measured_tables:
        raise ValueError('Lean integer tables differ from the declared measured witness')
    view = next(item for item in case['rows'] if item['start_id'] == 56)
    row_arrays, coefficients = out/row['arrays'], out/row['coefficients']
    linked = {relative(path): sha(path) for path in
              (out/'plan.json', case_path, row_path, row_arrays, coefficients,
               out/case['case']['arrays'], out/'reference_features.npz',
               out/'normalization_lock.json', out/'operators/family_24.json',
               out/'operators/planes_24.npz')}
    scientific_bindings = {**plan['source_sha256'], **plan['input_sha256']}
    if not all(sha(ROOT/name) == digest for name, digest in scientific_bindings.items()):
        raise ValueError('frozen scientific source/input bindings changed')
    if (sha(row_path) != view['audit_sha256'] or sha(row_arrays) != row['arrays_sha256']
            or sha(coefficients) != row['coefficients_sha256']
            or sha(out/case['case']['arrays']) != case['case']['arrays_sha256']):
        raise ValueError('declared witness artifact hash differs')
    if not (case['status'] == 'complete' and case['starts_checked'] == case['starts_started'] == 128
            and case['maximum_selected_start'] == 56 and row['start_id'] == 56
            and row['certified'] is True and row['raw_primary_gap'] >= .05
            and row['input_bins']['matched'] is True and row['input_bins']['minimum_margin'] >= 1e-5
            and row['safe_primary_future_bin_difference'] is True
            and all(item['passed'] for item in row['invariant_checks'])
            and set(row['actual_both_gpu_errors']) == {'gpu0', 'gpu1'}
            and max(row['actual_both_gpu_errors'].values()) <= 1e-10
            and measured_tables['currentReferenceBins'] == measured_tables['currentCandidateBins']
            and measured_tables['futureReferenceBins'][22] == 232
            and measured_tables['futureCandidateBins'][22] == 1007):
        raise ValueError('this receipt is restricted to the declared first certified source witness')
    archive_hash, compiler_hash = sha(args.archive), sha(args.lean)
    if archive_hash != ARCHIVE_SHA or compiler_hash != COMPILER_SHA:
        raise ValueError('pinned compiler/archive hash mismatch')
    version = subprocess.check_output([str(args.lean), '--version'], text=True).strip()
    if 'version 4.35.0' not in version or LEAN_COMMIT not in version:
        raise ValueError('pinned Lean version/commit mismatch')
    prior = read(ROOT/'data/evidence/closure_lean.json')
    if sha(dependency) != prior['source_sha256'] or prior['status'] != 'passed':
        raise ValueError('unchanged conditional-theorem source binding failed')
    cache = args.cache_dir or ROOT/'data/work/closure_v2_witness_lean'/sha(dependency)[:16]
    if cache.resolve() == formal.resolve() or formal.resolve() in cache.resolve().parents:
        raise ValueError('compiled dependencies must stay outside the frozen formal source tree')
    cache.mkdir(parents=True, exist_ok=True)
    cached_olean = cache/'OperationalClosure.olean'
    beside_source = formal/'OperationalClosure.olean'
    adjacent_before = sha(beside_source) if beside_source.exists() else None
    before_sources = {relative(path): sha(path) for path in (source, dependency, Path(__file__))}
    started = time.perf_counter()
    build = run([str(args.lean), '-o', str(cached_olean.resolve()), dependency.name], formal)
    environment = os.environ.copy()
    environment['LEAN_PATH'] = str(cache.resolve())+(os.pathsep+environment['LEAN_PATH'] if environment.get('LEAN_PATH') else '')
    compiled = run([str(args.lean), source.name], formal, environment)
    assumptions = {}
    for name in THEOREMS:
        match = re.search(rf"'{re.escape(NAMESPACE)}\.{re.escape(name)}' "
                          r'(?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)', compiled['stdout'])
        if match is not None:
            assumptions[name] = [item.strip() for item in match.group(1).split(',')] if match.group(1) else []
    adjacent_after = sha(beside_source) if beside_source.exists() else None
    unchanged = all(sha(ROOT/name) == digest for name, digest in before_sources.items())
    passed = (build['exit_code'] == compiled['exit_code'] == 0
              and len(assumptions) == len(THEOREMS) and all(not xs for xs in assumptions.values())
              and 'sorryAx' not in compiled['stdout']+compiled['stderr']
              and adjacent_before == adjacent_after and unchanged)
    receipt = dict(status='passed' if passed else 'failed', verified_utc=datetime.now(timezone.utc).isoformat(),
                   version=version, compiler_sha256=compiler_hash, archive_sha256=archive_hash,
                   compiler_archive_url='https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux.tar.zst',
                   source=relative(source), source_sha256=sha(source), verifier_sha256=sha(Path(__file__)),
                   dependency_source=relative(dependency), dependency_source_sha256=sha(dependency),
                   dependency_olean_path=relative(cached_olean), dependency_olean_sha256=sha(cached_olean) if cached_olean.exists() else None,
                   dependency_build=build, witness_compile=compiled, lean_path=environment['LEAN_PATH'],
                   formal_source_tree_olean_unchanged=adjacent_before == adjacent_after,
                   proof_sources_unchanged=unchanged, theorem_count=len(THEOREMS),
                   theorems=[NAMESPACE+'.'+name for name in THEOREMS], theorem_axioms=assumptions,
                   exact_integer_tables=exact_tables, primary_coordinate=22, primary_bins=[232, 1007],
                   linked_artifact_sha256=linked, scientific_source_revision=plan['source_revision'],
                   scientific_source_sha256=plan['source_sha256'], scientific_input_sha256=plan['input_sha256'],
                   reconstructed_coefficient_content_sha256=row['candidate_harmonics_sha256'],
                   measured_raw_gap=row['raw_primary_gap'], seconds=time.perf_counter()-started,
                   explicit_mathematical_hypotheses=[
                       'State and Global are arbitrary types; R : State -> Global x List Int and T : State -> State.',
                       'The two current Global components are equal (retained global representation, including its spectral bins).',
                       'The two current local components equal the exact currentReferenceBins/currentCandidateBins tables.',
                       'The two future local components under the same T equal the exact futureReferenceBins/futureCandidateBins tables.'
                   ],
                   conclusion='Under those hypotheses no deterministic F on the recorded product representation satisfies R(T x)=F(R x) for every state.',
                   scope='Kernel-checked integer-table distinction and specialization of the unchanged conditional theorem only. '
                         'Measured Float64 quantization, ducc sampling, GPU execution, raw 0.05 threshold, spectra, sky physics and '
                         'physical interpretation are not certified by Lean. No universal repair, cosmological or priority claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], theorem_count=len(THEOREMS), axioms=assumptions,
                         source_sha256=receipt['source_sha256'], seconds=receipt['seconds']), indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
