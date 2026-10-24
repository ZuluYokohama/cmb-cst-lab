"""Prepare or execute the separately frozen, bounded synthetic cut-sky H1 pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cmbcst.integration import (SplitPair, ObservingResponse, generate_counterfactual_family,
    write_split_pair_fits, read_split_pair_fits, write_split_pair_hdf5, read_split_pair_hdf5,
    DuccHealpixOperator, certify_candidate)
from cmbcst.integration.continuation import continue_fiber, direction_manifest
from cmbcst.integration.fixtures import simulated_split_alms
from cmbcst.integration.namaster import NaMasterCutSky, CutSkyProfile, cap_mask

OUT = ROOT/'results/cutsky_feasibility_v1'
VERSION_NAMES = ('numpy', 'scipy', 'ducc0', 'astropy', 'h5py', 'healpy', 'fitsio', 'pymaster', 'torch')
WRITE_LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    # The watchdog and scientific worker may publish the same stage path.
    text = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with WRITE_LOCK:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def source_files():
    files = sorted((ROOT/'src/cmbcst/integration').glob('*.py'))
    files += sorted((ROOT/'tests').glob('test_integration*.py'))
    files += [ROOT/p for p in ('scripts/run_cutsky_feasibility.py', 'scripts/preflight_integration_v1.py',
        'scripts/jovin_verify.py', 'tests/sandbox/jovin_tests_integration_cutsky_v1.py',
        'tests/test_cutsky_driver.py', 'docs/CUTSKY_FEASIBILITY_SPEC.md', 'pyproject.toml')]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files}


def frozen_artifacts(out):
    """Use historical source/result manifests; leave unrelated working drafts alone."""
    saved = {}
    for path in sorted((ROOT/'results').glob('*/verification.json')):
        if path.parent.resolve() == out.resolve():
            continue
        audit = read(path)
        saved[str(path.relative_to(ROOT)).replace('\\', '/')] = sha(path)
        for name, value in audit.get('artifact_sha256', {}).items():
            target = path.parent/name
            saved[str(target.relative_to(ROOT)).replace('\\', '/')] = value
    for path in sorted((ROOT/'results').glob('*/plan.json')):
        if path.parent.resolve() == out.resolve():
            continue
        plan = read(path)
        saved.update(plan.get('source_sha256', {}))
        saved.update(plan.get('v1_frozen_sha256', {}))
    return saved


def check_hashes(values, root=ROOT):
    bad = [name for name, value in values.items() if not (root/name).is_file() or sha(root/name) != value]
    if bad:
        raise RuntimeError('bound bytes differ: '+', '.join(bad[:10]))


def prepare(out):
    if out.exists():
        raise FileExistsError('choose a new versioned output; existing evidence is never reset')
    preflight_path = ROOT/'data/evidence/integration_v1_preflight_final.json'
    profile = read(preflight_path)
    if profile.get('status') != 'passed' or len(profile.get('devices', [])) != 2:
        raise RuntimeError('actual-library and both-GPU preflight must pass')
    check_hashes(profile['source_sha256'])
    fixture_path = preflight_path.with_name(preflight_path.stem+'_fixture.json')
    array_path = preflight_path.with_name(preflight_path.stem+'_arrays.npz')
    if sha(fixture_path) != profile['compact_fixture_sha256'] or sha(array_path) != profile['arrays_sha256']:
        raise RuntimeError('preflight output bindings differ')
    receipts = [r for r in read(ROOT/'data/evidence/jovin_receipts.json') if r['suite'] == 'integration_cutsky_v1']
    receipt = receipts[-1] if receipts else {}
    if not receipt.get('passed') or not receipt.get('receipt_verified') or not receipt.get('chain_valid'):
        raise RuntimeError('authentic current integration JoVin receipt required')
    sys.path.insert(0, str(ROOT/'scripts'))
    from jovin_verify import build_bundle
    bundle, _ = build_bundle('integration_cutsky_v1')
    if (hashlib.sha256(bundle.encode()).hexdigest() != receipt['bundle_sha256']
            or sha(ROOT/'tests/sandbox/jovin_tests_integration_cutsky_v1.py') != receipt['tests_sha256']):
        raise RuntimeError('integration receipt is stale for current source/tests')
    preserved = frozen_artifacts(out)
    check_hashes(preserved)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    sources = source_files()
    for name, value in sources.items():
        committed = subprocess.check_output(['git', 'cat-file', 'blob', f'{revision}:{name}'], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != value:
            raise RuntimeError('freeze commit differs from current scientific source: '+name)
    plan = dict(format='cmbcst.cutsky_feasibility.v1', created_utc=now().isoformat(),
        source_revision=revision, source_sha256=sources, prior_frozen_sha256=preserved,
        versions={name: importlib.metadata.version(name) for name in VERSION_NAMES},
        preflight_path=str(preflight_path.relative_to(ROOT)).replace('\\', '/'),
        preflight_sha256=sha(preflight_path), preflight_fixture_sha256=sha(fixture_path),
        preflight_arrays_sha256=sha(array_path), jovin_receipt_id=receipt['receipt_id'],
        jovin_bundle_sha256=receipt['bundle_sha256'], jovin_tests_sha256=receipt['tests_sha256'],
        seed=82024102027, nside=32, lmax=64, shells=list(range(30, 62)), planes_per_shell=4,
        n_planes=128, fractions=[.1, .3, .6], apodization_degrees=[8., 12.], holes=True,
        band_edges=[2, 18, 34, 50, 65], tolerances=[1e-10, 1e-12], starts_per_case=4,
        max_iterations=100, max_stalls=25, rank_cutoff=1e-10, step_norm=.1,
        corrections=5, halvings=10, minimum_harmonic_rms=.01, maximum_rms=.25,
        angle_limit=float(np.pi/4), compute_cap_seconds=1800., verification_reserve_seconds=180.,
        threads=8, blas_threads=1, purify_e=False, purify_b=True, mask_lmax=64,
        input_model='simulated Gaussian common sky and independent isotropic split noise; common .02*sin(pixel_index) residual; uK_CMB',
        scope='H1 finite-family feasibility only; coupled binned realized spectra and canonical harmonic Gram; no future-response or physical-inference claim',
        parameter_seed_derivation='family SeedSequence([seed,ell]); direction SeedSequence([seed,profile_index*4+start]); same directions at both tolerances',
        planned_profiles=6, planned_cases=12, planned_starts=48)
    out.mkdir(parents=True)
    save_json(out/'plan.json', plan)
    print(json.dumps(dict(status='prepared', out=str(out), source_revision=plan['source_revision'],
                          plan_sha256=sha(out/'plan.json'), prior_artifacts=len(preserved)), indent=2))


class Budget:
    def __init__(self, cap, reserve):
        if (isinstance(cap, bool) or isinstance(reserve, bool) or not np.isfinite([cap, reserve]).all()
                or not 0 <= reserve < cap <= 1800):
            raise ValueError('finite bounded cap and smaller nonnegative verification reserve required')
        self.utc, self.monotonic = now(), time.perf_counter()
        self.cap, self.reserve = float(cap), float(reserve)

    def clocks(self):
        monotonic = time.perf_counter()-self.monotonic
        utc = (now()-self.utc).total_seconds()
        return dict(elapsed_seconds=monotonic, utc_elapsed_seconds=utc,
                    budget_elapsed_seconds=max(monotonic, utc))

    def check(self, search=False):
        if self.clocks()['budget_elapsed_seconds'] >= self.cap-(self.reserve if search else 0):
            raise TimeoutError('frozen scientific budget/reserve exhausted')


def execute(out, expected_plan_sha256=None):
    plan = read(out/'plan.json')
    if (out/'stage_audit.json').exists():
        raise FileExistsError('a started pilot cannot be restarted or extended')
    if not isinstance(expected_plan_sha256, str) or sha(out/'plan.json') != expected_plan_sha256:
        raise RuntimeError('supply the independently frozen plan SHA256 before executing')
    check_hashes(plan['source_sha256'])
    current_versions = {name: importlib.metadata.version(name) for name in VERSION_NAMES}
    if current_versions != plan['versions']:
        raise RuntimeError('frozen numerical dependency versions differ')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('both actual GPUs required; no CPU substitution')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    lock = out/'.run.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    budget = Budget(plan['compute_cap_seconds'], plan['verification_reserve_seconds'])
    stage = dict(status='running', started_utc=budget.utc.isoformat(), plan_sha256=sha(out/'plan.json'),
                 scientific_scope=plan['scope'])
    save_json(out/'stage_audit.json', stage)
    stop = threading.Event()
    def hard_cap():
        while not stop.wait(.25):
            if budget.clocks()['budget_elapsed_seconds'] >= budget.cap:
                try:
                    save_json(out/'hard_cap.json', dict(status='incomplete', **budget.clocks()))
                    save_json(out/'stage_audit.json', dict(stage, status='incomplete_hard_cap', **budget.clocks()))
                finally:
                    os._exit(124)  # This job only; never terminate the shared WSL.
    threading.Thread(target=hard_cap, daemon=True).start()
    profiles, rows, cases, failure = [], [], [], None
    try:
        operator = DuccHealpixOperator(plan['nside'], plan['lmax'], nthreads=plan['threads'])
        original = simulated_split_alms(plan['lmax'], plan['seed'])
        maps = np.stack([operator.synthesize(a) for a in original])
        maps += .02*np.sin(np.arange(maps.shape[-1]))[None, None]
        identity = ObservingResponse('identity', {})
        pair = SplitPair.create(*maps, nside=plan['nside'], lmax=plan['lmax'],
            ordering='RING', coordinate_frame='G', units='uK_CMB', polarization_convention='COSMO',
            response_a=identity, response_b=identity)
        inputs = out/'inputs'; inputs.mkdir()
        write_split_pair_fits(inputs/'splits.fits', pair)
        write_split_pair_hdf5(inputs/'splits.h5', pair)
        fits_pair, hdf_pair = read_split_pair_fits(inputs/'splits.fits'), read_split_pair_hdf5(inputs/'splits.h5')
        if fits_pair.provenance()['a_sha256'] != hdf_pair.provenance()['a_sha256'] or fits_pair.provenance()['b_sha256'] != hdf_pair.provenance()['b_sha256']:
            raise RuntimeError('actual FITS/HDF canonical input mismatch')
        prepared = generate_counterfactual_family(fits_pair, n_planes=plan['n_planes'],
            shells=plan['shells'], planes_per_shell=plan['planes_per_shell'], seed=plan['seed'])
        family = prepared.family
        save_arrays(inputs/'family.npz', **family.to_arrays())
        save_arrays(inputs/'prepared_arrays.npz', reference_alms=family.reference_alms,
                    reference_maps=prepared.reference_maps, residual_maps=prepared.residual_maps)
        save_json(inputs/'provenance.json', prepared.provenance())
        directions = {}
        for profile_index in range(plan['planned_profiles']):
            for start in range(plan['starts_per_case']):
                gid = profile_index*plan['starts_per_case']+start
                directions[str(gid)] = direction_manifest(family.nplanes, plan['seed'], gid)[1]
        save_json(inputs/'directions.json', directions)
        input_manifest = {str(p.relative_to(out)).replace('\\', '/'): sha(p) for p in sorted(inputs.iterdir()) if p.is_file()}
        save_json(inputs/'manifest.json', input_manifest)
        profile_index = 0
        for fraction in plan['fractions']:
            for apod in plan['apodization_degrees']:
                budget.check(search=True)
                name = f'fsky_{round(fraction*100):02d}_apod_{int(apod)}'
                save_json(out/'progress.json', dict(phase='preprocessing', profile=name, **budget.clocks()))
                mask = cap_mask(plan['nside'], fraction, apod, holes=plan['holes'])
                profile = CutSkyProfile(plan['nside'], plan['lmax'], tuple(plan['band_edges']),
                    purify_e=plan['purify_e'], purify_b=plan['purify_b'], mask_lmax=plan['mask_lmax'])
                adapter = NaMasterCutSky(profile, mask)
                model, scales, model_audit = adapter.build_prepared_model(prepared, budget=lambda: budget.check(search=True))
                if model_audit['zero_model_scaled_error'] > min(plan['tolerances']):
                    raise RuntimeError('fixed zero-state direct/model error exceeds the stricter declared profile')
                model_path = out/'profiles'/f'{name}.npz'
                save_arrays(model_path, Q=model.Q, scales=scales, mask=mask,
                    reference_features=model_audit['reference_features'])
                record = dict(name=name, nominal_fraction=fraction, apodization_degrees=apod,
                    positive_pixel_fraction=float(np.mean(mask > 0)), mean_weight=float(np.mean(mask)),
                    mean_squared_weight=float(np.mean(mask**2)), operator=adapter.provenance(),
                    arrays=str(model_path.relative_to(out)), arrays_sha256=sha(model_path),
                    zero_model_scaled_error=model_audit['zero_model_scaled_error'])
                save_json(out/'profiles'/f'{name}.json', record); profiles.append(record)
                reference_features = adapter.direct_features_from_maps(prepared.reference_maps)
                gpu_models = [model.torch(f'cuda:{i}') for i in (0, 1)]
                for tolerance in plan['tolerances']:
                    case_name = f'{name}_gate_{tolerance:.0e}'
                    case_rows = []
                    for start in range(plan['starts_per_case']):
                        budget.check(search=True)
                        gid = profile_index*plan['starts_per_case']+start
                        row_path = out/'cases'/case_name/f'start_{start:04d}.json'
                        row = dict(case=case_name, profile=name, start_id=start, global_start_id=gid,
                                   tolerance=tolerance, status='in_progress')
                        save_json(row_path, row)
                        view = dict(case=case_name, profile=name, start_id=start, global_start_id=gid,
                                    tolerance=tolerance, status='in_progress', audit=str(row_path.relative_to(out)),
                                    audit_sha256=sha(row_path), harmonic_rms=None, iterations=0)
                        rows.append(view); case_rows.append(view)
                        save_json(out/'progress.json', dict(phase='continuation', case=case_name, start=start, **budget.clocks()))
                        solver = continue_fiber(model, scales, family, global_start_id=gid, seed=plan['seed'],
                            max_iterations=plan['max_iterations'], max_stalls=plan['max_stalls'],
                            step_norm=plan['step_norm'], rank_cutoff=plan['rank_cutoff'],
                            corrections=plan['corrections'], halvings=plan['halvings'],
                            restore_target=min(1e-12, tolerance/10), tolerance=tolerance,
                            minimum_rms=plan['minimum_harmonic_rms'], maximum_rms=plan['maximum_rms'],
                            budget=lambda: budget.check(search=True))
                        row.update(solver=solver, status=solver['status'])
                        save_json(row_path, row)
                        if solver['status'] == 'candidate_found':
                            budget.check()
                            theta = np.asarray(solver['theta'])
                            candidate = family.candidate_alms(theta)
                            candidate_maps = prepared.candidate_maps(theta)
                            direct = adapter.direct_features_from_maps(candidate_maps)
                            predicted = model.features(theta)
                            direct_error = float(np.max(np.abs(predicted-direct)/scales))
                            archive = out/'cases'/case_name/f'start_{start:04d}.npz'
                            save_arrays(archive, theta=theta, candidate_alms=candidate, candidate_maps=candidate_maps,
                                direct_features=direct, reference_features=reference_features, scales=scales,
                                predicted_features=predicted)
                            row.update(arrays=str(archive.relative_to(out)), arrays_sha256=sha(archive),
                                       verification_status='independent_verification_in_progress')
                            save_json(row_path, row)
                            gpu_errors, gpu_values = {}, []
                            for device, evaluator in enumerate(gpu_models):
                                budget.check()
                                x = torch.tensor(theta[None], dtype=torch.float64, device=evaluator.device)
                                value = evaluator.features(x).detach().cpu().numpy()[0]
                                gpu_values.append(value)
                                gpu_errors[f'cuda:{device}'] = float(np.max(np.abs(value-direct)/scales))
                            difference_error = float(np.max(np.abs((candidate_maps[0]-candidate_maps[1])
                                -(prepared.reference_maps[0]-prepared.reference_maps[1]))/(1+np.abs(prepared.reference_maps[0]-prepared.reference_maps[1]))))
                            certificate = certify_candidate(family.reference_alms, candidate,
                                reference_features, direct, scales, angle_values=theta,
                                tolerance=tolerance, min_rms=plan['minimum_harmonic_rms'], max_rms=plan['maximum_rms'],
                                direct_feature_error=max(direct_error, *gpu_errors.values()),
                                reference_maps=prepared.reference_maps, candidate_maps=candidate_maps, weights=mask**2,
                                provenance=dict(operator=record['operator']['operator_sha256'], plan_sha256=stage['plan_sha256'],
                                    profile_arrays_sha256=record['arrays_sha256'], independent_path='reconstructed original maps plus full NaMaster fields'))
                            verified = certificate['accepted'] and difference_error <= tolerance and max(gpu_errors.values()) <= tolerance
                            row.update(status='finite_feasible_candidate_verified' if verified else 'verification_failed',
                                certificate=certificate, actual_both_gpu_errors=gpu_errors,
                                observed_split_difference_scaled_error=difference_error,
                                direct_model_scaled_error=direct_error)
                            save_arrays(archive, theta=theta, candidate_alms=candidate, candidate_maps=candidate_maps,
                                direct_features=direct, reference_features=reference_features, scales=scales,
                                predicted_features=predicted, actual_gpu_features=np.stack(gpu_values))
                            row.update(arrays=str(archive.relative_to(out)), arrays_sha256=sha(archive))
                        save_json(row_path, row)
                        view.update(status=row['status'], audit_sha256=sha(row_path),
                                    harmonic_rms=solver['rms_ratio'], iterations=solver['iterations'])
                    case = dict(name=case_name, profile=name, tolerance=tolerance, starts=len(case_rows),
                        verified=sum(r['status'] == 'finite_feasible_candidate_verified' for r in case_rows), rows=case_rows)
                    cases.append(case); save_json(out/'cases'/case_name/'case.json', case)
                profile_index += 1
                del gpu_models
        budget.check()
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        save_json(out/'failure.json', failure)
        for view in rows:
            if view['status'] == 'in_progress':
                interrupted_path = out/view['audit']
                interrupted = read(interrupted_path)
                interrupted.update(status='incomplete', execution_failure=failure)
                save_json(interrupted_path, interrupted)
                view.update(status='incomplete', audit_sha256=sha(interrupted_path))
    finally:
        # No scientific calculations follow this boundary. Later serialization
        # is saved-evidence metadata work, with its own timestamp.
        scientific_clocks = budget.clocks()
        scientific_completed_utc = now().isoformat()
        stop.set()
        failures = sum(r['status'] in ('numerical_failure', 'verification_failed') for r in rows)
        incomplete = (len(rows) < plan['planned_starts'] or any(r['status'] == 'incomplete' for r in rows)
                      or scientific_clocks['budget_elapsed_seconds'] >= budget.cap)
        status = 'incomplete' if incomplete else ('complete_with_failures' if failure or failures else 'complete')
        result = dict(status=status, scientific_scope=plan['scope'], planned_starts=plan['planned_starts'],
            starts_started=len(rows), verified_candidates=sum(r['status'] == 'finite_feasible_candidate_verified' for r in rows),
            numerical_or_verification_failures=failures, profiles=profiles, cases=cases, rows=rows,
            planned_cases=plan['planned_cases'], completed_cases=len(cases), failure=failure,
            completed_utc=scientific_completed_utc, plan_sha256=stage['plan_sha256'], **scientific_clocks)
        result['input_manifest_sha256'] = sha(out/'inputs/manifest.json') if (out/'inputs/manifest.json').exists() else None
        result['scientific_outcome'] = ('verified_nontrivial_family' if result['verified_candidates'] else
            ('none_found_in_budget' if status == 'complete' else 'incomplete_or_failed'))
        save_json(out/'result.json', result)
        save_json(out/'stage_audit.json', dict(stage, status=status, completed_utc=result['completed_utc'],
            result_sha256=sha(out/'result.json'), metadata_completed_utc=now().isoformat(), **scientific_clocks))
        lock.unlink(missing_ok=True)
    print(json.dumps(dict(status=result['status'], starts=result['starts_started'], verified=result['verified_candidates'],
                         failures=result['numerical_or_verification_failures'], **scientific_clocks), indent=2))
    return 0 if result['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--plan-sha256', help='required frozen plan digest for run mode')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != (ROOT/'results').resolve() or not out.name.startswith('cutsky_feasibility_'):
        raise ValueError('use a separate versioned cutsky_feasibility output under results')
    if args.mode == 'prepare':
        prepare(out)
        return 0
    return execute(out, args.plan_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
