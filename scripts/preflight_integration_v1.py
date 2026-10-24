"""Actual optional-library and both-GPU controls before the H1 pilot freeze."""
from __future__ import annotations

import ast
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cmbcst.integration.family import RotationFamily
from cmbcst.integration.fixtures import simulated_split_alms
from cmbcst.integration.namaster import NaMasterCutSky, CutSkyProfile, cap_mask
from cmbcst.integration.quadratic import SymmetricQuadraticModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/evidence/integration_v1_preflight_final.json')
    args = parser.parse_args()
    out = args.output
    archive = out.with_name(out.stem+'_arrays.npz')
    fixture_path = out.with_name(out.stem+'_fixture.json')
    for path in (out, archive, fixture_path):
        if path.exists():
            raise FileExistsError(f'preflight evidence is immutable; choose a new output: {path}')
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sources = [str(p.relative_to(ROOT)).replace('\\', '/') for p in sorted((ROOT/'src/cmbcst/integration').glob('*.py'))]
    sources.append('scripts/preflight_integration_v1.py')
    source_hashes = {p: digest(ROOT/p) for p in sources}
    test_path = ROOT/'tests/sandbox/jovin_tests_integration_cutsky_v1.py'
    tree = ast.parse(test_path.read_text())
    fixture_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'quadratic_fixture')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=[fixture_function], type_ignores=[]), str(test_path), 'exec'), namespace)
    fixture = namespace['quadratic_fixture']()
    small = SymmetricQuadraticModel(fixture['Q'])
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('two actual CUDA devices are required; no CPU substitution')
    reference = simulated_split_alms(64, 741)
    family = RotationFamily(reference, np.arange(30, 62), planes_per_shell=4, seed=741)
    profile = CutSkyProfile(32, 64, (2, 18, 34, 50, 65))
    adapter = NaMasterCutSky(profile, cap_mask(32, .42, 10.))
    began = time.perf_counter()
    model, scales, audit = adapter.build_model(family)
    build_seconds = time.perf_counter()-began
    if audit['zero_model_scaled_error'] > 1e-10:
        raise AssertionError('zero/reference NaMaster quadratic discrepancy')
    angles = np.random.default_rng(741).uniform(-.15, .15, size=(128, family.nplanes))
    expected, expected_jac = model.features(angles), model.jacobian(angles)
    direct = np.stack([adapter.direct_features(family.candidate_alms(x)) for x in angles[:2]])
    direct_error = float(np.max(np.abs(expected[:2]-direct)/scales))
    if direct_error > 1e-10:
        raise AssertionError(f'direct reconstructed NaMaster discrepancy {direct_error}')
    # Explicit observed-reference control with finite power outside the retained
    # representation. Preserve that original residual, not a synthesized proxy.
    from cmbcst.integration.api import generate_counterfactual_family
    from cmbcst.integration.models import SplitPair, ObservingResponse
    from cmbcst.integration.operators import DuccHealpixOperator
    small_operator = DuccHealpixOperator(8, 12, nthreads=8)
    observed = np.stack([small_operator.synthesize(a) for a in simulated_split_alms(12, 321)])
    observed += .02*np.sin(np.arange(observed.shape[-1]))[None, None]
    response = ObservingResponse('identity', {})
    pair = SplitPair.create(*observed, nside=8, lmax=12, units='uK_CMB', ordering='RING',
        polarization_convention='COSMO', coordinate_frame='G', response_a=response, response_b=response)
    prepared = generate_counterfactual_family(pair, n_planes=4, shells=[6, 9], planes_per_shell=2)
    observed_adapter = NaMasterCutSky(CutSkyProfile(8, 12, (2, 6, 10, 13)), cap_mask(8, .6, 20., holes=False))
    observed_model, observed_scales, observed_audit = observed_adapter.build_prepared_model(prepared)
    ox = np.array([.11, -.14, .18, -.2])
    changed_maps = prepared.candidate_maps(ox)
    observed_direct = observed_adapter.direct_features_from_maps(changed_maps)
    observed_error = float(np.max(np.abs(observed_model.features(ox)-observed_direct)/observed_scales))
    difference_error = float(np.max(np.abs((changed_maps[0]-changed_maps[1])-(observed[0]-observed[1]))/(1+np.abs(observed[0]-observed[1]))))
    if max(observed_error, difference_error, observed_audit['zero_model_scaled_error']) > 1e-10:
        raise AssertionError('original observed-reference/residual control failed')
    arrays = dict(reference_alms=reference, Q=model.Q, scales=scales, angles=angles,
                  cpu_features=expected, cpu_jacobian=expected_jac, direct_features=direct)
    arrays.update({'family_'+key: value for key, value in family.to_arrays().items()})
    attestation = dict(fixture_id='integration_cutsky_v1_quadratic_741',
                       source_sha256=source_hashes, devices=[])
    devices = []
    for index, batch in ((0, 128), (1, 64)):
        device = f'cuda:{index}'
        sm = small.torch(device)
        sx = torch.tensor(fixture['angles'], dtype=torch.float64, device=device, requires_grad=True)
        sf = sm.features(sx).detach().cpu().numpy()
        sj = sm.jacobian(sx).detach().cpu().numpy()
        aj = torch.autograd.functional.jacobian(sm.features, sx)
        diagonal = torch.stack([aj[b, :, b, :] for b in range(len(sx))]).detach().cpu().numpy()
        compact_errors = dict(features=float(np.max(np.abs(sf-small.features(fixture['angles'])))),
                              jacobian=float(np.max(np.abs(sj-small.jacobian(fixture['angles'])))),
                              autograd=float(np.max(np.abs(sj-diagonal))))
        if max(compact_errors.values()) > 1e-10:
            raise AssertionError(f'{device} compact fixture discrepancy')
        attestation['devices'].append(dict(device=device, name=torch.cuda.get_device_name(index),
            angles=fixture['angles'].tolist(), features=sf.tolist(), jacobian=sj.tolist(),
            Q_sha256=small.sha256, errors=compact_errors))
        evaluator = model.torch(device)
        x = torch.tensor(angles[:batch], dtype=torch.float64, device=device)
        evaluator.features(x); evaluator.jacobian(x); torch.cuda.synchronize(index)
        start = time.perf_counter()
        actual = evaluator.features(x)
        jacobian = evaluator.jacobian(x)
        torch.cuda.synchronize(index)
        seconds = time.perf_counter()-start
        actual, jacobian = actual.cpu().numpy(), jacobian.cpu().numpy()
        errors = dict(features=float(np.max(np.abs(actual-expected[:batch])/scales)),
                      jacobian=float(np.max(np.abs(jacobian-expected_jac[:batch])/scales[None, :, None])))
        if max(errors.values()) > 1e-10:
            raise AssertionError(f'{device} native NaMaster quadratic discrepancy {errors}')
        arrays[f'gpu{index}_features'], arrays[f'gpu{index}_jacobian'] = actual, jacobian
        devices.append(dict(device=device, name=torch.cuda.get_device_name(index), native_batch=batch,
                            planes=family.nplanes, features=model.nfeatures, errors=errors,
                            evaluation_seconds_excluding_transfers=seconds,
                            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(index))))
    if any(digest(ROOT/name) != value for name, value in source_hashes.items()):
        raise RuntimeError('source changed during preflight; no successful receipt emitted')
    if any(path.exists() for path in (out, archive, fixture_path)):
        raise FileExistsError('a concurrent writer created the preflight output')
    out.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with fixture_path.open('x') as stream:
        stream.write(json.dumps(attestation, indent=2, allow_nan=False)+'\n')
    # Bind the exact fixture function independently of the attestation literal,
    # which will next be inserted into the sandbox tests before receipt issue.
    fixture_source = ast.get_source_segment(test_path.read_text(), fixture_function)
    report = dict(status='passed', verified_utc=datetime.now(timezone.utc).isoformat(),
                  scope='fixed synthetic preflight; no adversarial pilot search or cosmological finding',
                  source_sha256=source_hashes, fixture_function_sha256=hashlib.sha256(fixture_source.encode()).hexdigest(),
                  versions={x: importlib.metadata.version(x) for x in ('numpy', 'scipy', 'ducc0', 'astropy', 'h5py', 'healpy', 'fitsio', 'pymaster', 'torch')},
                  devices=devices, operator=adapter.provenance(), Q_shape=list(model.Q.shape),
                  model_build_seconds=build_seconds, zero_model_scaled_error=audit['zero_model_scaled_error'],
                  direct_reconstruction_scaled_error=direct_error,
                  observed_reference_control=dict(scaled_feature_error=observed_error,
                      split_difference_scaled_error=difference_error, reference_mode=observed_audit['reference_mode']),
                  arrays_sha256=digest(archive), compact_fixture_sha256=digest(fixture_path),
                  numerical_gate=1e-10, finite_difference_control_gate=1e-7,
                  actual_gpu_fixture_is_recorded_not_executed_inside_jovin=True)
    if any(digest(ROOT/name) != value for name, value in source_hashes.items()):
        raise RuntimeError('source changed before receipt completion; incomplete artifacts retained')
    with out.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
