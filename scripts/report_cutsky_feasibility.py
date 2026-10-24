"""Publish saved cut-sky H1 evidence; never run a transform or feature evaluator."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/cutsky_feasibility_v1'
VERIFIED = 'finite_feasible_candidate_verified'
TERMINAL = (VERIFIED, 'no_nontrivial_found', 'numerical_failure', 'verification_failed')


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def load_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    """Family, NaMaster and certificate framing; intentionally not models v1."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def models_array_sha(value):
    """Independent copy of the declared models array-v1 byte framing only.

    This hashes saved arrays; it performs no map or harmonic calculation.
    The domain prefix, canonical endian and JSON header distinguish this
    format from the family/NaMaster/certificate array framing above.
    """
    value = np.asarray(value)
    if value.dtype.kind not in 'bfciu' or value.dtype.hasobject:
        raise ValueError('numeric or boolean saved array required')
    value = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder('<'))
    header = json.dumps(dict(dtype=value.dtype.str, shape=list(value.shape)),
                        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    digest = hashlib.sha256(b'cmbcst.array.v1\0'+header+b'\0')
    digest.update(value.tobytes())
    return digest.hexdigest()


def number(value):
    return float(value) if isinstance(value, (float, int)) and np.isfinite(value) else None


def fmt(value, digits=6):
    return '—' if number(value) is None else f'{float(value):.{digits}g}'


def manifest(plan):
    cases = []
    for index, (fraction, apod) in enumerate((f, a) for f in plan['fractions'] for a in plan['apodization_degrees']):
        profile = f'fsky_{round(fraction*100):02d}_apod_{int(apod)}'
        for tolerance in plan['tolerances']:
            cases.append(dict(name=f'{profile}_gate_{tolerance:.0e}', profile=profile, profile_index=index,
                              nominal_fraction=fraction, apodization_degrees=apod, tolerance=tolerance))
    return cases


def collect_run(out=OUT):
    out = Path(out)
    plan = read(out/'plan.json')
    if plan is None:
        raise FileNotFoundError('frozen cut-sky plan required')
    stage = read(out/'stage_audit.json', dict(status='not_started'))
    result = read(out/'result.json')
    profiles = {p['name']: p for p in (result or {}).get('profiles', [])}
    entries, all_rows = [], []
    for expected in manifest(plan):
        name, profile = expected['name'], expected['profile']
        record = read(out/'cases'/name/'case.json')
        profile_record = read(out/'profiles'/f'{profile}.json', profiles.get(profile))
        rows = []
        for start in range(plan['starts_per_case']):
            path = out/'cases'/name/f'start_{start:04d}.json'
            row = read(path)
            if row is None:
                continue
            rows.append(dict(row=row, path=path, audit_sha256=sha(path)))
        all_rows.extend(rows)
        states = {}
        for item in rows:
            state = item['row'].get('status', 'invalid_missing_status')
            states[state] = states.get(state, 0)+1
        entries.append(dict(expected=expected, record=record, profile_record=profile_record, rows=rows,
            started=len(rows), terminal=sum(x['row'].get('status') in TERMINAL for x in rows),
            verified=states.get(VERIFIED, 0), no_nontrivial_found=states.get('no_nontrivial_found', 0),
            failed=states.get('numerical_failure', 0)+states.get('verification_failed', 0),
            incomplete=sum(n for state, n in states.items() if state not in TERMINAL), states=states))
    verified = [x['row'] for x in all_rows if x['row'].get('status') == VERIFIED]
    by_tolerance = {}
    for tolerance in plan['tolerances']:
        group = [e for e in entries if e['expected']['tolerance'] == tolerance]
        by_tolerance[f'{tolerance:.0e}'] = dict(planned=len(group)*plan['starts_per_case'],
            started=sum(e['started'] for e in group), verified=sum(e['verified'] for e in group),
            no_nontrivial_found=sum(e['no_nontrivial_found'] for e in group),
            failed=sum(e['failed'] for e in group), incomplete=sum(e['incomplete'] for e in group))
    status_counts, reason_counts = {}, {}
    for item in all_rows:
        row = item['row']; state = row.get('status', 'invalid_missing_status')
        status_counts[state] = status_counts.get(state, 0)+1
        reason = row.get('solver', {}).get('reason', 'solver_not_recorded')
        reason_counts[reason] = reason_counts.get(reason, 0)+1
    def maximum(values):
        actual = [float(v) for v in values if number(v) is not None]
        return max(actual) if actual else None
    def minimum(values):
        actual = [float(v) for v in values if number(v) is not None]
        return min(actual) if actual else None
    ranks = [rank for item in all_rows for rank in item['row'].get('solver', {}).get('rank_history', [])
             if isinstance(rank, int) and not isinstance(rank, bool)]
    direction_streams = {item['row']['global_start_id'] for item in all_rows}
    paired_amplitudes = {}
    for row in verified:
        paired_amplitudes.setdefault(row['global_start_id'], {})[row['tolerance']] = row['certificate']['metrics']['harmonic_coadd_perturbation_ratio']
    amplitude_ratios = [group[plan['tolerances'][1]]/group[plan['tolerances'][0]]
                        for group in paired_amplitudes.values()
                        if len(plan['tolerances']) == 2 and all(t in group for t in plan['tolerances'])
                        and group[plan['tolerances'][0]] > 0]
    summary = dict(execution_status=stage.get('status'), planned_profiles=plan['planned_profiles'],
        planned_cases=plan['planned_cases'], planned_starts=plan['planned_starts'],
        profiles_preprocessed=sum(read(out/'profiles'/f'{name}.json') is not None
                                  for name in {e['expected']['profile'] for e in entries}),
        closed_cases=sum(e['record'] is not None for e in entries), starts_started=len(all_rows),
        starts_terminal=sum(e['terminal'] for e in entries), verified_candidates=len(verified),
        no_nontrivial_found=sum(e['no_nontrivial_found'] for e in entries),
        numerical_or_verification_failures=sum(e['failed'] for e in entries),
        incomplete_started=sum(e['incomplete'] for e in entries),
        unstarted=plan['planned_starts']-len(all_rows), status_counts=status_counts, solver_reason_counts=reason_counts,
        tolerance_outcomes=by_tolerance,
        planned_direction_streams=plan['planned_profiles']*plan['starts_per_case'],
        direction_streams_started=len(direction_streams), verified_tolerance_pairs=len(amplitude_ratios),
        minimum_solver_rank=min(ranks, default=None), maximum_solver_rank=max(ranks, default=None),
        minimum_paired_harmonic_rms_ratio=minimum(amplitude_ratios),
        maximum_paired_harmonic_rms_ratio=maximum(amplitude_ratios),
        scientific_outcome='finite_feasibility_demonstrated_for_reported_profiles' if verified else
            ('none_found_in_budget' if stage.get('status') == 'complete' else 'incomplete_or_failed'),
        maximum_direct_model_scaled_error=maximum(r.get('direct_model_scaled_error') for r in verified),
        maximum_actual_gpu_scaled_error=maximum(v for r in verified for v in r.get('actual_both_gpu_errors', {}).values()),
        maximum_observed_D_scaled_error=maximum(r.get('observed_split_difference_scaled_error') for r in verified),
        maximum_canonical_gram_error=maximum(r.get('certificate', {}).get('metrics', {}).get('normalized_full_gram_error') for r in verified),
        maximum_canonical_D_scaled_error=maximum(r.get('certificate', {}).get('metrics', {}).get('full_split_difference_scaled_error') for r in verified),
        minimum_verified_harmonic_rms=min((r['certificate']['metrics']['harmonic_coadd_perturbation_ratio'] for r in verified), default=None),
        maximum_verified_harmonic_rms=maximum(r['certificate']['metrics']['harmonic_coadd_perturbation_ratio'] for r in verified),
        minimum_verified_weighted_observed_rms=minimum(r['certificate']['metrics'].get('observed_coadd_perturbation_ratio') for r in verified),
        maximum_verified_weighted_observed_rms=maximum(r['certificate']['metrics'].get('observed_coadd_perturbation_ratio') for r in verified),
        elapsed_seconds=stage.get('elapsed_seconds'), utc_elapsed_seconds=stage.get('utc_elapsed_seconds'),
        budget_elapsed_seconds=stage.get('budget_elapsed_seconds'),
        closure_counterexamples=0, universal_closure_verified=False, physical_noise_model_validated=False)
    return dict(plan=plan, stage=stage, result=result, entries=entries, summary=summary)


def plot(run, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan, entries = run['plan'], run['entries']
    profiles = list(dict.fromkeys(e['expected']['profile'] for e in entries))
    tolerances = plan['tolerances']
    table = np.zeros((len(profiles), len(tolerances)))
    lookup = {(e['expected']['profile'], e['expected']['tolerance']): e for e in entries}
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, name in enumerate(profiles):
        for j, tolerance in enumerate(tolerances):
            entry = lookup[(name, tolerance)]
            table[i, j] = entry['verified']
    image = ax.imshow(table, vmin=0, vmax=plan['starts_per_case'], cmap='Blues', aspect='auto')
    for i, name in enumerate(profiles):
        for j, tolerance in enumerate(tolerances):
            e = lookup[(name, tolerance)]
            ax.text(j, i, f"{e['verified']}/{e['started']}\n{e['failed']} failed; {e['incomplete']} unfinished",
                    ha='center', va='center', color='white' if table[i, j] >= 3 else '#25364a', fontsize=9)
    ax.set(xticks=range(len(tolerances)), xticklabels=[f'{x:.0e}' for x in tolerances],
           yticks=range(len(profiles)), yticklabels=profiles, xlabel='Prospectively fixed scaled lock tolerance',
           title='Cut-sky H1 finite feasibility\nVerified / started (four planned per cell)')
    fig.colorbar(image, ax=ax, label='Independently verified candidates', shrink=.8)
    fig.savefig(out/'coverage.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = plt.get_cmap('tab10')
    labels = set()
    for entry in entries:
        tolerance = entry['expected']['tolerance']
        for item in entry['rows']:
            row = item['row']; verified = row.get('status') == VERIFIED
            if verified:
                metrics = row['certificate']['metrics']
                rms, residual = metrics['harmonic_coadd_perturbation_ratio'], metrics['realized_feature_lock_error']
            else:
                rms = row.get('solver', {}).get('rms_ratio')
                residual = row.get('solver', {}).get('max_scaled_residual')
            if number(rms) is None or number(residual) is None:
                continue
            label = f"{tolerance:.0e} · {'direct verified' if verified else 'solver only / unverified'}"
            ax.scatter(rms, max(residual/tolerance, 1e-8), color=colors(tolerances.index(tolerance)),
                       marker='o' if verified else 'x', alpha=.8, label=label if label not in labels else None)
            labels.add(label)
    ax.axhline(1, color='#a33', linestyle='--', label='Declared lock gate')
    ax.axvline(plan['minimum_harmonic_rms'], color='#777', linestyle=':')
    ax.axvline(plan['maximum_rms'], color='#777', linestyle=':')
    ax.set(yscale='log', xlabel='Retained-harmonic coadd perturbation/reference RMS',
           ylabel='Saved realized residual / declared tolerance',
           title='Nontrivial amplitude and numerical locks\nSolver-only points are not independently qualified')
    ax.legend(fontsize=8)
    fig.savefig(out/'rms_residual.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=ROOT/'docs/CUTSKY_FEASIBILITY_RUN.md')
    args = parser.parse_args()
    if not args.run_dir.name.startswith('cutsky_feasibility_'):
        raise ValueError('only separate cut-sky output directories are supported')
    run = collect_run(args.run_dir)
    if run['stage'].get('status') == 'running':
        raise RuntimeError('do not publish a final report while scientific work is running')
    plot(run, args.run_dir)
    summary, plan, stage = run['summary'], run['plan'], run['stage']
    (args.run_dir/'report_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    import os
    link = lambda path: Path(os.path.relpath(path, args.report.parent)).as_posix()
    case_rows, points = [], []
    for e in run['entries']:
        x = e['expected']; p = e['profile_record'] or {}
        case_rows.append(f"| {x['profile']} | {x['tolerance']:.0e} | {e['started']}/4 | {e['verified']} | {e['no_nontrivial_found']} | {e['failed']} | {e['incomplete']} | {fmt(p.get('positive_pixel_fraction'))} |")
        for item in e['rows']:
            r = item['row']; c = r.get('certificate', {}); m = c.get('metrics', {})
            gpu_values = [v for v in r.get('actual_both_gpu_errors', {}).values() if number(v) is not None]
            points.append(f"| {x['name']} / {r['start_id']} | {r.get('status')} | {fmt(m.get('harmonic_coadd_perturbation_ratio',r.get('solver',{}).get('rms_ratio')))} | {fmt(m.get('realized_feature_lock_error',r.get('solver',{}).get('max_scaled_residual')))} | {fmt(r.get('direct_model_scaled_error'))} | {fmt(max(gpu_values,default=None))} |")
    tolerance_rows = '\n'.join(f"| {key} | {value['started']}/{value['planned']} | {value['verified']} | {value['no_nontrivial_found']} | {value['failed']} | {value['incomplete']} |" for key, value in summary['tolerance_outcomes'].items())
    text = f'''# Cut-sky H1 feasibility pilot

Execution: **{summary['execution_status']}**. Recorded starts:
**{summary['starts_started']}/{summary['planned_starts']}** across
{summary['closed_cases']}/{summary['planned_cases']} closed case records.
Independently verified nontrivial candidates: **{summary['verified_candidates']}**.
Finite searches without a nontrivial candidate: {summary['no_nontrivial_found']}.
Numerical/verification failures: {summary['numerical_or_verification_failures']};
unfinished started rows: {summary['incomplete_started']}; unstarted: {summary['unstarted']}.
Scientific outcome: `{summary['scientific_outcome']}`. A completed unsuccessful
finite search is inconclusive about global feasibility, and is never a closure pass.

![All profiles and tolerances]({link(args.run_dir/'coverage.png')})

## Scope and actual observing operator

This is H1 finite-family feasibility on synthetic split maps, using frozen
NaMaster {plan['versions'].get('pymaster','unknown')}, ducc and NumPy/CUDA checks.
It tests a common scalar apodized mask, common declared response, COSMO Q/U and
RING ordering. The retained observables are **coupled, binned, realized
pseudo-spectra** for AA, BB and AB, each with EE/EB/BE/BB channels and band edges
{plan['band_edges']}. They are neither all unbinned spectra nor decoupled
bandpowers, and no ensemble coupling matrix certifies an individual candidate.
The 48 recorded feature values need not be independent constraints. Saved
solver Jacobian ranks range from {fmt(summary['minimum_solver_rank'])} to
{fmt(summary['maximum_solver_rank'])} out of {plan['n_planes']} angle coordinates,
at the fixed numerical rank cutoff {plan['rank_cutoff']}. These sampled numerical
ranks establish neither constant rank over a neighborhood nor a global
100-dimensional feasible manifold.
Purification: E={plan['purify_e']}, B={plan['purify_b']}; field and mask band limits
are {plan['lmax']} and {plan['mask_lmax']} for this supported NaMaster profile.
Other mask/purification/response combinations remain unsupported until qualified.

The input is {plan['input_model']}. Actual FITS/HDF files retain the original
pixel observations. Their unmodeled residual is carried unchanged when a common
synthesized coadd increment is added to both splits. The full spectral Gram
constraint concerns the **canonical retained harmonics**, not an exact inverse
of all pixel data. The empirical A−B difference is checked on those full retained
coefficients and separately on the original observed-map intervention. It is not
a new noise realization or a validated probability law for counterfactual skies.
The full pilot uses **one synthetic input realization and six toy mask profiles**.
It does not establish performance across independent skies, realistic survey
masks, foregrounds, instrumental systematics or calibrated noise ensembles.

## All planned cases

Nominal cap fraction differs from positive pixel fraction and apodized weight.
Each tolerance uses the same four direction identities for a given profile.
There are {summary['planned_direction_streams']} planned direction streams,
of which {summary['direction_streams_started']} started, reused across the two
tolerances to create {summary['planned_starts']} planned starts. The
{summary['verified_tolerance_pairs']} verified matched pairs are repeated
tolerance checks, rather than independent random-search replications. Their
stricter/looser harmonic RMS ratios range from
{fmt(summary['minimum_paired_harmonic_rms_ratio'])} to
{fmt(summary['maximum_paired_harmonic_rms_ratio'])}.
The {plan['n_planes']} numerical planes and all parameters remain stored; a
verified finite point does not certify a globally smooth feasible manifold.

| Profile | Tolerance | Started | Verified | None found | Failed | Unfinished | Positive pixel fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(case_rows)}

| Tolerance | Started / planned | Verified | None found | Failed | Unfinished |
|---:|---:|---:|---:|---:|---:|
{tolerance_rows}

## Amplitude, residual and independent verification

The minimum retained-harmonic perturbation/reference RMS is
{plan['minimum_harmonic_rms']}; the maximum harmonic and weighted observed-map
ratio is {plan['maximum_rms']}. Angles are bounded by ±π/4. The original 1e-10
profile and separately declared 1e-12 profile are both reported, without relaxing
either after failures. Every verified row uses reconstructed observed maps and
actual NaMaster fields, and actual evaluation on both GPUs.
Verified harmonic RMS ratios range from
{fmt(summary['minimum_verified_harmonic_rms'])} to
{fmt(summary['maximum_verified_harmonic_rms'])}; weighted observed-map ratios
range from {fmt(summary['minimum_verified_weighted_observed_rms'])} to
{fmt(summary['maximum_verified_weighted_observed_rms'])}. The 1% minimum applies
to the retained harmonics; observed-map amplitude has only the declared upper
bound, and may be below 1%.

![Saved amplitude and residuals]({link(args.run_dir/'rms_residual.png')})

The plot shows direct feature residuals for verified points and labels solver-only
points separately. Zero residuals are displayed at 1e-8 of the declared tolerance
for log-axis plotting. Maximum saved direct/model disagreement:
{fmt(summary['maximum_direct_model_scaled_error'])}; maximum actual GPU/direct
disagreement: {fmt(summary['maximum_actual_gpu_scaled_error'])}. Maximum canonical
Gram error: {fmt(summary['maximum_canonical_gram_error'])}; canonical D error:
{fmt(summary['maximum_canonical_D_scaled_error'])}; observed-map D error:
{fmt(summary['maximum_observed_D_scaled_error'])}.

| Case / start | Outcome | Harmonic RMS ratio | Realized residual (solver-only if uncertified) | Direct/model error | Both-GPU maximum error |
|---|---|---:|---:|---:|---:|
{chr(10).join(points) or '| — | No recorded starts | — | — | — | — |'}

Solver reasons: {summary['solver_reason_counts']}. These are bounded continuation
outcomes, not global convergence or infeasibility proofs. Recorded execution
failure: `{(run['result'] or {}).get('failure', read(args.run_dir/'failure.json'))}`.
Partial candidate archives and failed audits remain preserved.

## Budget, provenance and interpretation

The {plan['compute_cap_seconds']}-second scientific cap includes preprocessing,
search and verification, with a {plan['verification_reserve_seconds']}-second
verification reserve. Monotonic elapsed: {fmt(summary['elapsed_seconds'])} s;
UTC elapsed: {fmt(summary['utc_elapsed_seconds'])} s; conservative budget clock:
{fmt(summary['budget_elapsed_seconds'])} s. Both clocks are retained. Prior frozen
analyses are bound by the new plan's preservation manifest.

[Frozen plan]({link(args.run_dir/'plan.json')}) ·
[Saved result]({link(args.run_dir/'result.json')}) ·
[Input provenance]({link(args.run_dir/'inputs/provenance.json')}) ·
[Metadata and live JoVin authentication]({link(args.run_dir/'verification.json')})

SHA-256 checks byte identity against bound references. Trust and runtime
authentication depend on separate provenance and receipt verification; neither
establishes correct physics. The 24 existing supporting Lean declarations and
prior studies remain unchanged. This pilot supplies **no future-feature closure
counterexample, statistical sufficiency result, physical-noise validation,
cosmological parameter inference or detector-superiority claim**. Any next phase
requires a separately frozen physical and statistical validation protocol.
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
