"""Describe saved V2 evidence only: no optimization, transforms or feature calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/closure_fibers_v2'
TIERS = (.010, .015, .025, .050)
CASE_LABELS = (('source', 'baseline'), ('fresh', 'baseline'), ('fresh', 'capacity'), ('fresh', 'repair'))
SUCCESS = ('stalled', 'iteration_cap')
FAILURE = ('nonfinite_reference', 'invalid_reference', 'nonfinite_jacobian',
           'nonfinite_gradient', 'nonfinite_rms_gradient')
CHECKPOINT_PHASES = ('before_initialization', 'initialized', 'periodic', 'complete', 'interrupted')


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def load_npz(path):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def name_of(case):
    return f"tile_{case['tile']}_{case['stage']}_{case['representation']}"


def finite(value):
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def fmt(value, digits=6):
    return '—' if value is None or not np.isfinite(value) else f'{value:.{digits}g}'


def checkpoint_binding(checkpoint, metadata, npz_sha256, count, planes, metadata_error=None):
    """Recover identity from the atomic NPZ, independently of companion JSON.

    The JSON may be absent or belong to the preceding GTX batch when a process
    stops between the two atomic replacements. A matching file hash with a
    contradictory identity is a real inconsistency; a stale hash is explicitly
    recorded as an interrupted companion write and never supplies row IDs.
    """
    info = dict(present=checkpoint is not None, valid=False, self_ids_valid=False,
                start_ids=[], phase=None, phase_source='unavailable', phase_valid=False,
                id_source='unavailable', json_matches_npz=False, pair_recovered=False,
                contradictory_json=False, issues=[], npz_sha256=npz_sha256)
    if checkpoint is None:
        return info
    metadata = metadata if isinstance(metadata, dict) else {}
    theta = np.asarray(checkpoint.get('theta', np.empty(0)))
    rows = theta.shape[0] if theta.ndim == 2 else 0
    if theta.shape != (rows, planes) or rows < 1:
        info['issues'].append('invalid_npz_theta_shape')
        return info
    raw_ids = checkpoint.get('checkpoint_start_ids')
    raw_first = checkpoint.get('checkpoint_first_start')
    if raw_ids is not None or raw_first is not None:
        ids = np.asarray(raw_ids)
        first = np.asarray(raw_first)
        valid = (ids.dtype.kind in 'iu' and ids.shape == (rows,)
                 and first.dtype.kind in 'iu' and first.shape == ()
                 and 0 <= int(first) and int(first)+rows <= count
                 and np.array_equal(ids, np.arange(int(first), int(first)+rows)))
        if not valid:
            info['issues'].append('invalid_npz_self_identifiers')
            return info
        info.update(valid=True, self_ids_valid=True, start_ids=ids.astype(int).tolist(), id_source='npz')
    else:
        # Legacy files can be read only when their JSON is cryptographically
        # bound to this exact NPZ. Never guess a range from unbound JSON.
        first = metadata.get('first_start')
        if (metadata.get('checkpoint_sha256') == npz_sha256 and type(first) is int
                and 0 <= first and first+rows <= count):
            info.update(valid=True, start_ids=list(range(first, first+rows)), id_source='hash_bound_json')
        else:
            info['issues'].append('missing_npz_self_identifiers_and_unbound_json')
            return info
    raw_phase = checkpoint.get('checkpoint_phase')
    if raw_phase is not None:
        phase = np.asarray(raw_phase)
        if phase.shape == () and phase.dtype.kind in 'US':
            info.update(phase=str(phase), phase_source='npz', phase_valid=str(phase) in CHECKPOINT_PHASES)
        if not info['phase_valid']:
            info['issues'].append('invalid_npz_checkpoint_phase')
    hash_match = metadata.get('checkpoint_sha256') == npz_sha256
    if metadata_error:
        info['issues'].append('json_unreadable')
        info['json_read_error'] = metadata_error
    elif not metadata:
        info['issues'].append('json_missing')
    elif not metadata.get('checkpoint_sha256'):
        info['issues'].append('json_npz_hash_missing')
    elif not hash_match:
        info['issues'].append('json_npz_hash_mismatch')
    expected_first = info['start_ids'][0]
    first_match = type(metadata.get('first_start')) is int and metadata['first_start'] == expected_first
    if metadata and not first_match:
        info['issues'].append('json_batch_mismatch')
    if raw_phase is None and hash_match:
        phase = metadata.get('phase', metadata.get('checkpoint_phase'))
        info.update(phase=phase, phase_source='hash_bound_json', phase_valid=phase in CHECKPOINT_PHASES)
    json_phase = metadata.get('phase', metadata.get('checkpoint_phase'))
    phase_match = json_phase == info['phase']
    if metadata and not phase_match:
        info['issues'].append('json_phase_mismatch')
    info['contradictory_json'] = bool(hash_match and (not first_match or not phase_match))
    info['json_matches_npz'] = bool(hash_match and first_match and phase_match and not metadata_error)
    info['pair_recovered'] = bool(info['self_ids_valid'] and not info['json_matches_npz']
                                  and not info['contradictory_json'])
    info['valid'] = bool(info['valid'] and not info['contradictory_json'])
    return info


def _arrays(out, name, count, planes):
    saved = load_npz(out/'cases'/f'{name}.npz')
    a = dict(started=np.zeros(count, dtype=int), status=np.full(count, 'not_started', dtype='<U32'),
             iterations=np.zeros(count, dtype=int), best_theta=np.zeros((count, planes)),
             theta=np.zeros((count, planes)), nonfinite_trials=np.zeros(count, dtype=int))
    if saved:
        a.update({k: np.array(v, copy=True) for k, v in saved.items()})
    checkpoint_path = out/'checkpoints'/f'{name}.npz'
    checkpoint = load_npz(checkpoint_path)
    metadata_error = None
    try:
        meta = read(out/'checkpoints'/f'{name}.json', {})
    except (ValueError, OSError) as exc:
        meta = {}
        metadata_error = f'{type(exc).__name__}: {exc}'
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint is not None else None
    info = checkpoint_binding(checkpoint, meta, digest, count, planes, metadata_error)
    overlaid = []
    if checkpoint is not None and info['valid']:
        ids = np.asarray(info['start_ids'], dtype=int)
        rows = len(ids)
        iterations = checkpoint.get('iterations', np.zeros(rows, dtype=int))
        changed = np.any(a['theta'][ids] != checkpoint['theta'], axis=1)
        newer = ((a['started'][ids] == 0) | (iterations > a['iterations'][ids])
                 | ((iterations == a['iterations'][ids]) & changed))
        for key, value in checkpoint.items():
            if (key in a and np.ndim(value) > 0 and len(value) == rows
                    and np.shape(a[key])[1:] == np.shape(value)[1:]):
                a[key][ids[newer]] = value[newer]
        a['started'][ids[newer]] = 1
        for key in ('gpu_features', 'gpu_target', 'gpu_ratios', 'gpu_all_features'):
            if key in a:
                a[key][ids[newer]] = np.nan
        overlaid = ids[newer].tolist()
    return a, saved, checkpoint, meta, overlaid, info


def collect_run(out=OUT):
    out = Path(out)
    plan = read(out/'plan.json')
    if plan is None:
        raise FileNotFoundError('a frozen V2 plan is required')
    stage = read(out/'stage_audit.json', {'status': 'missing_stage'})
    result = read(out/'result.json')
    aggregate = read(out/'cases.json', {'records': []})
    records = {name_of(c): c for c in aggregate.get('records', [])}
    records.update({name_of(c): c for c in (result or {}).get('cases', [])})
    entries = []
    for case in plan['cases']:
        name = name_of(case)
        record = read(out/'cases'/f'{name}.json', records.get(name, {}))
        arrays, saved, checkpoint, meta, overlay, checkpoint_info = _arrays(out, name, plan['starts_per_case'], plan['planes'])
        cert = read(out/'certificates_v2'/name/'case.json', read(out/'certificates_v2'/name/'worker_failure.json'))
        maximum = (cert or {}).get('maximum')
        if maximum is None and cert and cert.get('maximum_selected_start') is not None:
            row = out/'certificates_v2'/name/'rows'/f"start_{cert['maximum_selected_start']:04d}.json"
            maximum = read(row)
        begun = arrays['started'] > 0
        complete = begun & np.isin(arrays['status'], SUCCESS)
        failed = begun & np.isin(arrays['status'], FAILURE)
        checked = int((cert or {}).get('starts_checked', 0))
        gap = finite((maximum or {}).get('raw_primary_gap'))
        gap_scope = 'independent direct replay of the fixed cell maximum' if gap is not None else 'not directly verified'
        if gap is None:
            gap = finite((cert or {}).get('maximum_discovered_feasible_gap'))
            if gap is not None:
                gap_scope = 'provisional matrix endpoint; direct verification incomplete'
        tier = (cert or {}).get('tier_counts', {f'{t:.3f}': 0 for t in TIERS})
        entries.append(dict(name=name, case=case, record=record, arrays=arrays, saved_arrays=saved,
                            checkpoint=checkpoint, checkpoint_meta=meta, checkpoint_overlay=overlay,
                            checkpoint_info=checkpoint_info,
                            certificate=cert, maximum=maximum, starts_started=int(begun.sum()),
                            starts_completed=int(complete.sum()), terminal_failures=int(failed.sum()),
                            starts_verified=checked, maximum_gap=gap, gap_scope=gap_scope,
                            tier_counts=tier))
    matched = []
    for tile in plan['selected_tiles']:
        triple = [e for e in entries if e['case']['tile'] == tile and e['case']['stage'] == 'fresh']
        if len(triple) == 3 and all(e['starts_started'] == e['starts_completed'] == e['starts_verified'] == plan['starts_per_case']
                                    and e['terminal_failures'] == 0 and e['certificate']
                                    and e['certificate']['status'] == 'complete'
                                    and e['certificate'].get('verification_failures', 0) == 0 for e in triple):
            matched.append(int(tile))
    tiers = {f'{t:.3f}': sum(int(e['tier_counts'].get(f'{t:.3f}', 0)) for e in entries) for t in TIERS}
    row_tiers = {f'{t:.3f}': sum(int((e['certificate'] or {}).get('row_cpu_direct_raw_tier_counts', {}).get(f'{t:.3f}', 0))
                                for e in entries) for t in TIERS}
    status_counts = {}
    directions = {label: dict(started=0, completed=0) for label in ('maximize', 'minimize')}
    for e in entries:
        a = e['arrays']
        for state, n in zip(*np.unique(a['status'], return_counts=True)):
            status_counts[str(state)] = status_counts.get(str(state), 0)+int(n)
        if 'signs' in a:
            for label, direction in (('maximize', a['signs'] > 0), ('minimize', a['signs'] < 0)):
                directions[label]['started'] += int(np.sum((a['started'] > 0) & direction))
                directions[label]['completed'] += int(np.sum((a['started'] > 0) & direction & np.isin(a['status'], SUCCESS)))
    summary = dict(scientific_status=stage.get('status'), scientific_outcome=(result or {}).get('scientific_outcome', 'not_finalized'),
                   planned_cases=len(plan['cases']), planned_starts=plan['planned_starts'],
                   starts_started=sum(e['starts_started'] for e in entries), starts_completed=sum(e['starts_completed'] for e in entries),
                   starts_verified=sum(e['starts_verified'] for e in entries),
                   terminal_failure_starts=sum(e['terminal_failures'] for e in entries),
                   verification_failures=sum(sum(r.get('status') in ('precision_failure', 'verification_failure')
                                                 for r in (e['certificate'] or {}).get('rows', []))
                                             + int((e['certificate'] or {}).get('status') in ('execution_failed', 'verification_failed'))
                                             for e in entries),
                   tier_counts=tiers, row_cpu_direct_raw_tier_counts=row_tiers, matched_complete_tiles=matched,
                   status_counts=status_counts, directions=directions,
                   elapsed_seconds=finite(stage.get('elapsed_seconds')), utc_elapsed_seconds=finite(stage.get('utc_elapsed_seconds')),
                   budget_elapsed_seconds=finite(stage.get('budget_elapsed_seconds')),
                   rms_ratio_names=['tile', 'global'])
    summary['checkpoint_pair_recoveries'] = [dict(case=e['name'], **e['checkpoint_info']) for e in entries
                                             if e['checkpoint_info']['pair_recovered']]
    summary['unverified_started_rows'] = summary['starts_started']-summary['starts_verified']
    return dict(plan=plan, stage=stage, result=result, entries=entries, summary=summary)


def plots(run, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    tiles, entries, summary = run['plan']['selected_tiles'], run['entries'], run['summary']
    lookup = {(e['case']['tile'], e['case']['stage'], e['case']['representation']): e for e in entries}
    gaps = np.full((len(tiles), 4), np.nan)
    completion = np.zeros_like(gaps)
    labels = ['Source\nbaseline', 'Fresh\nbaseline', 'Fresh\npower control', 'Fresh\nCST repair']
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    for i, tile in enumerate(tiles):
        for j, (phase, rep) in enumerate(CASE_LABELS):
            e = lookup[(tile, phase, rep)]
            completion[i, j] = e['starts_completed']/run['plan']['starts_per_case']
            if e['maximum_gap'] is not None:
                gaps[i, j] = e['maximum_gap']
    image = axes[0].imshow(completion, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    for i, tile in enumerate(tiles):
        for j, (phase, rep) in enumerate(CASE_LABELS):
            e = lookup[(tile, phase, rep)]
            axes[0].text(j, i, f"{e['starts_started']}/{e['starts_completed']}/{e['starts_verified']}",
                         ha='center', va='center', fontsize=8, color='white' if completion[i, j] > .7 else '#24384a')
    axes[0].set_title('Started / completed / CPU-direct checked\n128 planned per cell')
    color = plt.get_cmap('viridis').copy()
    color.set_bad('#e5e7eb')
    g = axes[1].imshow(gaps, vmin=0, vmax=max(.05, np.nanmax(gaps) if np.isfinite(gaps).any() else .05), cmap=color, aspect='auto')
    for i in range(len(tiles)):
        for j in range(4):
            axes[1].text(j, i, '—' if not np.isfinite(gaps[i, j]) else f'{gaps[i,j]:.4f}', ha='center', va='center', fontsize=8, color='white')
    axes[1].set_title('Fixed maximum raw K4 gap\nCertification and failures remain in the table')
    for ax in axes:
        ax.set(xticks=range(4), xticklabels=labels, yticks=range(len(tiles)), yticklabels=tiles, ylabel='Fixed tile')
    fig.colorbar(image, ax=axes[0], label='Completed fraction', shrink=.7)
    fig.colorbar(g, ax=axes[1], label='Raw normalized K4 difference', shrink=.7)
    fig.suptitle(f"CST V2: {summary['scientific_status']} | finite discovered gaps are lower bounds", fontsize=15)
    fig.savefig(out/'coverage.png', dpi=170)
    plt.close(fig)

    table = np.array([[int(e['tier_counts'].get(f'{t:.3f}', 0)) for t in TIERS] for e in entries])
    fig, ax = plt.subplots(figsize=(8, max(10, len(entries)*.27)), constrained_layout=True)
    ax.imshow(table, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    ax.set(xticks=range(4), xticklabels=[f'≥ {t:.3f}' for t in TIERS],
           yticks=range(len(entries)), yticklabels=[f"{e['case']['tile']} · {e['case']['stage']} {e['case']['representation']}" for e in entries],
           xlabel='Raw K4 threshold; each cell maximum counted at most once')
    ax.set_title('Verified cell-maximum witnesses\nDirect replay + both GPUs + safe changed future bin', fontsize=12)
    for i, e in enumerate(entries):
        for j in range(4):
            missing = not e['certificate'] or e['certificate']['status'] != 'complete'
            ax.text(j, i, '?' if missing else ('yes' if table[i, j] else '—'), ha='center', va='center',
                    fontsize=8, color='white' if table[i, j] and not missing else '#24384a')
    fig.savefig(out/'tier_heatmap.png', dpi=160)
    plt.close(fig)

    telemetry = read(out/'telemetry.json', {'samples': []})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    metrics = [('utilization', 'GPU activity (%)'), ('memory_MiB', 'Device memory (MiB)'),
               ('power_W', 'Power (W)'), ('temperature_C', 'Temperature (°C)')]
    hardware = {}
    for device in (0, 1):
        data = [(s.get('seconds', 0), d) for s in telemetry.get('samples', [])
                for d in s.get('devices', []) if d.get('index') == device]
        hardware[f'gpu{device}'] = {}
        for ax, (key, title) in zip(axes.ravel(), metrics):
            pairs = [(float(t)/60, float(d[key])) for t, d in data if finite(d.get(key)) is not None]
            if pairs:
                x, y = np.asarray(pairs).T
                ax.plot(x, y, label=f'GPU {device}', linewidth=1)
                hardware[f'gpu{device}'][key] = dict(samples=len(y), mean=float(y.mean()), maximum=float(y.max()))
            ax.set(xlabel='Recorded run minutes', ylabel=title)
    for ax in axes.ravel():
        if ax.lines:
            ax.legend()
        else:
            ax.text(.5, .5, 'No saved telemetry', transform=ax.transAxes, ha='center')
    fig.suptitle('Device-wide counters; other applications may contribute. Activity does not measure FLOPS saturation.')
    fig.savefig(out/'hardware.png', dpi=170)
    plt.close(fig)
    return hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=ROOT/'docs/CLOSURE_FIBERS_V2_RUN.md')
    args = parser.parse_args()
    out = args.run_dir
    if out.resolve() == (ROOT/'results/closure_fibers_v1').resolve():
        raise ValueError('V1 outputs are immutable')
    run = collect_run(out)
    hardware = plots(run, out)
    summary, plan, stage = run['summary'], run['plan'], run['stage']
    (out/'report_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    def link(path):
        import os
        return Path(os.path.relpath(path, args.report.parent)).as_posix()
    rows, source, failures, maxima = [], [], [], []
    for e in run['entries']:
        c, m = e['certificate'] or {}, e['maximum'] or {}
        case = e['case']
        rows.append(f"| {case['tile']} | {case['stage']} {case['representation']} | {e['starts_started']}/{e['starts_completed']}/{e['starts_verified']} | {e['terminal_failures']} | {fmt(e['maximum_gap'])} | {c.get('status','not verified')} / {m.get('status','—')} |")
        if c.get('maximum_selected_start') is not None:
            errors = [x for x in [*m.get('errors', {}).values(), *m.get('actual_both_gpu_errors', {}).values()] if finite(x) is not None]
            maxima.append(f"| {case['tile']} {case['stage']} {case['representation']} | {c['maximum_selected_start']} | {fmt(m.get('raw_primary_gap'))} | {fmt(m.get('input_bins',{}).get('minimum_margin'))} | {', '.join(fmt(x) for x in m.get('perturbation_ratios', [])) or '—'} | {fmt(max(errors) if errors else None)} | {m.get('status','incomplete')} |")
        if case['stage'] == 'source' and c.get('reported_cell_maximum_no_factor_witness'):
            d = m.get('distinguished_by', {})
            source.append(f"| {case['tile']} | {fmt(m.get('raw_primary_gap'))} | {d.get('capacity','unknown')} | {d.get('repair','unknown')} |")
        for r in c.get('rows', []):
            if r.get('status') in ('precision_failure', 'verification_failure', 'incomplete'):
                failures.append(f"| {e['name']} / {r['start_id']} | {r['status']} | [{r.get('audit_path','row audit')}]({link(out/r['audit_path'])}) |" if r.get('audit_path') else f"| {e['name']} / {r['start_id']} | {r['status']} | Per-row JSON retained |")
        errors = e['record'].get('errors', [])
        if e['record'].get('error'):
            errors = [*errors, e['record']['error']]
        for error in errors:
            failures.append(f"| {e['name']} | {error.get('type','execution interruption')} | Saved case JSON and checkpoint |")
        if c.get('error'):
            failures.append(f"| {e['name']} verifier | {c.get('status','verification failure')} | Saved worker_failure.json |")
    tier_rows = '\n'.join(f"| {t:.3f} | {summary['tier_counts'][f'{t:.3f}']} | {summary['row_cpu_direct_raw_tier_counts'][f'{t:.3f}']} |" for t in TIERS)
    proof = read(ROOT/'data/evidence/closure_lean.json', {})
    fresh_rows = []
    for tile in plan['selected_tiles']:
        group = {e['case']['representation']: e for e in run['entries'] if e['case']['tile'] == tile and e['case']['stage'] == 'fresh'}
        fresh_rows.append(f"| {tile} | {'yes' if tile in summary['matched_complete_tiles'] else 'incomplete / unequal'} | "
                          + ' | '.join(fmt(group[rep]['maximum_gap']) for rep in ('baseline','capacity','repair'))+' |')
    recovered_rows = []
    for e in run['entries']:
        info = e['checkpoint_info']
        if info['present'] and info['issues']:
            ids = info['start_ids']
            recovered_rows.append(f"| {e['name']} | {str(ids[0])+'–'+str(ids[-1]) if ids else 'unavailable'} | {info['id_source']} | {info['phase'] or 'unavailable'} | {', '.join(info['issues'])} |")
    report = f'''# CST smoothing fibers V2: saved evidence

Scientific status: **{summary['scientific_status']}**. Completed starts:
**{summary['starts_completed']:,}/{summary['planned_starts']:,}**; started:
{summary['starts_started']:,}; independently CPU/direct checked:
{summary['starts_verified']:,}. Started endpoints still unverified:
{summary['unverified_started_rows']:,}. Terminal numerical starts:
{summary['terminal_failure_starts']}; recorded verification failures:
{summary['verification_failures']}. These are finite counterfactual searches,
anchored to real Planck 143 GHz splits, not new observed skies or cosmological detections.

![Coverage and discovered raw gaps]({link(out/'coverage.png')})

## Meaning of the thresholds

The primary response is the dimensionless normalized connected fourth-order K4
at 2 degrees after one common additional Gaussian smoothing step. Each tier below
is a **raw K4 difference**, not a multiple of a robust-scale bin width. Current
local feature bins have width 0.001 robust-scale units and a minimum safety margin
of 1e-5 robust-scale units. Global normalized spectral bins have width 1e-6.
Direct sampling epsilon is 1e-12 and the scaled numerical agreement gate is 1e-10.

| Raw K4 gap at least | Reported cell-maximum witnesses (≤48) | Separately qualified CPU/direct/search-GPU rows |
|---:|---:|---:|
{tier_rows}

A reported cell-maximum witness requires independent full-harmonic replay,
matching safe current bins, all invariants and bounds, actual all-feature checks
on both GPUs at u=0,1,2,3, and a safely changed future representation bin. The
separate per-row count uses independent CPU/direct and its archived assigned
search-GPU outputs; it does not imply both GPUs evaluated every row. Each started
start's best saved endpoint is replayed even when its gap is below every tier.
The cell maximum is fixed before numerical rejection; a failed maximum is never
replaced by a passing runner-up. All discovered maxima are **lower bounds from
a finite search**, not worst-case upper bounds, physical ceilings or proofs of closure.

![Four raw thresholds for all planned cells]({link(out/'tier_heatmap.png')})

## All 48 planned cases

Counts are started/completed/CPU-direct checked. Missing cells remain explicit.
Maximum gaps without finished certification are provisional; saved precision
failures are retained instead of being counted as witnesses.

| Tile | Search | Starts S/C/V | Terminal failures | Fixed maximum raw gap | Verification / maximum status |
|---:|---|---:|---:|---:|---|
{chr(10).join(rows)}

## Numerical margins and RMS bounds

RMS columns have the explicit order **tile, global**. Both ratios must be ≤0.25;
angles must stay within ±π/4. The margin is in robust-scale units, whereas the gap
is raw normalized K4. Detailed direct/matrix/CPU/GPU errors, all four-time
spectral and fixed-split-difference checks, parameter vectors and SHA-256 bindings
are retained in [the row/cell certificates]({link(out/'certificates_v2')}).

| Case | Fixed maximum start | Raw gap | Current minimum margin | RMS ratios: tile, global | Maximum scaled error | Status |
|---|---:|---:|---:|---|---:|---|
{chr(10).join(maxima) or '| — | — | — | — | — | — | No selected endpoint |'}

## Does the added information distinguish source witnesses?

Only reported source-baseline witnesses enter this table. Distinguishing one
source pair supports that specific augmentation; it does not establish universal
sufficiency. Fresh baseline, capacity-control and CST-repair searches use common
initializations. Complete, equal-budget, independently audited fresh triples:
**{summary['matched_complete_tiles']}**. Unequal or incomplete triples remain in
the coverage table and are not promoted to completed comparisons.

| Source tile | Raw future gap | Ordinary-power response distinguishes | Connected-fourth response distinguishes |
|---:|---:|---|---|
{chr(10).join(source) or '| — | — | No reported source witness | No reported source witness |'}

| Tile | Complete matched triple | Fresh baseline raw maximum | Fresh power-control raw maximum | Fresh CST-repair raw maximum |
|---:|---|---:|---:|---:|
{chr(10).join(fresh_rows)}

## Preserved failures and incomplete verification

| Case / start | Status or exception | Evidence |
|---|---|---|
{chr(10).join(failures) or '| — | No recorded failure | All missing cells remain visible above |'}

Checkpoint arrays carry their own global start IDs and phase. If execution stops
between replacing the NPZ and its companion JSON, recovery uses the NPZ IDs;
stale or absent JSON is diagnosed explicitly and never shifts a second GTX batch
onto starts 0–63. A recovered angle has no newly measured GPU endpoint or CPU
certificate; missing measurements remain unverified.

| Case | NPZ global start IDs | Identity source | NPZ checkpoint phase | Companion-write diagnosis |
|---|---|---|---|---|
{chr(10).join(recovered_rows) or '| — | — | — | — | No interrupted checkpoint pair recorded |'}

## Hardware and the bounded run

![Recorded hardware counters]({link(out/'hardware.png')})

The production cap is {plan.get('compute_cap_seconds',plan.get('cap_seconds',14400))}
seconds, including preprocessing, optimization and independent replay; the final
{plan.get('verification_reserve_seconds',900)} seconds are reserved for finishing
verification. One eight-thread CPU replay worker overlaps GPU search. Recorded
monotonic elapsed: {fmt(summary['elapsed_seconds'])} s; UTC elapsed:
{fmt(summary['utc_elapsed_seconds'])} s; budget clock (maximum of both):
{fmt(summary['budget_elapsed_seconds'])} s. The saved UTC interval runs from
`{stage.get('started_utc','not recorded')}` to `{stage.get('completed_utc','not recorded')}`.
Both clocks and any discrepancy are preserved. GPU counters are device-wide and
can include other applications; activity does not establish FLOPS saturation or
CPU/GPU speedup. Post-cap reporting and verification use only saved values and
hashes, with no new science or candidate promotion.

## What is proved, measured, inferred and unknown

- **Proved:** the conditional collision/no-factor-map implication and an exact
  rational nonclosure toy, kernel-checked in pinned Lean 4.35.0. Saved Lean status:
  `{proof.get('status','unavailable')}`, theorem count {proof.get('theorem_count','unknown')}.
  Existing rotation identities are reused. These proofs do not certify floating-point sky processing.
- **Measured:** only the saved, explicitly qualified counterfactual endpoints,
  numerical errors, margins, bounds, coverage and device counters.
- **Inferred:** a verified matching-current/different-future pair rules out an
  autonomous factor map for this declared finite-precision representation on a
  domain containing that pair.
- **Unknown:** exact continuous-summary closure, universal repair, physical
  origin, primordial structure, fractal dimension and general superiority.

Provenance is in [the frozen plan]({link(out/'plan.json')}),
[result]({link(out/'result.json')}), [report summary]({link(out/'report_summary.json')})
and [metadata/receipt verification]({link(out/'verification.json')}). The frozen
plan binds source, input data, numerical planes, initialization, protocol and
current JoVin receipts; verification binds the final report and plot bytes.
V1 outputs and its protocol are preserved byte for byte. The supplied CST paper
is referenced under its existing package provenance and is not redistributed.
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding='utf-8')
    (out/'hardware_summary.json').write_text(json.dumps(hardware, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
