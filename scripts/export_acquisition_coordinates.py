"""Preserve the frozen raw export and write a sign-corrected coordinate view."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cmbcst.acquisition import instrument_design,detector_table
from cmbcst.acquisition_index import look_coordinates,analytic_phase_error
OUT = ROOT/'results/acquisition_v1'
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    target = OUT/'labeled_samples_corrected.csv'
    if target.exists():
        raise FileExistsError('preserve corrected view')
    source = OUT/'labeled_samples.csv'
    with source.open(newline='') as stream:
        original = list(csv.DictReader(stream))
    table = detector_table()
    rows = []
    for old in original:
        row = dict(old)
        row['registration_plus_offset_x_deg'] = row.pop('pointing_x_deg')
        row['registration_plus_offset_y_deg'] = row.pop('pointing_y_deg')
        h = int(row['channel'])
        u,w = look_coordinates(int(row['pixel_x']),int(row['pixel_y']),table['offset_x_pixels'][h],table['offset_y_pixels'][h])
        for axis,j in (('x',0),('y',1)):
            row[f'look_{axis}_unwrapped_deg'] = u[j]
            row[f'look_{axis}_wrapped_deg'] = w[j]
        rows.append(row)
    if not rows:
        raise ValueError('source contains no sample rows')
    error,opposite = analytic_phase_error(instrument_design(32)[0],table)
    if error>1e-10 or opposite<=.01:
        raise ValueError('independent phase convention check failed')
    with target.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    audit = dict(status='passed',original_sha256=sha(source),corrected_sha256=sha(target),
        source_sha256={name:sha(ROOT/name) for name in ('src/cmbcst/acquisition_index.py','scripts/export_acquisition_coordinates.py')},
        analytic_negative_phase_error=error,incorrect_positive_phase_error=opposite,
        scope='Coordinate-label erratum; frozen simulation, observations, inverse operators and metric files are unchanged')
    (OUT/'coordinate_export_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
