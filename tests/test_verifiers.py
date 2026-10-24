"""Regression checks for executable provenance and independent evidence checks."""
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def verifiers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return (importlib.import_module('verify_lean'),
            importlib.import_module('verify_multiscale_evidence'))


@pytest.fixture
def lean_run(verifiers,monkeypatch,tmp_path):
    verifier,_ = verifiers
    compiler = tmp_path/'lean'
    compiler.write_bytes(b'archive compiler')
    archive = tmp_path/'lean.tar.zst'
    archive.write_bytes(b'archive fixture')
    output = tmp_path/'evidence.json'
    monkeypatch.setattr(verifier,'RELEASE_SHA256',verifier.sha(archive))
    monkeypatch.setattr(sys,'argv',['verify_lean.py','--lean',str(compiler),
        '--archive',str(archive),'--output',str(output)])
    source = (ROOT/'formal/ScaleInvariance.lean').read_text()
    names = re.findall(r'^theorem\s+(\w+)',source,re.MULTILINE)
    state = SimpleNamespace(compiler=compiler,archive=archive,output=output,names=names,
        calls=[],stdout='\n'.join(f"'CmbCst.{name}' depends on axioms: "
            '[propext, Classical.choice, Quot.sound]' for name in names),stderr='',returncode=0)

    def check_output(command,**kwargs):
        state.calls.append(command)
        if command[0]=='tar':
            assert command==['tar','--zstd','-xOf',str(archive),'lean-4.35.0-linux/bin/lean']
            return b'archive compiler'
        assert command==[str(compiler.resolve()),'--version']
        return 'Lean (version 4.35.0, commit eb6cafb7290e8bd5f2937522d8d804f58f02bafc)'

    def compile_proofs(command,**kwargs):
        state.calls.append(command)
        assert command==[str(compiler.resolve()),'--stdin']
        assert kwargs['input'].startswith(source+'\n')
        for name in names:
            assert f'#print axioms CmbCst.{name}\n' in kwargs['input']
        return state

    monkeypatch.setattr(verifier.subprocess,'check_output',check_output)
    monkeypatch.setattr(verifier.subprocess,'run',compile_proofs)
    state.run = verifier.main
    return state


def test_lean_rejects_substitute_before_version_check(lean_run):
    lean_run.compiler.write_bytes(b'different compiler claiming the same version')
    with pytest.raises(ValueError,match='binary in the pinned archive'):
        lean_run.run()
    assert len(lean_run.calls)==1 and lean_run.calls[0][0]=='tar'
    assert not lean_run.output.exists()


def test_lean_rejects_wrong_archive_before_executing(lean_run):
    lean_run.archive.write_bytes(b'changed archive')
    with pytest.raises(ValueError,match='archive SHA256'):
        lean_run.run()
    assert not lean_run.calls


@pytest.mark.parametrize('axiom_free',[False,True])
def test_lean_records_all_nine_axiom_sets(lean_run,axiom_free):
    if axiom_free:
        lean_run.stdout='\n'.join(f"'CmbCst.{name}' does not depend on any axioms"
            for name in lean_run.names)
    lean_run.run()
    evidence=json.loads(lean_run.output.read_text())
    assert evidence['status']=='passed'
    assert evidence['compiler_sha256']==hashlib.sha256(b'archive compiler').hexdigest()
    assert set(evidence['theorem_axioms'])==set(lean_run.names)
    assert len(evidence['theorem_axioms'])==9


@pytest.mark.parametrize('index',range(9))
@pytest.mark.parametrize('failure',['missing','forged'])
def test_lean_requires_allowlisted_output_for_each_theorem(lean_run,index,failure):
    lines=lean_run.stdout.splitlines()
    if failure=='missing':
        del lines[index]
    else:
        lines[index]=lines[index].replace('Quot.sound','Quot.sound, forged')
        # A later allowed report must not hide a forbidden dependency.
        lines.append(lean_run.stdout.splitlines()[index])
    lean_run.stdout='\n'.join(lines)
    with pytest.raises(SystemExit) as exc:
        lean_run.run()
    assert exc.value.code==1
    assert json.loads(lean_run.output.read_text())['status']=='failed'


@pytest.mark.parametrize('failure',['exit_code','sorry_stdout','sorry_stderr'])
def test_lean_retains_exit_code_and_sorry_checks(lean_run,failure):
    if failure=='exit_code':
        lean_run.returncode=1
    elif failure=='sorry_stdout':
        lean_run.stdout+='\nsorryAx\n'
    else:
        lean_run.stderr='sorryAx'
    with pytest.raises(SystemExit):
        lean_run.run()
    assert json.loads(lean_run.output.read_text())['status']=='failed'


@pytest.fixture
def multiscale_records():
    out=ROOT/'results/multiscale_v1'
    result,positive,ensemble=[json.loads((out/name).read_text()) for name in
        ('result.json','positive_control.json','ensemble.json')]
    return result,positive,ensemble['features'][:999]


def test_stored_multiscale_comparisons_pass(verifiers,multiscale_records):
    _,verifier=verifiers
    checks=verifier.comparison_checks(*multiscale_records)
    assert set(checks)=={'all_sky_pvalues','no_threshold_flags','observed_compression','positive_control'}
    assert all(checks.values())


def test_failed_calibration_has_no_observed_comparisons(verifiers,multiscale_records):
    _,verifier=verifiers
    _,positive,reference=multiscale_records

    class NoObservations(dict):
        def __getitem__(self,key):
            assert key not in {'observed','observed_curves','raw_p','p_holm'}
            return super().__getitem__(key)

    result=NoObservations(status='calibration_failed',candidate_flags=[])
    assert verifier.comparison_checks(result,positive,reference)=={
        'no_observed_comparisons':True,'positive_control':True}


@pytest.mark.parametrize('key',['observed','observed_curves','raw_p','p_holm',
    'observed_quality','reference_mean','reference_sd','candidate_flags'])
def test_failed_calibration_rejects_observed_fields(verifiers,multiscale_records,key):
    _,verifier=verifiers
    _,positive,reference=multiscale_records
    result={'status':'calibration_failed','candidate_flags':[]}
    result[key]=[1]
    assert not verifier.comparison_checks(result,positive,reference)['no_observed_comparisons']


def test_positive_control_rejects_inconsistent_curves(verifiers,multiscale_records):
    _,verifier=verifiers
    result,positive,reference=multiscale_records
    positive['curves'][0][0][0]*=2
    assert not verifier.comparison_checks(result,positive,reference)['positive_control']


def test_positive_control_rejects_forged_features_with_consistent_pvalues(verifiers,multiscale_records):
    _,verifier=verifiers
    result,positive,reference=multiscale_records
    positive['features'][0][0]+=1
    raw,adjusted=verifier.probabilities(positive['features'],reference)
    positive['raw_p']=[raw[i:i+6] for i in range(0,18,6)]
    positive['p_holm']=[adjusted[i:i+6] for i in range(0,18,6)]
    assert not verifier.comparison_checks(result,positive,reference)['positive_control']
