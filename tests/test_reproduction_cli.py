"""Exercise exact CLI/stage source without loading full-sky transform dependencies."""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    for name in ("CMB_CST_WORK_DIR", "CMB_CST_RESULTS_DIR", "CMB_CST_THREADS"):
        monkeypatch.delenv(name, raising=False)
    source = Path(__file__).resolve().parents[1] / "scripts/run_reproduction.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_stage_names", "_output_directory", "parse_args", "main", "stage_transfer"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=selected, type_ignores=[])
    repo = tmp_path / "repo"
    repo.mkdir()
    namespace = {
        "__doc__": "CLI test", "argparse": argparse, "os": os, "sys": sys, "Path": Path, "np": np,
        "REPO": repo, "DEFAULT_WORK": repo / "data/work", "DEFAULT_RESULTS": repo / "results",
        "STAGES": ("masks", "alms", "spectra", "transfer", "analyse", "persistence"), "MASKS": ("0", "30"),
        "spectra": SimpleNamespace(NTHREADS=12), "beams": SimpleNamespace(BEAM_DIR=None),
        "_THEORY_CACHE": {"prior_run": True}, "log": lambda message: None,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def test_defaults_remain_compatible(runner):
    args = runner["parse_args"]([])
    assert args.work_dir == runner["DEFAULT_WORK"]
    assert args.results_dir == runner["DEFAULT_RESULTS"]
    assert (args.n_transfer_sims, args.n_sims, args.n_persistence_sims) == (8, 2000, 1000)
    assert args.stages == list(runner["STAGES"])
    assert not args.work_dir.exists()


def test_cli_overrides_environment_and_dispatches_selected_stages(runner, monkeypatch):
    repo = runner["REPO"]
    monkeypatch.setenv("CMB_CST_WORK_DIR", str(repo / "env-work"))
    monkeypatch.setenv("CMB_CST_RESULTS_DIR", str(repo / "env-results"))
    monkeypatch.setenv("CMB_CST_THREADS", "3")
    env = runner["parse_args"]([])
    assert env.work_dir == repo / "env-work"
    assert env.results_dir == repo / "env-results"
    assert env.threads == 3
    work, results = repo / "run/cache", repo / "run/output"
    calls = []
    runner["stage_transfer"] = lambda force, n: calls.append(("transfer", force, n))
    runner["stage_analyse"] = lambda n: calls.append(("analyse", n))
    masks = SimpleNamespace(NTHREADS=12)
    monkeypatch.setitem(sys.modules, "cmbcst.masks", masks)
    status = runner["main"](["--stages", "transfer,analyse", "--force", "transfer", "--n-transfer-sims", "12",
                             "--n-sims", "200", "--threads", "4", "--work-dir", str(work), "--results-dir", str(results)])
    assert status == 0
    assert calls == [("transfer", True, 12), ("analyse", 200)]
    assert runner["WORK"] == work and runner["RESULTS"] == results
    assert runner["beams"].BEAM_DIR == work / "beams"
    assert runner["_THEORY_CACHE"] == {}
    assert runner["spectra"].NTHREADS == masks.NTHREADS == 4
    assert work.is_dir() and results.is_dir()
    assert not (repo / "env-work").exists()


@pytest.mark.parametrize("arguments", [
    ["--stages", "alms,unknown"], ["--stages", ""], ["--stages", "masks,"],
    ["--stages", "masks,masks"], ["--stages", "analyse", "--force", "masks"],
    ["--stages", "analyse", "--force", "analyse"], ["--n-transfer-sims", "0"],
    ["--n-persistence-sims", "1"], ["--n-sims", "1"], ["--n-sims", "-1"], ["--threads", "0"],
])
def test_invalid_run_rejected_before_creating_directories(runner, arguments):
    with pytest.raises(SystemExit) as error:
        runner["main"](arguments)
    assert error.value.code == 2
    assert not runner["DEFAULT_WORK"].exists()
    assert not runner["DEFAULT_RESULTS"].exists()


def test_invalid_or_overlapping_directories_rejected(runner):
    repo = runner["REPO"]
    file = repo / "existing-file"
    file.write_text("keep")
    for path in (repo, repo.parent, repo / "src/generated", repo / "data/raw/run", file / "child"):
        with pytest.raises(SystemExit):
            runner["parse_args"](["--work-dir", str(path)])
    with pytest.raises(SystemExit):
        runner["parse_args"](["--work-dir", str(repo / "shared"), "--results-dir", str(repo / "shared/outputs")])
    assert file.read_text() == "keep"


def test_transfer_cache_matches_requested_ensemble_without_loading_physics(runner):
    work = runner["REPO"] / "cache"
    work.mkdir()
    runner["WORK"] = work
    for mask in runner["MASKS"]:
        np.savez(work / f"transfer_m{mask}.npz", n_sims=8)
    runner["stage_transfer"](False, 8)
    with pytest.raises(ValueError, match="--force transfer"):
        runner["stage_transfer"](False, 16)


def test_zero_bootstrap_is_still_supported(runner):
    assert runner["parse_args"](["--n-sims", "0"] ).n_sims == 0
