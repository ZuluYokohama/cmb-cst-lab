"""Record Gram matrices computed on this machine's accelerators, for the JoVin sandbox to verify.

The sandbox cannot reach a GPU or NPU, but it can check their output. For three real bin sizes of
the Planck mask-0 spectra (smallest, middle and largest nu_b), this script

  1. builds deterministic normals Z with accel.counter_normals (pure uint64 arithmetic, so the
     sandbox regenerates them bit for bit),
  2. computes Z Z^T / nu on the selected CUDA/XPU GPU in IEEE float32,
     and optionally on an OpenVINO NPU,
  3. writes the device results, the device names and the accept/reject decision under
     accel.REL_TOL into tests/sandbox/jovin_tests_accel.py between the ATTESTATION markers.

The sandbox suite recomputes every Gram matrix in float64 from the same normals and checks each
recorded error and decision. A device is used by the pipeline only if it is accepted.

Usage:
    python scripts/gpu_attest.py --device cuda:0

Without the map cache, use the previously recorded Planck bin sizes explicitly as
reference workloads. Retain historical devices measured on those same cases;
the timestamp inside each new device record identifies the fresh measurement.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cmbcst import accel  # noqa: E402

TESTS = REPO / "tests" / "sandbox" / "jovin_tests_accel.py"
NF = 16


def cases() -> list[dict]:
    spectra = REPO / "data" / "work" / "spectra_m0.npz"
    if not spectra.exists():
        previous = previous_attestation()
        if not previous or previous.get("nf") != NF:
            raise RuntimeError("No spectra cache or recorded Planck workload sizes available")
        return previous["cases"]
    nu = np.load(spectra)["nu"]
    out = []
    for k, b in enumerate((0, len(nu) // 2, len(nu) - 1)):
        out.append({"bin": int(b), "dof": int(max(round(float(nu[b])), NF + 1)), "seed": 1000 + k})
    return out


def previous_attestation() -> dict | None:
    text = TESTS.read_text(encoding="utf-8")
    block = text.split("# ATTESTATION:START", 1)[1].split("# ATTESTATION:END", 1)[0]
    tree = ast.parse(block.strip())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ATTESTATION" for t in statement.targets):
            call = statement.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
                    and call.func.attr == "loads" and len(call.args) == 1):
                return json.loads(ast.literal_eval(call.args[0]))
    return None


def run_gpu(z: np.ndarray) -> tuple[np.ndarray, str]:
    return accel.gpu_gram(z)[0], accel.backend_name()


def run_npu(z: np.ndarray) -> tuple[np.ndarray, str]:
    import openvino as ov
    import openvino.opset13 as ops

    core = ov.Core()
    nf, dof = z.shape
    p = ops.parameter([1, nf, dof], ov.Type.f32, name="z")
    g = ops.divide(ops.matmul(p, p, transpose_a=False, transpose_b=True), ops.constant(np.float32(dof)))
    model = ov.Model([g], [p], "gram")
    compiled = core.compile_model(model, "NPU")
    res = compiled([z[None]])[compiled.output(0)][0]
    return np.asarray(res, dtype=np.float32), f"{core.get_property('NPU', 'FULL_DEVICE_NAME')} via OpenVINO {ov.__version__}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("CMB_CST_DEVICE", "auto"),
                        help="auto, cuda:N, or xpu:N; explicitly selects the measured GPU")
    parser.add_argument("--include-npu", action="store_true", help="also measure an available OpenVINO NPU")
    args = parser.parse_args(argv)
    os.environ["CMB_CST_DEVICE"] = args.device
    torch, dev = accel._device()
    devices = {}
    if dev is not None:
        selected = "xpu" if dev.type == "xpu" and dev.index in (None, 0) else str(dev)
        devices[selected] = run_gpu
    else:
        selected = None
    if args.include_npu:
        import openvino as ov
        if "NPU" in ov.Core().available_devices:
            devices["npu"] = run_npu
    if not devices:
        print("no accelerator found; nothing recorded", file=sys.stderr)
        return 2

    measured_at = datetime.now(timezone.utc).isoformat()
    att = {"generated_utc": measured_at, "rel_tol": accel.REL_TOL, "nf": NF,
           "cases": cases(), "devices": {}, "selected_device": selected,
           "case_source": "local_spectra" if (REPO / "data/work/spectra_m0.npz").exists() else "historical_recorded_planck_bin_sizes",
           "note": "Synthetic deterministic normals at recorded Planck bin sizes; no real-sky inference is performed."}
    # Keep records of devices not present on this machine, if they were measured on the same cases
    prev = previous_attestation()
    if prev and prev.get("cases") == att["cases"] and prev.get("nf") == NF:
        att["devices"].update({k: v for k, v in prev["devices"].items() if k not in devices})
    for name, fn in devices.items():
        rec = {"label": None, "grams": [], "max_rel_err": 0.0, "error": None,
               "measured_utc": measured_at,
               "accel_source_sha256": hashlib.sha256((REPO / "src/cmbcst/accel.py").read_bytes()).hexdigest()}
        if name == selected and dev.type == "cuda":
            rec["cuda_runtime"] = torch.version.cuda
            rec["compute_capability"] = list(torch.cuda.get_device_capability(dev))
            rec["total_memory_bytes"] = torch.cuda.get_device_properties(dev).total_memory
            rec["precision"] = "IEEE float32 matmul, float64 CPU reference"
        try:
            for c in att["cases"]:
                z = accel.counter_normals(c["seed"], NF, c["dof"])
                g, rec["label"] = fn(z)
                rec["grams"].append(np.asarray(g, dtype=np.float32).astype(float).tolist())
                rec["max_rel_err"] = max(rec["max_rel_err"], accel.max_rel_err(g, accel.gram64(z)))
        except Exception as e:  # a device that cannot run the product is recorded as such
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["accepted"] = rec["error"] is None and rec["max_rel_err"] <= accel.REL_TOL
        att["devices"][name] = rec
        print(f"{name}: {rec['label'] or rec['error']}  max_rel_err={rec['max_rel_err']:.3e}  accepted={rec['accepted']}")

    text = TESTS.read_text(encoding="utf-8")
    start, end = "# ATTESTATION:START", "# ATTESTATION:END"
    i, j = text.index(start) + len(start), text.index(end)
    body = "\nATTESTATION = json.loads(" + repr(json.dumps(att, separators=(",", ":"))) + ")\n"
    TESTS.write_text(text[:i] + body + text[j:], encoding="utf-8")
    print(f"wrote {TESTS.relative_to(REPO)} ({len(body) / 1024:.0f} KB of attestation data)")
    evidence = REPO / "data/evidence/gpu_attestation_latest.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(att, indent=2) + "\n", encoding="utf-8")
    return 0 if selected and att["devices"][selected]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
