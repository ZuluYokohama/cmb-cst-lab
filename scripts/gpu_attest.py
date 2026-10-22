"""Record Gram matrices computed on this machine's accelerators, for the JoVin sandbox to verify.

The sandbox cannot reach a GPU or NPU, but it can check their output. For three real bin sizes of
the Planck mask-0 spectra (smallest, middle and largest nu_b), this script

  1. builds deterministic normals Z with accel.counter_normals (pure uint64 arithmetic, so the
     sandbox regenerates them bit for bit),
  2. computes Z Z^T / nu on each available device:
       - Intel Arc GPU via PyTorch XPU (float32), the device the bootstrap uses,
       - Intel NPU via an OpenVINO MatMul model (the NPU computes in reduced precision),
  3. writes the device results, the device names and the accept/reject decision under
     accel.REL_TOL into tests/sandbox/jovin_tests_accel.py between the ATTESTATION markers.

The sandbox suite recomputes every Gram matrix in float64 from the same normals and checks each
recorded error and decision. A device is used by the pipeline only if it is accepted.

Usage:
    python scripts/gpu_attest.py
"""
from __future__ import annotations

import json
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
    nu = np.load(REPO / "data" / "work" / "spectra_m0.npz")["nu"]
    out = []
    for k, b in enumerate((0, len(nu) // 2, len(nu) - 1)):
        out.append({"bin": int(b), "dof": int(max(round(float(nu[b])), NF + 1)), "seed": 1000 + k})
    return out


def previous_attestation() -> dict | None:
    text = TESTS.read_text(encoding="utf-8")
    block = text.split("# ATTESTATION:START", 1)[1].split("# ATTESTATION:END", 1)[0]
    ns: dict = {"json": json}
    exec(block, ns)
    return ns.get("ATTESTATION")


def run_xpu(z: np.ndarray) -> tuple[np.ndarray, str]:
    import torch

    return accel.gpu_gram(z)[0], f"{torch.xpu.get_device_name(0)} via PyTorch {torch.__version__} (float32)"


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


def main() -> int:
    devices = {}
    try:
        import torch

        if torch.xpu.is_available():
            devices["xpu"] = run_xpu
    except ImportError:
        pass
    try:
        import openvino as ov

        if "NPU" in ov.Core().available_devices:
            devices["npu"] = run_npu
    except ImportError:
        pass
    if not devices:
        print("no accelerator found; nothing recorded", file=sys.stderr)
        return 2

    att = {"generated_utc": datetime.now(timezone.utc).isoformat(), "rel_tol": accel.REL_TOL, "nf": NF,
           "cases": cases(), "devices": {}}
    # Keep records of devices not present on this machine, if they were measured on the same cases
    prev = previous_attestation()
    if prev and prev.get("cases") == att["cases"] and prev.get("nf") == NF:
        att["devices"].update({k: v for k, v in prev["devices"].items() if k not in devices})
    for name, fn in devices.items():
        rec = {"label": None, "grams": [], "max_rel_err": 0.0, "error": None}
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
