# Exact rational support for the finite-scale features

`ScaleInvariance.lean` contains nine kernel-checked algebraic theorems with
Lean 4.35.0, using only its standard library. They prove cancellation of an
affine brightness offset in increments, homogeneous scaling of second/fourth/
sixth sample moment sums, and invariance of normalized fourth/sixth moment
ratios and a directional contrast when the denominators and scale are nonzero.

The proof uses exact rationals and explicit assumptions. It does not prove
the log-slope summaries, the Python/CUDA floating-point implementation on every
input, any sky anomaly, multifractality, cosmology, or cryptographic security.
It uses Lean's standard foundations: propext, Classical.choice and Quot.sound.
No admitted theorem or extra axiom declaration is present. No Mathlib download
is needed. The supporting algebra is elementary established mathematics, not
a new theorem or a physical discovery.

Pinned official Linux archive:
`https://github.com/leanprover/lean4/releases/download/v4.35.0/lean-4.35.0-linux.tar.zst`

SHA256: `7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d`.

After checking and extracting that archive, run from the repository:

```sh
python scripts/verify_lean.py --lean /path/to/lean/bin/lean --archive /path/to/lean-4.35.0-linux.tar.zst
```

The verifier records the compiler version/commit, source and numerical module
hashes, nine theorem names, compiler output, assumptions and limits in
`results/multiscale_v1/lean_verification.json`. CI repeats the proof with the
same pinned compiler archive. SHA256 identifies bytes; these receipts are not
digital signatures and do not establish authorship or scientific truth.
