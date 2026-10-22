"""CMB-CST Lab: reproduction of the Lonappan, Keating & Arnold (2026) differential
polarization-calibration network on public Planck PR4 (NPIPE) maps, with a corrected
cochain/Hodge reading of the same network."""

# Map order used everywhere: index i in [0, 8) <-> DETECTORS[i]
DETECTORS = ("100A", "100B", "143A", "143B", "217A", "217B", "353A", "353B")
FREQS = ("100", "143", "217", "353")
