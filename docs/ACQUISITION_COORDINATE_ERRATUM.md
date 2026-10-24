# Acquisition V1 coordinate-export sign erratum

The frozen forward operator uses `exp(-2*pi*i*frequency*offset)`. Under the FFT
convention used here, that samples the beam-smoothed field at nominal position
**minus** the offset. The original labeled_samples.csv incorrectly called
nominal-plus-offset values `pointing_x_deg` and `pointing_y_deg`.

The original CSV, numerical source, observations and metrics are preserved.
Use [labeled_samples_corrected.csv](../results/acquisition_v1/labeled_samples_corrected.csv)
for physical look coordinates. It retains the original values as
`registration_plus_offset_x/y_deg` and adds explicit `look_x/y_unwrapped_deg`
and `look_x/y_wrapped_deg`. Wrapping belongs to the toy
periodic boundary, not to an actual instrument's pointing solution.

For the first detector at pixel (0,0), x offset +0.75 pixels yields unwrapped
look x = -10.029296875 degrees, equivalent to +9.970703125 degrees on the 20 degree
periodic patch. The original plus-offset label was -9.970703125 degrees.

An independent analytic cosine-mode test checks the sample coordinate sign,
including the beam transfer and gain. It distinguishes the two conventions;
the forward/inverse roundtrip alone cannot detect this shared labeling error.
The coordinate helper and test are separately JoVin-verified after the
original scientific freeze. No seed, numerical threshold or physical operator
was changed to obtain a passing result. This is a metadata correction and does
not establish physical accuracy of the broader toy instrument.
