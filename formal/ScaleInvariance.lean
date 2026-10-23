import Std

/-!
Exact algebra for normalized even-increment moments. These are supporting
invariants, not a proof of a physical anomaly, fractal dimension, numerical
floating-point correctness, or SHA256 collision resistance.
-/
namespace CmbCst

theorem affine_increment (a b x y : Rat) :
    (a*x+b) - (a*y+b) = a*(x-y) := by
  grind

theorem normalized_fourth_invariant (a m2 m4 : Rat)
    (ha : a ≠ 0) (hm : m2 ≠ 0) :
    (a^4*m4) / (a^2*m2)^2 = m4 / m2^2 := by
  grind

theorem normalized_sixth_invariant (a m2 m6 : Rat)
    (ha : a ≠ 0) (hm : m2 ≠ 0) :
    (a^6*m6) / (a^2*m2)^3 = m6 / m2^3 := by
  grind

theorem directional_contrast_invariant (a x y : Rat)
    (ha : a ≠ 0) (hxy : x+y ≠ 0) :
    (a^2*x-a^2*y) / (a^2*x+a^2*y) = (x-y)/(x+y) := by
  grind

def rawMoment (n : Nat) (xs : List Rat) : Rat :=
  (xs.map (fun x => x^n)).sum

theorem rawMoment_two_scale (a : Rat) (xs : List Rat) :
    rawMoment 2 (xs.map (fun x => a*x)) = a^2 * rawMoment 2 xs := by
  induction xs with
  | nil => simp [rawMoment]
  | cons x xs ih =>
    simp [rawMoment] at *
    grind

theorem rawMoment_four_scale (a : Rat) (xs : List Rat) :
    rawMoment 4 (xs.map (fun x => a*x)) = a^4 * rawMoment 4 xs := by
  induction xs with
  | nil => simp [rawMoment]
  | cons x xs ih =>
    simp [rawMoment] at *
    grind

theorem rawMoment_six_scale (a : Rat) (xs : List Rat) :
    rawMoment 6 (xs.map (fun x => a*x)) = a^6 * rawMoment 6 xs := by
  induction xs with
  | nil => simp [rawMoment]
  | cons x xs ih =>
    simp [rawMoment] at *
    grind

def flatness (xs : List Rat) : Rat :=
  xs.length * rawMoment 4 xs / (rawMoment 2 xs)^2

def sixthRatio (xs : List Rat) : Rat :=
  (xs.length : Rat)^2 * rawMoment 6 xs / (rawMoment 2 xs)^3

theorem finite_flatness_invariant (a : Rat) (xs : List Rat)
    (ha : a ≠ 0) (hm : rawMoment 2 xs ≠ 0) :
    flatness (xs.map (fun x => a*x)) = flatness xs := by
  simp only [flatness, List.length_map, rawMoment_two_scale, rawMoment_four_scale]
  grind

theorem finite_sixth_ratio_invariant (a : Rat) (xs : List Rat)
    (ha : a ≠ 0) (hm : rawMoment 2 xs ≠ 0) :
    sixthRatio (xs.map (fun x => a*x)) = sixthRatio xs := by
  simp only [sixthRatio, List.length_map, rawMoment_two_scale, rawMoment_six_scale]
  grind

#print axioms affine_increment
#print axioms normalized_fourth_invariant
#print axioms normalized_sixth_invariant
#print axioms directional_contrast_invariant
#print axioms finite_flatness_invariant
#print axioms finite_sixth_ratio_invariant

end CmbCst
