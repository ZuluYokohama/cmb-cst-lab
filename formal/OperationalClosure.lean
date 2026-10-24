import Std

/- Conditional operational closure, following the supplied CST working paper
   section 8.1. These statements concern typed mathematical functions and an
   exact rational example. They do not certify a floating-point measurement,
   an optimizer, a sky model, or a cosmological interpretation.

   Rotation invariants used by the numerical study remain in the unchanged
   PolarizationGeometry.lean; no stronger rotation claim is introduced here. -/
namespace CmbCst

universe u v w

theorem no_predictor_from_collision
    {S : Type u} {Z : Type v} {W : Type w}
    (R : S → Z) (Y : S → W) (s t : S)
    (same : R s = R t) (different : Y s ≠ Y t) :
    ¬ ∃ F : Z → W, ∀ x : S, Y x = F (R x) := by
  intro factor
  obtain ⟨F, agrees⟩ := factor
  apply different
  calc
    Y s = F (R s) := agrees s
    _ = F (R t) := congrArg F same
    _ = Y t := (agrees t).symm

theorem no_factor_from_collision
    {S : Type u} {Z : Type v}
    (R : S → Z) (T : S → S) (s t : S)
    (same : R s = R t) (different : R (T s) ≠ R (T t)) :
    ¬ ∃ F : Z → Z, ∀ x : S, R (T x) = F (R x) := by
  exact no_predictor_from_collision R (fun x => R (T x)) s t same different

structure FourState where
  a : Rat
  b : Rat
  c : Rat
  d : Rat

def adjacentAverage (x : FourState) : FourState :=
  ⟨(x.a + x.b) / 2, (x.b + x.c) / 2,
   (x.c + x.d) / 2, (x.d + x.a) / 2⟩

def momentProjection (x : FourState) : Rat × Rat :=
  ((x.a * x.a + x.b * x.b + x.c * x.c + x.d * x.d) / 4,
   ((x.a * x.a) * (x.a * x.a) + (x.b * x.b) * (x.b * x.b) +
    (x.c * x.c) * (x.c * x.c) + (x.d * x.d) * (x.d * x.d)) / 4)

def toyLeft : FourState := ⟨1, 1, -1, -1⟩
def toyRight : FourState := ⟨1, -1, 1, -1⟩

theorem toy_initial_moments :
    momentProjection toyLeft = (1, 1) ∧
    momentProjection toyRight = (1, 1) := by
  constructor <;> apply Prod.ext <;> dsimp [momentProjection, toyLeft, toyRight] <;> grind

theorem toy_left_next_moments :
    momentProjection (adjacentAverage toyLeft) = (1 / 2, 1 / 2) := by
  apply Prod.ext <;> dsimp [momentProjection, adjacentAverage, toyLeft] <;> grind

theorem toy_right_next_moments :
    momentProjection (adjacentAverage toyRight) = (0, 0) := by
  apply Prod.ext <;> dsimp [momentProjection, adjacentAverage, toyRight] <;> grind

theorem toy_next_moments_differ :
    momentProjection (adjacentAverage toyLeft) ≠
    momentProjection (adjacentAverage toyRight) := by
  rw [toy_left_next_moments, toy_right_next_moments]
  intro equal
  have : (1 / 2 : Rat) = 0 := congrArg Prod.fst equal
  grind

theorem toy_no_moment_factor :
    ¬ ∃ F : (Rat × Rat) → (Rat × Rat),
      ∀ x : FourState, momentProjection (adjacentAverage x) = F (momentProjection x) := by
  exact no_factor_from_collision momentProjection adjacentAverage toyLeft toyRight
    (toy_initial_moments.1.trans toy_initial_moments.2.symm) toy_next_moments_differ

#print axioms no_predictor_from_collision
#print axioms no_factor_from_collision
#print axioms toy_initial_moments
#print axioms toy_left_next_moments
#print axioms toy_right_next_moments
#print axioms toy_next_moments_differ
#print axioms toy_no_moment_factor
end CmbCst
