import OperationalClosure

/- Exact recorded-bin support for V2 source tile 24, start 56.

   These four tables are measured-data hypotheses when related to a sky state.
   The kernel checks integer table arithmetic and the conditional implication;
   it does not verify floating-point quantization, ducc sampling, GPU execution,
   the global spectral measurements, or any physical interpretation.

   The imported theorem is the unchanged CmbCst.no_factor_from_collision.
   Its compiled dependency belongs in an external temporary LEAN_PATH prefix,
   never next to the frozen OperationalClosure.lean source. -/
namespace CmbCst.ClosureFibersV2

universe u v

def currentReferenceBins : List Int :=
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def currentCandidateBins : List Int :=
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def futureReferenceBins : List Int :=
  [-3448, -4662, -4284, -4565, -3520, 643, -533, -1200,
   -3171, -4163, -1846, -3492, -3986, 1049, -1188, -594,
   -2755, -2157, -1071, -2213, -2884, 1096, 232, -231,
   -3034, -1896, -959, -1930, -3134, 1086, 280, -1112,
   -589, -489, -508, -1730, -1715, -1217, -4624, -6260,
   -1540, -5019, -5406, 74]

def futureCandidateBins : List Int :=
  [-3445, -4659, -4273, -4560, -3520, 656, -413, -1238,
   -3167, -4161, -1832, -3487, -3986, 1063, -895, -627,
   -2750, -2155, -1059, -2208, -2884, 1107, 1007, -261,
   -3025, -1895, -949, -1925, -3134, 1095, 616, -1054,
   -601, -499, -524, -1723, -1715, -1211, -4622, -6259,
   -1534, -5019, -5406, 74]

/- Zero-based lookup by structural recursion, retaining direct kernel reduction
   of the integer tables without dependent-index library rewriting. -/
def binAt : Nat → List Int → Option Int
  | _, [] => none
  | 0, x :: _ => some x
  | n + 1, _ :: xs => binAt n xs

theorem recorded_table_lengths :
    currentReferenceBins.length = 44 ∧ currentCandidateBins.length = 44 ∧
    futureReferenceBins.length = 44 ∧ futureCandidateBins.length = 44 := by
  decide

theorem recorded_current_bins_equal :
    currentReferenceBins = currentCandidateBins := by
  decide

theorem recorded_primary_future_bins :
    binAt 22 futureReferenceBins = some (232 : Int) ∧
    binAt 22 futureCandidateBins = some (1007 : Int) := by
  decide

theorem recorded_primary_values_differ : (232 : Int) ≠ 1007 := by
  decide

theorem recorded_future_bins_differ :
    futureReferenceBins ≠ futureCandidateBins := by
  intro same
  have selected : some (232 : Int) = some (1007 : Int) := by
    calc
      some (232 : Int) = binAt 22 futureReferenceBins := recorded_primary_future_bins.1.symm
      _ = binAt 22 futureCandidateBins := congrArg (binAt 22) same
      _ = some (1007 : Int) := recorded_primary_future_bins.2
  exact recorded_primary_values_differ (Option.some.inj selected)

/- Global is the complete retained global representation, including its
   recorded spectral bins. Its equality at the two current states is an
   explicit hypothesis. No equality of future global records is required:
   the different future local coordinate already separates the product. -/
theorem conditional_recorded_no_factor
    {State : Type u} {Global : Type v}
    (R : State → Global × List Int) (T : State → State) (s t : State)
    (same_global : (R s).1 = (R t).1)
    (current_reference : (R s).2 = currentReferenceBins)
    (current_candidate : (R t).2 = currentCandidateBins)
    (future_reference : (R (T s)).2 = futureReferenceBins)
    (future_candidate : (R (T t)).2 = futureCandidateBins) :
    ¬ ∃ F : (Global × List Int) → (Global × List Int),
      ∀ x : State, R (T x) = F (R x) := by
  apply CmbCst.no_factor_from_collision R T s t
  · apply Prod.ext
    · exact same_global
    · calc
        (R s).2 = currentReferenceBins := current_reference
        _ = currentCandidateBins := recorded_current_bins_equal
        _ = (R t).2 := current_candidate.symm
  · intro same
    apply recorded_future_bins_differ
    calc
      futureReferenceBins = (R (T s)).2 := future_reference.symm
      _ = (R (T t)).2 := congrArg Prod.snd same
      _ = futureCandidateBins := future_candidate

#print axioms recorded_table_lengths
#print axioms recorded_current_bins_equal
#print axioms recorded_primary_future_bins
#print axioms recorded_primary_values_differ
#print axioms recorded_future_bins_differ
#print axioms conditional_recorded_no_factor
#check conditional_recorded_no_factor
end CmbCst.ClosureFibersV2
