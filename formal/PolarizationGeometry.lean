import Std

/- Exact rational basis identities; not a proof of floating-point inversion,
   beam physics, calibration accuracy or a cosmological discovery. -/
namespace CmbCst

theorem polarization_amplitude_invariant (c s q u : Rat)
    (unit : c*c+s*s=1) :
    (c*q-s*u)^2+(s*q+c*u)^2=q^2+u^2 := by
  grind

theorem corotated_measurement_invariant (c s q u a b : Rat)
    (unit : c*c+s*s=1) :
    (c*a-s*b)*(c*q-s*u)+(s*a+c*b)*(s*q+c*u)=a*q+b*u := by
  grind

#print axioms polarization_amplitude_invariant
#print axioms corotated_measurement_invariant
end CmbCst
