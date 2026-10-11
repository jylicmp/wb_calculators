"""SI conversion factors introduced by the collected calculators."""

from scipy.constants import angstrom as Ang_SI
from scipy.constants import elementary_charge, hbar


TAU_UNIT = 1e-15  # one femtosecond in seconds

# Raw IMD tensors contain the 3D Brillouin-zone integral of a third
# energy derivative (eV after integration).  Convert eV to joule and supply
# -e^3 tau^2 / hbar^3.  Response symmetrization factors are applied outside
# the calculator, following WannierBerri's nonlinear-Drude convention.
factor_IMD2nd = -(
    elementary_charge**3 / hbar**2
    * TAU_UNIT**2
    * elementary_charge / hbar
)

# Raw IMQ tensors contain the 3D Brillouin-zone integral of a fourth
# energy derivative (eV Angstrom after integration).  Convert it to J m and
# supply e^4 tau^3 / hbar^4.
factor_IMQ3rd = (
    elementary_charge**4 / hbar**3
    * TAU_UNIT**3 * Ang_SI
    * elementary_charge / hbar
)

# Convert the 3D BCP quadrupole integral from Angstrom/eV to A m / V^3
# and supply the tau e^4 / hbar^2 response prefactor.
factor_BCPQ3rd = (
    elementary_charge**4 / hbar**3
    * TAU_UNIT * Ang_SI
    * hbar / elementary_charge
)
