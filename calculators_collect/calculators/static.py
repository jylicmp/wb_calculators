"""Metric, BCP, IMD and IMQ calculators using the upstream integration engine.

Metric and BCP velocity-product formulas use additive=False. WannierBerri
integrates differences of cumulative subspace traces; at zero temperature the
sea result is the occupied-subspace trace. BCPQ_FermiSea includes its SI
third-order-response conversion factor for one femtosecond. IMD/IMQ instead
use band/group-additive energy derivatives and one-femtosecond SI factors.
"""
from wannierberri.calculators.static import StaticCalculator
from ..formula import covariant as frml
from ..factors import factor_BCPQ3rd, factor_IMD2nd, factor_IMQ3rd


class Metric(StaticCalculator):
    r"""Quantum metric (:math:`\mathrm{\AA}^{-1}` in 3D).

        | With Fermi sea integral at zero temperature.
        | Output: :math:`I^{ab} = \int [dk]\,\mathrm{Tr}_{\mathrm{occ}}\widetilde G^{ab}`
        | Tensor axes: :math:`(a,b)`.

        Uses the occupied-subspace metric with ``additive=False``.
        No electrical-response prefactor is included.
    """
    def __init__(self, **kwargs):
        super().__init__(Formula=frml.Metric, fder=0, **kwargs)


class MetricDipole_FermiSea(StaticCalculator):
    r"""Quantum metric dipole (dimensionless in 3D).

        | With Fermi sea integral at zero temperature.
        | Output: :math:`D^{abd} = \int [dk]\,\mathrm{Tr}_{\mathrm{occ}}\widetilde G^{ab:d}`
        | Tensor axes: :math:`(a,b,d)`.

        The covariant derivative acts on a smooth subspace kernel, not on the
        occupation step. No electrical-response prefactor is included.
    """
    def __init__(self, **kwargs):
        super().__init__(Formula=frml.DerMetric, fder=0, **kwargs)


class MetricQuadrupole_FermiSurf(StaticCalculator):
    r"""Quantum metric derivative surface kernel (:math:`\mathrm{\AA}` in 3D).

        | With Fermi surface integral at zero temperature.
        | Output: :math:`Q^{cabd} = \partial_\mu\int [dk]\,\mathrm{Tr}_{\mathrm{occ}}[V^c\widetilde G^{ab:d}]`
        | Tensor axes: :math:`(c,a,b,d)`; :math:`V^c=\partial_c\mathcal H`.

        Uses the nonadditive ``VelDerMetric`` kernel and ``fder=1``. The
        non-tetra path approximates the chemical-potential derivative by a
        centered finite difference. No electrical-response prefactor is included.
    """
    def __init__(self, **kwargs):
        super().__init__(Formula=frml.VelDerMetric, fder=1, **kwargs)


class IMD_FermiSea(StaticCalculator):
    r"""Inverse-mass dipole contribution (:math:`\mathrm{A/V^2}`).

        | With Fermi sea integral.
        | Output: :math:`\chi^{abc}=-e^3\tau^2\hbar^{-3}\int[dk]f_n\,\partial_a\partial_b\partial_c\varepsilon_n`
        | Tensor axes: :math:`(a,b,c)`.

        The default ``constant_factor`` uses :math:`\tau=1\,\mathrm{fs}`.
        The conventional factor associated with symmetrizing the two electric
        fields is applied when constructing the final conductivity tensor.
        The energy-derivative group trace is additive.
    """

    def __init__(self, constant_factor=factor_IMD2nd, **kwargs):
        super().__init__(
            Formula=frml.Der3E, fder=0,
            constant_factor=constant_factor, **kwargs)


class IMD_FermiSurf(StaticCalculator):
    r"""Inverse-mass dipole contribution (:math:`\mathrm{A/V^2}`).

        | With Fermi surface integral.
        | Output: :math:`\chi^{abc}=e^3\tau^2\hbar^{-3}\int[dk](\partial_a\partial_b\varepsilon_n)(\partial_c\varepsilon_n)f'_n`
        | Tensor axes: :math:`(a,b,c)`.

        ``fder=1`` supplies the positive :math:`-f'_n` weight; together with
        the negative default factor this gives the displayed expression.  The
        product is evaluated separately for each band or exact degeneracy group.
    """

    def __init__(self, constant_factor=factor_IMD2nd, **kwargs):
        super().__init__(
            Formula=frml.MassVel, fder=1,
            constant_factor=constant_factor, **kwargs)


class IMQ_FermiSea(StaticCalculator):
    r"""Inverse-mass quadrupole contribution (:math:`\mathrm{A\,m/V^3}`).

        | With Fermi sea integral.
        | Output: :math:`\sigma^{abcd}=e^4\tau^3\hbar^{-4}\int[dk]f_n\,\partial_a\partial_b\partial_c\partial_d\varepsilon_n`
        | Tensor axes: :math:`(a,b,c,d)`.

        The default ``constant_factor`` uses :math:`\tau=1\,\mathrm{fs}` and
        converts the raw 3D integral from :math:`\mathrm{eV\,\AA}` to
        :math:`\mathrm{J\,m}`.  The fourth-energy-derivative trace is additive.
    """

    def __init__(self, constant_factor=factor_IMQ3rd, **kwargs):
        super().__init__(
            Formula=frml.Der4E, fder=0,
            constant_factor=constant_factor, **kwargs)


class IMQ_FermiSurf(StaticCalculator):
    r"""Inverse-mass quadrupole contribution (:math:`\mathrm{A\,m/V^3}`).

        | With Fermi surface integral.
        | Output: :math:`\sigma^{abcd}=-e^4\tau^3\hbar^{-4}\int[dk](\partial_a\partial_b\partial_c\varepsilon_n)(\partial_d\varepsilon_n)f'_n`
        | Tensor axes: :math:`(a,b,c,d)`.

        ``fder=1`` supplies :math:`-f'_n`, so the positive default factor is
        identical to the Fermi-sea factor.  The product is evaluated separately
        for each band or exact degeneracy group.
    """

    def __init__(self, constant_factor=factor_IMQ3rd, **kwargs):
        super().__init__(
            Formula=frml.Der3EVel, fder=1,
            constant_factor=constant_factor, **kwargs)


class BCPD_FermiSea(StaticCalculator):
    r"""Berry connection polarizability dipole (:math:`\mathrm{eV}^{-1}` in 3D).

        | With Fermi sea integral at zero temperature.
        | Output: :math:`\mathcal D^{abd} = C\int [dk]\,\mathrm{Tr}_{\mathrm{occ}}\widetilde{\mathcal G}^{ab:d}`
        | Tensor axes: :math:`(a,b,d)`; transpose to :math:`(d,a,b)` with ``(2,0,1)``.

        Here :math:`C` is ``constant_factor`` (default 1). The default output
        is geometric; charge, hbar and response symmetrizations are not included.
        The derivative trace is additive and the sea weight is :math:`f`, not :math:`f'`.
    """
    def __init__(self, constant_factor=1., **kwargs):
        super().__init__(Formula=frml.DerBCP, fder=0, constant_factor=constant_factor, **kwargs)


class BCPQ_FermiSea(StaticCalculator):
    r"""BCP quadrupole kernel for third-order conductivity (:math:`\mathrm{A\,m/V^3}`).

        | With Fermi sea integral at zero temperature.
        | Output: :math:`\mathcal Q^{abde} = C\int [dk]\,\mathrm{Tr}_{\mathrm{occ}}\widetilde{\mathcal G}^{ab:de}`
        | Tensor axes: :math:`(a,b,d,e)`; transpose to :math:`(d,a,b,e)` with ``(2,0,1,3)``.

        Uses the full Sylvester second derivative. The default ``constant_factor``
        is :math:`e^4\tau\hbar^{-2}` together with the conversion from
        :math:`\mathrm{\AA/eV}` to :math:`\mathrm{m/J}`, for :math:`\tau=1\,\mathrm{fs}`.
        The three index permutations entering a specified conductivity tensor
        are formed from this raw :math:`\mathcal Q^{abde}` output separately.
    """
    def __init__(self, constant_factor=factor_BCPQ3rd, **kwargs):
        super().__init__(Formula=frml.Der2BCP, fder=0, constant_factor=constant_factor, **kwargs)


class BCPD_FermiSurf(StaticCalculator):
    r"""Berry connection polarizability velocity surface kernel (:math:`\mathrm{eV}^{-1}` in 3D).

        | With Fermi surface integral at zero temperature.
        | Output: :math:`\mathcal D_{\mathrm{surf}}^{cab} = C\partial_\mu\int [dk]\,\mathrm{Tr}_{\mathrm{occ}}[V^c\widetilde{\mathcal G}^{ab}]`
        | Tensor axes: :math:`(c,a,b)`; :math:`V^c=\partial_c\mathcal H`.

        Here :math:`C` is ``constant_factor`` (default 1). ``fder=1`` applies
        the positive :math:`-f'` weight to cumulative-subspace trace differences
        of the nonadditive ``VelBCP`` kernel. This is not automatically the
        isolated-band response or the sea dipole after axis permutation.
    """
    def __init__(self, constant_factor=1., **kwargs):
        super().__init__(Formula=frml.VelBCP, fder=1, constant_factor=constant_factor, **kwargs)


class BCPQ_FermiSurf(StaticCalculator):
    r"""Berry connection polarizability derivative surface kernel (:math:`\mathrm{\AA}/\mathrm{eV}` in 3D).

        | With Fermi surface integral at zero temperature.
        | Output: :math:`\mathcal Q_{\mathrm{surf}}^{cabd} = C\partial_\mu\int [dk]\,\mathrm{Tr}_{\mathrm{occ}}[V^c\widetilde{\mathcal G}^{ab:d}]`
        | Tensor axes: :math:`(c,a,b,d)`; :math:`V^c=\partial_c\mathcal H`.

        Here :math:`C` is ``constant_factor`` (default 1). ``fder=1`` uses the
        nonadditive ``VelDerBCP`` kernel and cumulative-subspace differences.
        No isolated-band sea/surface identity or electrical prefactor is assumed.
    """
    def __init__(self, constant_factor=1., **kwargs):
        super().__init__(Formula=frml.VelDerBCP, fder=1, constant_factor=constant_factor, **kwargs)
