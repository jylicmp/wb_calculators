"""Metric, BCP and fourth-energy-derivative additions.

This is an extension module, not a replacement for upstream covariant.py.
G is the quantum metric; BCP uses calligraphic G.
Cartesian axes: Metric (a,b), DerMetric (a,b,d).
external_terms=False means the internal model only; cross_terms controls
D-A cross terms within the enabled external contribution.
Der3E, InvMass and MassVel are re-exported unchanged from upstream;
Der4E and Der3EVel are the collected IMQ additions.
"""
import numpy as np
from wannierberri.utility import cached_einsum
from wannierberri.formula.formula import Formula_ln, FormulaProduct, Matrix_GenDer_ln
from wannierberri.formula.covariant import (
    Dcov,
    DerDcov,
    Der2Dcov,
    Der3E,
    DerWln,
    InvMass,
    MassVel,
)
from wannierberri.symmetry.point_symmetry import transform_ident, transform_odd
from ..system.needed_data import validate_metric_data


class Metric(Formula_ln):
    r"""Subspace quantum metric G_mn^{ab}; eq:Finterpolation-components.

    Requires AA and FF for the full physical geometry. A larger active space
    gives its subspace metric, not the sum of individual band metrics.
    """

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        validate_metric_data(data_K, external_terms=self.external_terms)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.F = data_K.covariant('FF')

        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)

        if self.internal_terms:
            summ += - (1/2) * cached_einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out),
                self.D.ln(ik, inn, out))

        if self.external_terms:
            summ += (1/2) * self.F.nn(ik, inn, out)
            if self.cross_terms:
                summ += (1j/2) * cached_einsum(
                    "mla,lnb->mnab",
                    self.D.nl(ik, inn, out),
                    self.A.ln(ik, inn, out))
                summ += (1j/2) * cached_einsum(
                    "mla,lnb->mnab",
                    self.A.nl(ik, inn, out),
                    self.D.ln(ik, inn, out))
            # k denotes the contracted n' in A; both factors use .nn.
            summ += -(1/2) * cached_einsum(
                "mka,knb->mnab",
                self.A.nn(ik, inn, out),
                self.A.nn(ik, inn, out))

        summ += summ.swapaxes(0, 1).conj()
        return summ

    @property
    def additive(self):
        """A subspace metric is not additive under regrouping bands."""
        return False

    def ln(self, ik, inn, out):
        raise NotImplementedError()

class DerMetric(Formula_ln):
    r"""Covariant derivative G_mn^{ab:d}; eq:Ffirst-components.

    The two negative AA-AA products contract n' in the active space.
    On taking the active-space trace, this is the ordinary derivative.
    """

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        validate_metric_data(data_K, external_terms=self.external_terms)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.dF = data_K.covariant('FF', gender=1)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -0.5 * cached_einsum(
                "mla,lnbd->mnabd",
                self.D.nl(ik, inn, out),
                self.dD.ln(ik, inn, out))
            summ += -0.5 * cached_einsum(
                "mlad,lnb->mnabd",
                self.dD.nl(ik, inn, out),
                self.D.ln(ik, inn, out))

        if self.external_terms:
            summ += 0.5 * self.dF.nn(ik, inn, out)
            if self.cross_terms:
                summ += 0.5j * cached_einsum(
                    "mlad,lnb->mnabd",
                    self.dD.nl(ik, inn, out),
                    self.A.ln(ik, inn, out))
                summ += 0.5j * cached_einsum(
                    "mla,lnbd->mnabd",
                    self.D.nl(ik, inn, out),
                    self.dA.ln(ik, inn, out))
                summ += 0.5j * cached_einsum(
                    "mlad,lnb->mnabd",
                    self.dA.nl(ik, inn, out),
                    self.D.ln(ik, inn, out))
                summ += 0.5j * cached_einsum(
                    "mla,lnbd->mnabd",
                    self.A.nl(ik, inn, out),
                    self.dD.ln(ik, inn, out))
            # Differentiate the AA-AA subtraction: k=n' stays in A.
            summ += -0.5 * cached_einsum(
                "mkad,knb->mnabd",
                self.dA.nn(ik, inn, out),
                self.A.nn(ik, inn, out))
            summ += -0.5 * cached_einsum(
                "mka,knbd->mnabd",
                self.A.nn(ik, inn, out),
                self.dA.nn(ik, inn, out))

        summ += summ.swapaxes(0, 1).conj()
        return summ

    @property
    def additive(self):
        """A subspace metric is not additive under regrouping bands."""
        return False

    def ln(self, ik, inn, out):
        raise NotImplementedError()

class VelMetric(FormulaProduct):
    """Energy gradient times metric kernel; raw Cartesian axes (c,a,b).

    The Ham derivative is E^c = hbar*v^c, not the SI velocity.
    """
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Metric(data_K, **kwargs_formula)], name='VelMetric')

    @property
    def additive(self):
        return False


class MetricVel(FormulaProduct):
    """Energy gradient times metric kernel; raw Cartesian axes (a,b,c).

    The Ham derivative is E^c = hbar*v^c, not the SI velocity.
    """
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Metric(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='MetricVel')

    @property
    def additive(self):
        return False


class VelDerMetric(FormulaProduct):
    """Energy gradient times metric kernel; raw Cartesian axes (c,a,b,d).

    The Ham derivative is E^c = hbar*v^c, not the SI velocity.
    """
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), DerMetric(data_K, **kwargs_formula)], name='VelDerMetric')

    @property
    def additive(self):
        return False



class BCP(Formula_ln):
    r"""Finite-model mathcal G_mn^{ab}; handbook eq:bcp-code-components.

    Uses 1/4*(T+T^dagger), T=N_AB M_BA-M_AB N_BA, without an extra factor 2.
    internal_terms selects DD, external_terms AA, and external+cross_terms
    the two DA products. The trace is additive; the matrix is not bandwise.
    """
    derivative_order = 0

    def __init__(self, data_K, **parameters):
        from .covariant_basic import _BCPConnections
        from ..system.needed_data import validate_bcp_data
        super().__init__(data_K, **parameters)
        validate_bcp_data(data_K, external_terms=self.external_terms)
        self.ndim = 2 + self.derivative_order
        parity = transform_odd if self.derivative_order % 2 else transform_ident
        self.transformTR = parity
        self.transformInv = parity
        self.connections = _BCPConnections(data_K, self.derivative_order, self.external_terms)

    def nn(self, ik, inn, out):
        from .covariant_basic import _bcp_product
        result = np.zeros((len(inn), len(inn)) + (3,) * self.ndim, dtype=complex)
        if not len(inn) or not len(out):
            return result
        pairs = []
        if self.internal_terms:
            pairs.append(('D', 'D'))
        if self.external_terms:
            pairs.append(('A', 'A'))
            if self.cross_terms:
                pairs.extend([('D', 'A'), ('A', 'D')])
        if not pairs:
            return result
        jets = self.connections.evaluate(ik, inn, out)
        count = len(inn)
        def ab(x):
            return [t[:count, count:] for t in x]
        def ba(x):
            return [t[count:, :count] for t in x]
        for left, right in pairs:
            ml, nl = jets[left]
            mr, nr = jets[right]
            result += _bcp_product(ab(nl), ba(mr), self.derivative_order)
            result -= _bcp_product(ab(ml), ba(nr), self.derivative_order)
        return .25 * (result + result.swapaxes(0, 1).conj())

    @property
    def additive(self):
        """Pair contributions cancel in the trace because Delta_mn=-Delta_nm."""
        return True

    def ln(self, ik, inn, out):
        raise NotImplementedError('BCP is defined within the selected subspace.')


class DerBCP(BCP):
    r"""mathcal G^{ab:d}, axes (a,b,d), via the full Sylvester derivative."""
    derivative_order = 1


class Der2BCP(BCP):
    r"""mathcal G^{ab:de}, axes (a,b,d,e), ordered Sylvester second derivative.

    No diagonal-velocity or inverse-mass substitution is made for intrablock
    Hamiltonian derivatives. Trace derivatives commute on smooth subspaces.
    """
    derivative_order = 2


class VelBCP(FormulaProduct):
    r"""V^c mathcal G^{ab}; axes (c,a,b). A weighted trace is nonadditive."""
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), BCP(data_K, **kwargs_formula)], name='VelBCP')

    @property
    def additive(self):
        return False


class VelDerBCP(FormulaProduct):
    r"""V^c mathcal G^{ab:d}; axes (c,a,b,d), nonadditive subspace kernel."""
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), DerBCP(data_K, **kwargs_formula)], name='VelDerBCP')

    @property
    def additive(self):
        return False


class DerYln(Matrix_GenDer_ln):
    r"""Interblock covariant derivative :math:`\overline Y^{abc:d}`."""

    def __init__(self, data_K):
        super().__init__(
            data_K.covariant('Ham', commader=3),
            data_K.covariant('Ham', commader=4),
            data_K.Dcov,
        )
        self.transformTR = transform_ident
        self.transformInv = transform_ident


class Der2W(Formula_ln):
    r"""Second covariant derivative :math:`\overline W^{ab:cd}`."""

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dY = DerYln(data_K)
        self.dW = DerWln(data_K)
        self.W = data_K.covariant('Ham', commader=2)
        self.D = Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.ndim = 4
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        result = self.dY.nn(ik, inn, out)
        result += cached_einsum(
            'mlabd,lnc->mnabcd',
            self.dW.nl(ik, inn, out), self.D.ln(ik, inn, out))
        result += cached_einsum(
            'mlab,lncd->mnabcd',
            self.W.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        result -= cached_einsum(
            'mlcd,lnab->mnabcd',
            self.dD.nl(ik, inn, out), self.W.ln(ik, inn, out))
        result -= cached_einsum(
            'mlc,lnabd->mnabcd',
            self.D.nl(ik, inn, out), self.dW.ln(ik, inn, out))
        return result

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Der2V(Formula_ln):
    r"""Second covariant derivative :math:`\overline V^{a:cd}`."""

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dW = DerWln(data_K)
        self.dV = InvMass(data_K)
        self.D = Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def ln(self, ik, inn, out):
        result = self.dW.ln(ik, inn, out)
        result += cached_einsum(
            'lpad,pnc->lnacd',
            self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        result += cached_einsum(
            'lpa,pncd->lnacd',
            self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        result -= cached_einsum(
            'lmcd,mna->lnacd',
            self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        result -= cached_einsum(
            'lmc,mnad->lnacd',
            self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))
        return result

    def nn(self, ik, inn, out):
        raise NotImplementedError()


class Der4E(Formula_ln):
    r"""Fourth band-energy derivative :math:`\partial_a\partial_b\partial_c\partial_d\varepsilon`.

    The trace over a smooth degenerate group is the fourth derivative of the
    sum of its eigenenergies.  Cartesian axes are ``(a,b,c,d)``.
    """

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddW = Der2W(data_K)
        self.ddV = Der2V(data_K)
        self.D = Dcov(data_K)
        self.dV = InvMass(data_K)
        self.dD = DerDcov(data_K)
        self.ddD = Der2Dcov(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.ndim = 4
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        result = self.ddW.nn(ik, inn, out)
        result += cached_einsum(
            'mlacd,lnb->mnabcd',
            self.ddV.nl(ik, inn, out), self.D.ln(ik, inn, out))
        result += cached_einsum(
            'mlac,lnbd->mnabcd',
            self.dV.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        result += cached_einsum(
            'mlad,lnbc->mnabcd',
            self.dV.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        result += cached_einsum(
            'mla,lnbcd->mnabcd',
            self.V.nl(ik, inn, out), self.ddD.ln(ik, inn, out))
        result -= cached_einsum(
            'mlbcd,lna->mnabcd',
            self.ddD.nl(ik, inn, out), self.V.ln(ik, inn, out))
        result -= cached_einsum(
            'mlbc,lnad->mnabcd',
            self.dD.nl(ik, inn, out), self.dV.ln(ik, inn, out))
        result -= cached_einsum(
            'mlbd,lnac->mnabcd',
            self.dD.nl(ik, inn, out), self.dV.ln(ik, inn, out))
        result -= cached_einsum(
            'mlb,lnacd->mnabcd',
            self.D.nl(ik, inn, out), self.ddV.ln(ik, inn, out))
        return result

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        """Derivatives of disjoint band-group energy sums add."""
        return True


class Der3EVel(FormulaProduct):
    r""":math:`(\partial_a\partial_b\partial_c\varepsilon)(\partial_d\varepsilon)`.

    Cartesian axes are ``(a,b,c,d)``.  The formula is evaluated for each
    band or exactly degenerate group, as required by the Fermi-surface IMQ.
    """

    def __init__(self, data_K, **kwargs_formula):
        super().__init__(
            [Der3E(data_K, **kwargs_formula),
             data_K.covariant('Ham', commader=1)],
            name='Der3EVel',
        )

    @property
    def additive(self):
        """IMQ is a direct sum over bands or exact degeneracy groups."""
        return True
