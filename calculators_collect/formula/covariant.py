"""Metric and BCP additions, checked against notes/formulae_temp_clean.tex.

This is an extension module, not a replacement for upstream covariant.py.
G is the quantum metric; BCP uses calligraphic G.
Cartesian axes: Metric (a,b), DerMetric (a,b,d).
external_terms=False means the internal model only; cross_terms controls
D-A cross terms within the enabled external contribution.
"""
import numpy as np
from wannierberri.utility import cached_einsum
from wannierberri.formula.formula import Formula_ln, FormulaProduct
from wannierberri.formula.covariant import Dcov, DerDcov
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
