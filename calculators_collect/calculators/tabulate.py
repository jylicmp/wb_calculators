"""Metric tabulators reuse upstream per-band / degeneracy-group averaging.

Their rows are not cumulative occupied-subspace differences. A weighted sum
of these rows need not equal the corresponding nonadditive static integral.
"""
from wannierberri.calculators.tabulate import Tabulator
from ..formula import covariant as frml


class Metric(Tabulator):
    """G_n^{ab}; for a degenerate group, Tr_A G^{ab}/len(A). Units: angstrom^2."""
    def __init__(self, **kwargs):
        super().__init__(frml.Metric, **kwargs)


class DerMetric(Tabulator):
    """G_n^{ab:d}, axes (a,b,d); group average at degeneracy. Units: angstrom^3."""
    def __init__(self, **kwargs):
        super().__init__(frml.DerMetric, **kwargs)


class IMD(Tabulator):
    r""":math:`\partial_a\partial_b\partial_c\varepsilon_n`, axes ``(a,b,c)``.

    Exact degeneracies are represented by the trace average of the degenerate
    band group, following the upstream tabulator convention.
    """

    def __init__(self, **kwargs):
        super().__init__(frml.Der3E, **kwargs)


class IMQ(Tabulator):
    r""":math:`\partial_a\partial_b\partial_c\partial_d\varepsilon_n`, axes ``(a,b,c,d)``.

    Exact degeneracies are represented by the trace average of the degenerate
    band group, following the upstream tabulator convention.
    """

    def __init__(self, **kwargs):
        super().__init__(frml.Der4E, **kwargs)


class BCP(Tabulator):
    r"""mathcal G^{ab}, axes (a,b), angstrom^2/eV; degeneracy-group average."""
    def __init__(self, **kwargs):
        super().__init__(frml.BCP, **kwargs)


class DerBCP(Tabulator):
    r"""mathcal G^{ab:d}, axes (a,b,d), angstrom^3/eV; group average."""
    def __init__(self, **kwargs):
        super().__init__(frml.DerBCP, **kwargs)


class BCPQuadrupole(Tabulator):
    r"""mathcal G^{ab:de}, axes (a,b,d,e), angstrom^4/eV; group average."""
    def __init__(self, **kwargs):
        super().__init__(frml.Der2BCP, **kwargs)
