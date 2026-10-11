"""IMD/IMQ formula, prefactor and calculator regressions.

Run with::

    python -m unittest calculators_collect.tests.test_imdq -v
"""

import itertools
import math
import unittest

import numpy as np

from wannierberri import factors as upstream_factors
from wannierberri.data_K.data_K import get_transform_Inv, get_transform_TR
from wannierberri.formula.formula import Matrix_ln
from wannierberri.formula.covariant import Dcov

from calculators_collect.calculators import static, tabulate
from calculators_collect.factors import (
    Ang_SI,
    TAU_UNIT,
    elementary_charge,
    factor_IMD2nd,
    factor_IMQ3rd,
    hbar,
)
from calculators_collect.formula import covariant as f
from calculators_collect.system.needed_data import (
    imd_required_matrices,
    imq_required_matrices,
)


def _symmetrize_cartesian(tensor):
    order = tensor.ndim - 2
    result = np.zeros_like(tensor)
    for permutation in itertools.permutations(range(order)):
        result += tensor.transpose((0, 1) + tuple(2 + i for i in permutation))
    return result / math.factorial(order)


def _hermitian_coefficients(seed=1804, nb=4):
    rng = np.random.default_rng(seed)
    result = [np.diag([-3.2, -1.1, 1.6, 4.0]).astype(complex)]
    scales = (0.28, 0.11, 0.045, 0.018)
    for order, scale in enumerate(scales, start=1):
        shape = (nb, nb) + (3,) * order
        value = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        value = 0.5 * (value + value.swapaxes(0, 1).conj())
        result.append(scale * _symmetrize_cartesian(value))
    return result


def _hamiltonian(x, coefficients):
    result = coefficients[0].copy()
    for order, coefficient in enumerate(coefficients[1:], start=1):
        result += np.einsum(
            'ij' + ''.join(chr(97 + i) for i in range(order)) + ','
            + ','.join(chr(97 + i) for i in range(order)) + '->ij',
            coefficient, *([x] * order), optimize=True) / math.factorial(order)
    return result


def _hamiltonian_derivatives(x, coefficients):
    c1, c2, c3, c4 = coefficients[1:]
    return {
        1: (c1 + np.einsum('ijab,b->ija', c2, x)
            + 0.5 * np.einsum('ijabc,b,c->ija', c3, x, x)
            + np.einsum('ijabcd,b,c,d->ija', c4, x, x, x) / 6),
        2: (c2 + np.einsum('ijabc,c->ijab', c3, x)
            + 0.5 * np.einsum('ijabcd,c,d->ijab', c4, x, x)),
        3: c3 + np.einsum('ijabcd,d->ijabc', c4, x),
        4: c4,
    }


class TaylorModel:
    """Fourth-order local Hamiltonian supplied through the real data_K API."""

    def __init__(self, x, coefficients, group_rotation=None):
        self.x = np.asarray(x, dtype=float)
        self.coefficients = coefficients
        energy, eigenvectors = np.linalg.eigh(_hamiltonian(self.x, coefficients))
        if group_rotation is not None:
            eigenvectors = eigenvectors @ group_rotation

        def rotate(matrix):
            return np.einsum(
                'im,ij...,jn->mn...',
                eigenvectors.conj(), matrix, eigenvectors,
                optimize=True)[None]

        self.raw = {
            order: rotate(value)
            for order, value in _hamiltonian_derivatives(self.x, coefficients).items()
        }
        gap = energy[:, None] - energy[None, :]
        gap_inverse = np.zeros_like(gap)
        np.divide(1.0, gap, out=gap_inverse, where=np.abs(gap) > 1e-10)
        self.E_K = energy[None]
        self.dEig_inv = gap_inverse[None]
        self.D_H = -self.raw[1] * gap_inverse[None, :, :, None]
        self.Dcov = Dcov(self)
        self.force_internal_terms_only = False
        self.num_wann = len(energy)
        self.nk = 1
        self.cell_volume = 1.0

    def covariant(self, name, commader=0, gender=0):
        if name != 'Ham':
            raise KeyError(name)
        if gender:
            commader = 1
        return Matrix_ln(
            self.raw[commader],
            transformTR=get_transform_TR(name, commader),
            transformInv=get_transform_Inv(name, commader),
        )


def _directional_finite_difference(function, derivative_order, step):
    nodes = np.arange(-4, 5, dtype=float)
    vandermonde = np.array([nodes**power for power in range(len(nodes))])
    target = np.zeros(len(nodes))
    target[derivative_order] = math.factorial(derivative_order)
    weights = np.linalg.solve(vandermonde, target) / step**derivative_order
    return sum(weight * function(node * step) for node, weight in zip(nodes, weights))


class IMDQTests(unittest.TestCase):
    def setUp(self):
        self.coefficients = _hermitian_coefficients()
        self.x = np.array([0.13, -0.17, 0.09])
        self.data = TaylorModel(self.x, self.coefficients)

    def _parts(self, active):
        active = np.asarray(active, dtype=int)
        return active, np.setdiff1d(np.arange(self.data.num_wann), active)

    def test_factors_match_dimensional_conversion(self):
        expected_imd = -(elementary_charge**3 / hbar**3 * TAU_UNIT**2
                         * elementary_charge)
        expected_imq = (elementary_charge**4 / hbar**4 * TAU_UNIT**3
                        * Ang_SI * elementary_charge)
        np.testing.assert_allclose(factor_IMD2nd, expected_imd, rtol=3e-16)
        np.testing.assert_allclose(factor_IMQ3rd, expected_imq, rtol=3e-16)
        self.assertEqual(factor_IMD2nd, upstream_factors.factor_nldrude)
        # factor_IMQ3rd exists in the archived modified 1.7 environment.
        if hasattr(upstream_factors, 'factor_IMQ3rd'):
            self.assertEqual(factor_IMQ3rd, upstream_factors.factor_IMQ3rd)

    def test_data_requirements(self):
        self.assertEqual(imd_required_matrices(), frozenset({'Ham'}))
        self.assertEqual(imq_required_matrices(), frozenset({'Ham'}))

    def test_calculator_wiring_and_axes(self):
        ef = np.array([-0.2, 0.2])
        cases = (
            (static.IMD_FermiSea, f.Der3E, 0, factor_IMD2nd),
            (static.IMD_FermiSurf, f.MassVel, 1, factor_IMD2nd),
            (static.IMQ_FermiSea, f.Der4E, 0, factor_IMQ3rd),
            (static.IMQ_FermiSurf, f.Der3EVel, 1, factor_IMQ3rd),
        )
        for calculator_class, formula, fder, factor in cases:
            calculator = calculator_class(Efermi=ef)
            self.assertIs(calculator.Formula, formula)
            self.assertEqual(calculator.fder, fder)
            self.assertEqual(calculator.constant_factor, factor)
        self.assertIs(tabulate.IMD().Formula, f.Der3E)
        self.assertIs(tabulate.IMQ().Formula, f.Der4E)
        self.assertEqual(f.Der3E(self.data).ndim, 3)
        self.assertEqual(f.Der4E(self.data).ndim, 4)
        self.assertEqual(f.Der3EVel(self.data).ndim, 4)
        for formula in (f.Der3E, f.MassVel, f.Der4E, f.Der3EVel):
            self.assertTrue(formula(self.data).additive)

    def test_directional_derivatives_against_independent_eigenvalues(self):
        directions = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.2, -0.7, 0.4]),
            np.array([-0.5, 0.3, 0.8]),
        )
        for active in ([0], [0, 1]):
            inn, out = self._parts(active)
            der3 = f.Der3E(self.data).trace(0, inn, out)
            der4 = f.Der4E(self.data).trace(0, inn, out)
            for direction in directions:
                direction = direction / np.linalg.norm(direction)
                reference = lambda displacement: np.linalg.eigvalsh(
                    _hamiltonian(self.x + displacement * direction,
                                 self.coefficients))[active].sum()
                value3 = np.einsum('abc,a,b,c', der3, direction, direction, direction)
                value4 = np.einsum(
                    'abcd,a,b,c,d', der4,
                    direction, direction, direction, direction)
                fd3 = _directional_finite_difference(reference, 3, 0.012)
                fd4 = _directional_finite_difference(reference, 4, 0.018)
                np.testing.assert_allclose(value3, fd3, rtol=3e-6, atol=3e-7)
                np.testing.assert_allclose(value4, fd4, rtol=2e-4, atol=2e-5)

    def test_fourth_derivative_is_permutation_symmetric(self):
        inn, out = self._parts([0, 1])
        value = f.Der4E(self.data).trace(0, inn, out)
        for permutation in itertools.permutations(range(4)):
            np.testing.assert_allclose(
                value, value.transpose(permutation), rtol=2e-10, atol=2e-11)

    def test_energy_derivative_trace_additivity_all_subsets(self):
        all_bands = np.arange(self.data.num_wann)
        for formula_class in (f.Der3E, f.Der4E):
            formula = formula_class(self.data)
            for count in range(1, self.data.num_wann + 1):
                for subset in itertools.combinations(all_bands, count):
                    inn, out = self._parts(subset)
                    group = formula.trace(0, inn, out)
                    singles = sum((
                        formula.trace(0, [band], all_bands[all_bands != band])
                        for band in subset
                    ), np.zeros_like(group))
                    np.testing.assert_allclose(
                        group, singles, rtol=3e-9, atol=3e-10)

    def test_exact_degenerate_group_gauge_covariance(self):
        coefficients = list(_hermitian_coefficients(seed=1818))
        coefficients[0] = np.diag([-1.5, -1.5, 1.8, 4.2]).astype(complex)
        # Keep the group degenerate at x=0; its derivatives may mix the group.
        x = np.zeros(3)
        base = TaylorModel(x, coefficients)
        rng = np.random.default_rng(77)
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        rotation = np.eye(4, dtype=complex)
        rotation[:2, :2] = q
        rotated = TaylorModel(x, coefficients, group_rotation=rotation)
        inn = np.array([0, 1])
        out = np.array([2, 3])
        for formula in (f.Der3E, f.Der4E):
            original = formula(base).nn(0, inn, out)
            transformed = formula(rotated).nn(0, inn, out)
            expected = np.einsum(
                'mi,mn...,nj->ij...', q.conj(), original, q,
                optimize=True)
            np.testing.assert_allclose(
                transformed, expected, rtol=2e-9, atol=2e-10)


if __name__ == '__main__':
    unittest.main()
