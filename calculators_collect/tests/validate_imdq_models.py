#!/usr/bin/env python3
"""Haldane/FeS IMD and IMQ validation for the remote v1.7 campaign.

This script expects ``run_audit.py`` and the original model files on
``PYTHONPATH``.  It writes all numerical results below ``imdq/results``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from wannierberri.calculators.static import StaticCalculator

from calculators_collect.calculators import static as C
from calculators_collect.formula import covariant as F
from run_audit import THRESH, data, groups, integrate_direct, model, parts, wb


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'imdq' / 'results'


def maximum(value):
    return float(np.max(np.abs(value), initial=0.0))


def error(value, reference):
    absolute = maximum(value - reference)
    scale = maximum(reference)
    return {'max_abs': absolute, 'scale': scale,
            'relative_to_scale': absolute / max(scale, 1e-14)}


def _fd_weights(order, step):
    nodes = np.arange(-4, 5, dtype=float)
    matrix = np.array([nodes**power for power in range(len(nodes))])
    target = np.zeros(len(nodes))
    target[order] = math.factorial(order)
    return nodes, np.linalg.solve(matrix, target) / step**order


def directional_energy_derivative(system, reduced_k, direction, order, step):
    """Independent derivative of sorted eigenenergies along Cartesian k."""
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    reduced_direction = direction @ np.linalg.inv(system.recip_lattice)
    nodes, weights = _fd_weights(order, step)
    energies = np.array([
        data(system, 1, np.asarray(reduced_k) + node * step * reduced_direction).E_K[0]
        for node in nodes
    ])
    return np.einsum('s,sn->n', weights, energies)


def point_checks(system, name):
    points = np.array([[0.217, 0.381, 0.0],
                       [0.411, 0.153, 0.0],
                       [0.693, 0.572, 0.0]])
    directions = np.array([[1.0, 0.0, 0.0],
                           [0.3, -0.7, 0.0],
                           [-0.6, 0.2, 0.0]])
    steps = {3: (8e-3, 4e-3, 2e-3), 4: (1.2e-2, 8e-3, 5e-3)}
    report = []
    for reduced_k in points:
        d = data(system, 1, reduced_k)
        band_groups = groups(d.E_K[0])
        row = {'k': reduced_k.tolist(), 'groups': band_groups,
               'minimum_gap': float(np.min(np.diff(d.E_K[0]))), 'checks': []}
        tensors = {3: F.Der3E(d), 4: F.Der4E(d)}
        for direction in directions:
            direction = direction / np.linalg.norm(direction)
            for order, formula in tensors.items():
                exact = []
                for first, last in band_groups:
                    tensor = formula.trace(0, *parts(first, last, d.num_wann))
                    exact.append(np.einsum(
                        ''.join(chr(97 + i) for i in range(order)) + ','
                        + ','.join(chr(97 + i) for i in range(order)),
                        tensor, *([direction] * order), optimize=True))
                exact = np.asarray(exact)
                convergence = []
                for step in steps[order]:
                    band_reference = directional_energy_derivative(
                        system, reduced_k, direction, order, step)
                    reference = np.array([
                        band_reference[first:last].sum()
                        for first, last in band_groups
                    ])
                    convergence.append({'step': step, **error(exact, reference)})
                row['checks'].append({
                    'order': order, 'direction': direction.tolist(),
                    'formula_scale': maximum(exact), 'finite_difference': convergence,
                })
        # Directly verify all Cartesian permutations, including mixed terms.
        for order, formula in tensors.items():
            for first, last in band_groups:
                tensor = formula.trace(0, *parts(first, last, d.num_wann))
                permutation_error = max(
                    maximum(tensor - tensor.transpose(permutation))
                    for permutation in itertools.permutations(range(order)))
                assert permutation_error < 1e-8
        report.append(row)
    return report


def group_values(d, formula_class):
    all_groups = [groups(energy) for energy in d.E_K]
    formula = formula_class(d)
    values = [[
        formula.trace(ik, *parts(first, last, d.num_wann))
        for first, last in band_groups
    ] for ik, band_groups in enumerate(all_groups)]
    return all_groups, values


def native(d, efermi, formula_class, additive):
    class FormulaWithSelectedAdditivity(formula_class):
        @property
        def additive(self):
            return additive
    return StaticCalculator(
        Formula=FormulaWithSelectedAdditivity, fder=0,
        Efermi=efermi, degen_thresh=THRESH)(d).data


def smooth_integral(all_groups, energies, values, mus, temperature, volume,
                    derivative=False):
    result = np.zeros((len(mus),) + np.asarray(values[0][0]).shape)
    tensor_axes = (1,) * (result.ndim - 1)
    for band_groups, energy, point_values in zip(all_groups, energies, values):
        for (first, last), value in zip(band_groups, point_values):
            level = float(np.mean(energy[first:last]))
            occupation = 1.0 / (1.0 + np.exp(np.clip((level - mus) / temperature,
                                                       -700, 700)))
            if derivative:
                occupation = occupation * (1.0 - occupation) / temperature
            result += occupation.reshape((-1,) + tensor_axes) * value
    return result / (len(energies) * volume)


def grid_check(system, name, n):
    start = time.time()
    d = data(system, n)
    efermi = np.linspace(-3.2, 3.2, 257) if name == 'Haldane' \
        else np.linspace(-2.2, 2.2, 257)
    temperature = 0.12
    formulae = {
        'Der3E': F.Der3E,
        'MassVel': F.MassVel,
        'Der4E': F.Der4E,
        'Der3EVel': F.Der3EVel,
    }
    arrays = {'mu': efermi, 'energies': d.E_K, 'kpoints': d.kpoints_all}
    values_by_name = {}
    groups_by_name = None
    report = {'model': name, 'n': n, 'nk': int(d.nk),
              'temperature_eV': temperature, 'formulae': {}}
    for formula_name, formula_class in formulae.items():
        all_groups, values = group_values(d, formula_class)
        groups_by_name = all_groups
        values_by_name[formula_name] = values
        direct = integrate_direct(
            all_groups, d.E_K, values, efermi, d.cell_volume, fder=0)
        additive_true = native(d, efermi, formula_class, True)
        additive_false = native(d, efermi, formula_class, False)
        arrays[formula_name + '_direct'] = direct
        arrays[formula_name + '_additive_true'] = additive_true
        arrays[formula_name + '_additive_false'] = additive_false
        report['formulae'][formula_name] = {
            'additive_true': error(additive_true, direct),
            'additive_false': error(additive_false, direct),
        }
        assert report['formulae'][formula_name]['additive_true']['max_abs'] < 1e-10

    smooth = {}
    for formula_name, values in values_by_name.items():
        smooth[formula_name] = smooth_integral(
            groups_by_name, d.E_K, values, efermi, temperature,
            d.cell_volume,
            derivative=formula_name in ('MassVel', 'Der3EVel'))
        arrays[formula_name + '_smooth'] = smooth[formula_name]
    report['sea_surface'] = {
        'IMD': error(smooth['Der3E'], smooth['MassVel']),
        'IMQ': error(smooth['Der4E'], smooth['Der3EVel']),
    }

    # The named wrappers must reproduce the raw native result when C=1.
    wrappers = {
        'IMD_FermiSea': (C.IMD_FermiSea, 'Der3E'),
        'IMD_FermiSurf': (C.IMD_FermiSurf, 'MassVel'),
        'IMQ_FermiSea': (C.IMQ_FermiSea, 'Der4E'),
        'IMQ_FermiSurf': (C.IMQ_FermiSurf, 'Der3EVel'),
    }
    for wrapper_name, (wrapper_class, formula_name) in wrappers.items():
        fder = 0 if wrapper_name.endswith('Sea') else 1
        expected = integrate_direct(
            groups_by_name, d.E_K, values_by_name[formula_name], efermi,
            d.cell_volume, fder=fder)
        actual = wrapper_class(Efermi=efermi, constant_factor=1.0,
                               degen_thresh=THRESH)(d).data
        arrays[wrapper_name] = actual
        report.setdefault('wrappers', {})[wrapper_name] = error(actual, expected)
        assert report['wrappers'][wrapper_name]['max_abs'] < 1e-10

    report['seconds'] = time.time() - start
    model_dir = OUT / name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_dir / f'grid_{n}.npz', **arrays)
    (model_dir / f'grid_{n}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('Haldane', 'FeS'), required=True)
    parser.add_argument('--grids', nargs='+', type=int, default=(21, 41))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    system = wb.System_PythTB(ptb_model=model(args.model), berry=True)
    checks = point_checks(system, args.model)
    for order, tolerance in ((3, 5e-6), (4, 5e-4)):
        worst = max(
            check['finite_difference'][-1]['max_abs']
            for point in checks for check in point['checks']
            if check['order'] == order)
        assert worst < tolerance, (args.model, order, worst, tolerance)
    model_dir = OUT / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'point_checks.json').write_text(json.dumps(checks, indent=2))
    if args.preflight:
        grid_check(system, args.model, 5)
        return
    summaries = [grid_check(system, args.model, n) for n in args.grids]
    (model_dir / 'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
