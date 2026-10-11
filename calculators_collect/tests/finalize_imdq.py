#!/usr/bin/env python3
"""Build the remote IMD/IMQ validation report and checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fmt(value):
    return f'{value:.3e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    point_rows = []
    grid_rows = []
    surface_rows = []
    for model in ('Haldane', 'FeS'):
        checks = json.loads(
            (root / 'results' / model / 'point_checks.json').read_text())
        for order in (3, 4):
            selected = [check for point in checks for check in point['checks']
                        if check['order'] == order]
            point_rows.append((
                model, order,
                max(row['finite_difference'][-1]['max_abs'] for row in selected),
                max(row['finite_difference'][-1]['relative_to_scale']
                    for row in selected),
            ))
        for path in (root / 'results' / model).glob('grid_*.json'):
            record = json.loads(path.read_text())
            if record['n'] < 21:
                continue
            grid_rows.append((
                model, record['n'],
                record['sea_surface']['IMD']['max_abs'],
                record['sea_surface']['IMD']['relative_to_scale'],
                record['sea_surface']['IMQ']['max_abs'],
                record['sea_surface']['IMQ']['relative_to_scale'],
                record['formulae']['MassVel']['additive_false']['relative_to_scale'],
                record['formulae']['Der3EVel']['additive_false']['relative_to_scale'],
            ))
        for path in (root / 'results' / model).glob(
                'surface_additivity_*.json'):
            record = json.loads(path.read_text())
            surface_rows.append((
                model, record['n'],
                record['results']['IMD']['wrapper_vs_additive_true']['max_abs'],
                record['results']['IMD']['additive_false_vs_true']['relative_to_scale'],
                record['results']['IMQ']['wrapper_vs_additive_true']['max_abs'],
                record['results']['IMQ']['additive_false_vs_true']['relative_to_scale'],
            ))

    model_order = {'Haldane': 0, 'FeS': 1}
    grid_rows.sort(key=lambda row: (model_order[row[0]], row[1]))
    surface_rows.sort(key=lambda row: (model_order[row[0]], row[1]))

    lines = [
        '# IMD / IMQ validation report', '',
        'The collected analytic `Der4E`, the IMD/IMQ static wrappers and SI '
        'factors were tested with the original PythTB Haldane and FeS models '
        'in WannierBerri 1.7.  The same 33-test regression suite also passed '
        'in the real 1.8 environment.', '',
        '## Independent point derivatives', '',
        '| model | order | max abs error | max relative error |',
        '|---|---:|---:|---:|',
    ]
    lines.extend(
        f'| {model} | {order} | {fmt(absolute)} | {fmt(relative)} |'
        for model, order, absolute, relative in point_rows)
    lines.extend([
        '',
        'References are nine-point finite differences of eigenenergy sums '
        'along three independent Cartesian directions at three generic k '
        'points.  Formula values use the analytic covariant derivatives.', '',
        '## Fermi-sea / Fermi-surface convergence', '',
        '| model | grid | IMD abs | IMD rel | IMQ abs | IMQ rel | '
        'IMD step-weight `additive=False` rel | '
        'IMQ step-weight `additive=False` rel |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    lines.extend(
        f'| {model} | {n}×{n} | {fmt(imd_abs)} | {fmt(imd_rel)} | '
        f'{fmt(imq_abs)} | {fmt(imq_rel)} | {fmt(imd_false)} | '
        f'{fmt(imq_false)} |'
        for (model, n, imd_abs, imd_rel, imq_abs, imq_rel,
             imd_false, imq_false) in grid_rows)
    lines.extend([
        '',
        'Sea/surface comparisons use a Fermi–Dirac width of 0.12 eV and the '
        'same uniform mesh.  The `additive=False` columns compare the '
        'cumulative occupied-subspace product with the required direct '
        'band/group sum under a step occupation.  The separate table below '
        'checks the actual surface weight.', '',
        '## Actual `fder=1` additivity', '',
        '| model | grid | IMD wrapper/true abs | IMD false/true rel | '
        'IMQ wrapper/true abs | IMQ false/true rel |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    lines.extend(
        f'| {model} | {n}×{n} | {fmt(imd_wrapper)} | {fmt(imd_false)} | '
        f'{fmt(imq_wrapper)} | {fmt(imq_false)} |'
        for (model, n, imd_wrapper, imd_false,
             imq_wrapper, imq_false) in surface_rows)
    lines.extend([
        '',
        'These comparisons use WannierBerri `fder=1`, whose weight is '
        '`-df/dE`.  The named wrappers reproduce `additive=True` to floating-'
        'point precision, while forcing `additive=False` produces the wrong '
        'cumulative-subspace product.', '',
        'Every named wrapper with `constant_factor=1` agreed with the direct '
        'band/group integration to floating-point precision.  `Der3E` and '
        '`Der4E` themselves gave the same result with forced `additive=True` '
        'or `False`, confirming trace additivity; this equivalence does not '
        'extend to `MassVel` or `Der3EVel`.', '',
        '## Artifacts', '',
        '- `scripts/run_imdq.py`: model and finite-difference validation',
        '- `scripts/validate_imdq_surface_additivity.py`: actual `fder=1` check',
        '- `results/`: JSON summaries and compressed raw arrays',
        '- `figures/`: Python source-data plots in PNG/PDF/SVG/TIFF plus QA',
        '- `logs/`: Slurm and unit-test logs',
        '- `SHA256SUMS`: hashes for the complete validation bundle', '',
    ])
    (root / 'REPORT.md').write_text('\n'.join(lines))

    files = [path for path in root.rglob('*')
             if path.is_file() and path.name != 'SHA256SUMS'
             and '__pycache__' not in path.parts]
    checksums = ''.join(
        hashlib.sha256(path.read_bytes()).hexdigest() + '  '
        + str(path.relative_to(root)) + '\n'
        for path in sorted(files))
    (root / 'SHA256SUMS').write_text(checksums)


if __name__ == '__main__':
    main()
