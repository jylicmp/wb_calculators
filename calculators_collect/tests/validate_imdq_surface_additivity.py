#!/usr/bin/env python3
"""Check IMD/IMQ product additivity with the actual ``fder=1`` weight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from wannierberri.calculators.static import StaticCalculator

from calculators_collect.calculators import static as C
from calculators_collect.formula import covariant as F
from run_audit import THRESH, data, model, wb


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'imdq' / 'results'


def error(value, reference):
    absolute = float(np.max(np.abs(value - reference), initial=0.0))
    scale = float(np.max(np.abs(reference), initial=0.0))
    return {
        'max_abs': absolute,
        'scale': scale,
        'relative_to_scale': absolute / max(scale, 1e-14),
    }


def forced_surface(data_k, efermi, formula_class, additive):
    class FormulaWithSelectedAdditivity(formula_class):
        @property
        def additive(self):
            return additive

    return StaticCalculator(
        Formula=FormulaWithSelectedAdditivity,
        fder=1,
        Efermi=efermi,
        degen_thresh=THRESH,
    )(data_k).data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('Haldane', 'FeS'), required=True)
    parser.add_argument('--grid', type=int, required=True)
    args = parser.parse_args()

    system = wb.System_PythTB(ptb_model=model(args.model), berry=True)
    data_k = data(system, args.grid)
    efermi = (np.linspace(-3.2, 3.2, 257) if args.model == 'Haldane'
              else np.linspace(-2.2, 2.2, 257))
    cases = {
        'IMD': (F.MassVel, C.IMD_FermiSurf),
        'IMQ': (F.Der3EVel, C.IMQ_FermiSurf),
    }
    report = {'model': args.model, 'n': args.grid, 'nk': int(data_k.nk),
              'weight': 'fder=1 (-df/dE)', 'results': {}}
    for name, (formula_class, wrapper_class) in cases.items():
        additive_true = forced_surface(data_k, efermi, formula_class, True)
        additive_false = forced_surface(data_k, efermi, formula_class, False)
        wrapper = wrapper_class(
            Efermi=efermi, constant_factor=1.0, degen_thresh=THRESH,
        )(data_k).data
        report['results'][name] = {
            'wrapper_vs_additive_true': error(wrapper, additive_true),
            'additive_false_vs_true': error(additive_false, additive_true),
        }
        assert report['results'][name]['wrapper_vs_additive_true']['max_abs'] < 1e-10

    model_dir = OUT / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f'surface_additivity_{args.grid}.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
