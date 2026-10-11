#!/usr/bin/env python3
"""Plot Haldane/FeS IMD/IMQ sea-surface validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'axes.linewidth': 0.75,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
    'legend.fontsize': 7.2,
    'legend.frameon': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

SEA = '#2166AC'
SURFACE = '#B2182B'
GRID21 = '#9ECAE1'
GRID41 = '#08519C'
GRID81 = '#08306B'
GRID121 = '#041F4A'


def component_label(index):
    letters = 'xyz'
    return ''.join(letters[i] for i in index)


def select_component(sea, surface):
    scale = np.max(np.maximum(np.abs(sea), np.abs(surface)), axis=0)
    return np.unravel_index(np.argmax(scale), scale.shape)


def tensor_error(sea, surface):
    return np.max(np.abs(sea - surface), axis=tuple(range(1, sea.ndim)))


def plot_model(root, model):
    grids = [n for n in (21, 41, 81, 121)
             if (root / 'results' / model / f'grid_{n}.npz').exists()]
    data = {n: np.load(root / 'results' / model / f'grid_{n}.npz')
            for n in grids}
    finest = max(grids)
    mu = data[finest]['mu']
    pairs = (
        ('Der3E_smooth', 'MassVel_smooth', 'IMD'),
        ('Der4E_smooth', 'Der3EVel_smooth', 'IMQ'),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.75),
                             constrained_layout=True)
    for row, (sea_key, surface_key, quantity) in enumerate(pairs):
        sea = data[finest][sea_key]
        surface = data[finest][surface_key]
        component = select_component(sea, surface)
        selection = (slice(None),) + component
        ax = axes[row, 0]
        ax.plot(mu, sea[selection], color=SEA, lw=1.35, label='Fermi sea')
        ax.plot(mu, surface[selection], color=SURFACE, lw=1.15,
                ls=(0, (4, 2)), label='Fermi surface')
        ax.axhline(0.0, color='0.78', lw=0.55, zorder=0)
        ax.set_ylabel(f'Raw {quantity} kernel')
        ax.set_title(f'{model}: {quantity} component {component_label(component)}')
        if row == 0:
            ax.legend(loc='best')

        ax = axes[row, 1]
        styles = {21: (GRID21, 0.8), 41: (GRID41, 1.0),
                  81: (GRID81, 1.2), 121: (GRID121, 1.4)}
        for n in grids:
            color, width = styles[n]
            difference = tensor_error(data[n][sea_key], data[n][surface_key])
            ax.semilogy(mu, np.maximum(difference, 1e-14),
                        color=color, lw=width, label=f'{n} × {n}')
        ax.set_ylabel('Max |sea − surface|')
        ax.set_title(f'{quantity} full-tensor difference')
        if row == 0:
            lower, upper = ax.get_ylim()
            ax.set_ylim(lower, upper * 100.0)
            ax.legend(loc='upper center', ncol=2)

    for ax in axes[-1]:
        ax.set_xlabel('Chemical potential (eV)')
    for label, ax in zip('abcd', axes.ravel()):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='bottom', ha='right')

    output = root / 'figures' / f'{model}_IMDQ_sea_surface'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(output) + '.alignment.json',
        overlay_svg=str(output) + '.alignment.svg',
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(str(output) + '.svg', bbox_inches='tight')
    fig.savefig(str(output) + '.pdf', bbox_inches='tight')
    fig.savefig(str(output) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(output) + '.tiff', dpi=600, bbox_inches='tight')
    plt.close(fig)
    return {
        'model': model,
        'figure': str(output.relative_to(root)),
        'selected_components': {
            quantity: component_label(select_component(
                data[finest][sea_key], data[finest][surface_key]))
            for sea_key, surface_key, quantity in pairs
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = [plot_model(args.root, model) for model in ('Haldane', 'FeS')]
    (args.root / 'figures' / 'plot_report.json').write_text(
        json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
