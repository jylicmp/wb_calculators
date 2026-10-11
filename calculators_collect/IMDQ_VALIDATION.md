# IMD / IMQ implementation and validation

## Definitions and axes

Write

\[
 E_n^a=\partial_a\varepsilon_n,\qquad
 E_n^{ab}=\partial_a\partial_b\varepsilon_n,
\]

and analogously for higher derivatives.  The collected calculators return the
raw Cartesian order used by the formulas:

| calculator | raw kernel | axes | occupation weight |
|---|---|---|---|
| `IMD_FermiSea` | \(E_n^{abc}\) | `(a,b,c)` | \(f_n\) |
| `IMD_FermiSurf` | \(E_n^{ab}E_n^c\) | `(a,b,c)` | \(-f'_n\) |
| `IMQ_FermiSea` | \(E_n^{abcd}\) | `(a,b,c,d)` | \(f_n\) |
| `IMQ_FermiSurf` | \(E_n^{abc}E_n^d\) | `(a,b,c,d)` | \(-f'_n\) |

Periodic integration by parts gives

\[
 \int[dk]f_nE_n^{abc}
 =-\int[dk]f'_nE_n^{ab}E_n^c,
\]

and

\[
 \int[dk]f_nE_n^{abcd}
 =-\int[dk]f'_nE_n^{abc}E_n^d.
\]

Therefore the two representations use the same stored factor in the
WannierBerri convention, because `fder=1` supplies the positive weight
\(-f'_n\).  `factor_IMD2nd` is negative, whereas `factor_IMQ3rd` is positive.
The conventional numerical factor used when symmetrizing the external electric
fields is not embedded in these raw WannierBerri tensors.

## Fourth derivative

`Der3E` and `InvMass` are unchanged upstream WannierBerri formulas.  The new
code retained from the historical modification is the analytic fourth
derivative.  It differentiates the covariant third derivative once more:

- `DerYln` forms \(\overline Y^{abc:d}\) from Hamiltonian comma derivatives
  of orders three and four;
- `Der2W` forms \(\overline W^{ab:cd}\);
- `Der2V` forms the interblock \(\overline V^{a:cd}\);
- `Der4E` applies the product rule to all `Der3E` terms, including both mixed
  `InvMass`–`DerDcov` contractions and the complete upstream `Der2Dcov`.

No finite difference is used during a calculator run.  For a smooth degenerate
group, `trace(Der4E.nn(...))` is the fourth derivative of the sum of the group
eigenenergies and is invariant under unitary rotations inside that group.

## Additivity

The traces of `Der3E` and `Der4E` are additive: derivatives of the energy sum
of disjoint groups add.  The physical Fermi-surface formulas are likewise a
sum over bands or exact degeneracy groups, so `MassVel` and `Der3EVel` must be
evaluated through WannierBerri's `additive=True` path.

This differs from a subspace quantum metric.  Forcing a surface product to
`additive=False` evaluates a product of cumulative occupied-subspace matrices;
off-diagonal blocks then generate cross terms that are absent from
\(\sum_n E_n^{ab}E_n^c\) or \(\sum_n E_n^{abc}E_n^d\).

## SI factors in three dimensions

WannierBerri's uniform-grid integral divides by `nk * cell_volume`, which is
the discrete form of \(\int d^3k/(2\pi)^3\).  Consequently,

- \(\int[dk]E^{abc}\) has units eV;
- \(\int[dk]E^{abcd}\) has units eV Å.

For \(\tau=1\,\mathrm{fs}\), the code uses

\[
 C_{\rm IMD}=-\frac{e^3\tau^2}{\hbar^3}
   \,(1\,\mathrm{eV}\to e\,\mathrm{J}),
\]

and

\[
 C_{\rm IMQ}=\frac{e^4\tau^3}{\hbar^4}
   \,(1\,\mathrm{eV\,\AA}\to e\times10^{-10}\,\mathrm{J\,m}).
\]

The resulting units are `A/V^2` for IMD and `A m/V^3` for IMQ.
`factor_IMD2nd` is bitwise equal to the archived 1.7
`factor_nldrude`, and `factor_IMQ3rd` is bitwise equal to the archived custom
1.7 factor.

## Automated validation

`tests/test_imdq.py` checks:

1. calculator/formula/fder wiring and both SI factors;
2. `Der3E` and `Der4E` against independent high-order finite differences of
   eigenenergies for one-band and two-band active groups;
3. all 24 Cartesian permutations of the fourth derivative;
4. covariance under a random unitary rotation inside an exactly degenerate
   two-band group;
5. compatibility with the real upstream 1.7 and 1.8 base classes.

The full Metric/BCP/IMD/IMQ suite passes 33 tests in each of the two actual
WannierBerri environments.  On the remote Haldane and FeS PythTB models, the
smallest-step independent eigenenergy finite differences gave the following
worst errors over three generic k points and three directions:

| model | derivative | max absolute error | max relative error |
|---|---:|---:|---:|
| Haldane | third | `3.21e-7` | `1.87e-6` |
| Haldane | fourth | `1.83e-5` | `5.32e-5` |
| FeS | third | `3.59e-7` | `2.98e-7` |
| FeS | fourth | `2.44e-5` | `9.78e-6` |

The smooth Fermi--Dirac sea/surface identity was then checked on uniform
meshes with a width of 0.12 eV.  Each entry is the maximum over chemical
potential and every Cartesian component:

| model | mesh | IMD absolute | IMD relative | IMQ absolute | IMQ relative |
|---|---:|---:|---:|---:|---:|
| Haldane | 21×21 | `4.95e-5` | `2.36e-3` | `1.07e-4` | `5.26e-4` |
| Haldane | 41×41 | `1.41e-8` | `6.70e-7` | `3.79e-8` | `1.86e-7` |
| FeS | 21×21 | `1.03e-1` | `1.96` | `1.25e1` | `8.21` |
| FeS | 41×41 | `2.99e-2` | `1.71` | `3.02` | `1.41` |
| FeS | 81×81 | `1.48e-3` | `3.87e-1` | `1.24e-1` | `4.68e-2` |
| FeS | 121×121 | `2.48e-5` | `7.74e-3` | `2.47e-2` | `9.35e-3` |

Haldane is converged by 41×41.  FeS converges more slowly because its
multiband derivatives are sharper, but both identities agree below one per
cent on the 121×121 mesh.  Its IMD signal is forbidden by time-reversal
symmetry, so the reference scale tends to zero and the relative error can
become ill-conditioned; the monotonic absolute error is the primary
convergence measure in that column.

The additivity choice was also tested with the actual WannierBerri `fder=1`
weight, rather than inferred from a step-occupation calculation:

| model | mesh | wrapper vs `additive=True` | IMD false/true relative | IMQ false/true relative |
|---|---:|---:|---:|---:|
| Haldane | 41×41 | `0` | `4.20` | `1.41` |
| FeS | 81×81 | `0` | `2.38` | `1.86` |

Thus the named surface wrappers use the required band/group sum exactly.
Forcing `additive=False` is numerically wrong for both surface products.

The complete campaign is stored under
`/share/home/zhuhaiyuan/NLC/test_wb_calculators_v1.7/imdq`, including the
JSON/NPZ results, Slurm and unit-test logs, source-data plotting script,
PNG/PDF/SVG/TIFF figures, QA reports, final report and SHA256 manifest.
