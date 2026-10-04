"""Audit archived DerMetric against differentiated Metric on smooth model data.

Requires NumPy and --formula-base pointing to WannierBerri 1.7.0's unmodified
wannierberri/formula/formula.py. Loads actual classes by AST without importing
WannierBerri's optional runtime dependencies. No archived source is modified.
The candidate changes only the four block accessors in the final two products.
"""
import argparse
import ast
import abc
import copy
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'v1.7/wannierberri/formula/covariant.py'

def load_classes(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return selected


def run(base):
    ns = dict(np=np, abc=abc, cached_einsum=np.einsum,
              transform_ident=None, transform_odd=None)
    load_classes(base, ['Formula', 'Formula_ln', 'Matrix_ln', 'Matrix_GenDer_ln'], ns)
    nodes = load_classes(SOURCE, ['DEinv_ln', 'Dcov', 'DerDcov', 'Metric', 'DerMetric'], ns)
    original = ns['DerMetric']
    candidate_node = copy.deepcopy(next(n for n in nodes if n.name == 'DerMetric'))
    nn = next(n for n in candidate_node.body if isinstance(n, ast.FunctionDef) and n.name == 'nn')
    ext = next(n for n in nn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.external_terms')
    products = [n for n in ext.body if isinstance(n, ast.AugAssign)][-2:]
    changes = 0
    for product in products:
        for node in ast.walk(product):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('nl', 'ln'):
                node.func.attr = 'nn'
                changes += 1
    assert changes == 4
    candidate_node.name = 'DerMetricCandidate'
    exec(compile(ast.Module(body=[candidate_node], type_ignores=[]), '<candidate-in-memory>', 'exec'), ns)
    candidate = ns['DerMetricCandidate']
    Matrix, GenDer = ns['Matrix_ln'], ns['Matrix_GenDer_ln']

    class Model:
        def __init__(self, x, h0, v, w, t, c, b):
            # H^W quadratic; physical Wannier frame S=exp(-i f(x) T),
            # f=c.x + x.B.x/2. Thus A^W_a=f_a T and F^W_ab=f_a f_b T^2.
            h = h0 + np.einsum('a,ija->ij', x, v) + .5*np.einsum('a,b,ijab->ij', x, x, w)
            e, u = np.linalg.eigh(h)
            def rotate(z):
                return np.einsum('im,ij...,jn->mn...', u.conj(), z, u)[None]
            grad = c + b @ x
            aa = np.einsum('a,ij->ija', grad, t)
            da = np.einsum('ad,ij->ijad', b, t)
            ff = np.einsum('a,b,ij->ijab', grad, grad, t@t)
            df = (np.einsum('ad,b,ij->ijabd', b, grad, t@t)
                  + np.einsum('a,bd,ij->ijabd', grad, b, t@t))
            self.raw = {('Ham', 1): rotate(v + np.einsum('ijab,b->ija', w, x)),
                        ('Ham', 2): rotate(w), ('AA', 0): rotate(aa), ('AA', 1): rotate(da),
                        ('FF', 0): rotate(ff), ('FF', 1): rotate(df)}
            gap = e[:, None] - e[None, :]
            inv = np.zeros_like(gap)
            np.divide(1., gap, out=inv, where=abs(gap)>1e-7)
            self.E_K, self.dEig_inv = e[None], inv[None]
            self.D_H = -self.raw['Ham', 1]*inv[None, :, :, None]
            self.Dcov = ns['Dcov'](self)
            self.force_internal_terms_only = False
        def covariant(self, name, commader=0, gender=0):
            if gender and name == 'Ham':
                return Matrix(self.raw['Ham', 1])
            if gender:
                return GenDer(self.covariant(name), self.covariant(name, commader=1), self.Dcov)
            return Matrix(self.raw[name, commader])

    def check(parameters, x, inn, external, steps):
        inn = np.asarray(inn)
        out = np.setdiff1d(np.arange(len(parameters[0])), inn)
        data = Model(x, *parameters)
        old = original(data, external_terms=external).trace(0, inn, out)
        new = candidate(data, external_terms=external).trace(0, inn, out)
        errors = []
        for step in steps:
            fd = np.empty((3, 3, 3))
            for d in range(3):
                dx = np.eye(3)[d]*step
                def value(point):
                    return ns['Metric'](Model(point, *parameters), external_terms=external).trace(0, inn, out)
                fd[:, :, d] = (value(x+dx)-value(x-dx))/(2*step)
            errors.append(dict(step=step, original_max_error=float(np.max(abs(old-fd))),
                               candidate_max_error=float(np.max(abs(new-fd)))))
        # Independent reference: covariant_basic tildeFab_d followed by Cartesian symmetrization.
        reference = ns['tildeFab_d'](data, external_terms=external).nn(0, inn, out)
        reference = .5*(reference + reference.swapaxes(2, 3))
        new_matrix = candidate(data, external_terms=external).nn(0, inn, out)
        matrix_error = float(np.max(abs(new_matrix-reference)))
        assert matrix_error < 1e-11
        assert errors[-1]['candidate_max_error'] < 2e-7
        if not external:
            assert errors[-1]['original_max_error'] < 2e-7
        return dict(active=inn.tolist(), external_terms=external,
                    original_xxx=float(old[0, 0, 0]), candidate_xxx=float(new[0, 0, 0]),
                    candidate_vs_basic_matrix_error=matrix_error, finite_difference=errors)

    load_classes(ROOT/'v1.7/wannierberri/formula/covariant_basic.py', ['tildeFab_d'], ns)
    simple = (np.diag([-1., 1.]), np.zeros((2, 2, 3)), np.zeros((2, 2, 3, 3)),
              np.array([[0., 1.], [1., 0.]]), np.zeros(3), np.diag([1., 0., 0.]))
    results = {'two_band': check(simple, np.array([.7, 0., 0.]), [0], True, [1e-3, 1e-4, 1e-5]),
               'two_band_full_active': check(simple, np.array([.7, 0., 0.]), [0, 1], True, [1e-4])}
    assert abs(results['two_band']['original_xxx']) < 1e-12
    assert abs(results['two_band']['candidate_xxx']-1.4) < 1e-12
    rng = np.random.default_rng(1707)
    def herm(shape):
        a = rng.normal(size=shape)+1j*rng.normal(size=shape)
        return .5*(a+a.swapaxes(0, 1).conj())
    w = .13*herm((4, 4, 3, 3)); w = .5*(w+w.swapaxes(2, 3))
    t = .3*herm((4, 4))
    b = np.array([[.4,.1,-.2],[.1,-.3,.15],[-.2,.15,.25]])
    params = (np.diag([-3., -1., 1.4, 3.7]), .25*herm((4,4,3)), w, t,
              np.array([.6,-.3,.5]), b)
    results['four_band'] = [check(params, np.array([.17,-.21,.13]), inn, ext, [1e-2,1e-3,1e-4])
                            for inn in ([0], [0,1]) for ext in (False, True)]
    # Same test at an exact degeneracy within A, separated from B.
    deg = list(params); deg[0] = np.diag([-2., -2., 1.4, 3.7])
    results['degenerate_active_pair'] = check(tuple(deg), np.zeros(3), [0,1], True, [1e-2,1e-3,1e-4])
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formula-base', type=Path, required=True)
    args = parser.parse_args()
    run(args.formula_base)
