"""New BCP derivative helpers; unchanged upstream basic formulae are imported.

Implements the handbook Sylvester rules, eq:sylvester1/2, in Hamiltonian gauge.
Only the A-B gap is inverted. Degeneracies within A or B are allowed.
"""
import numpy as np


def _commutator(x, y):
    return x @ y - y @ x


class _BCPConnections:
    """M and N jets through order two in a frame parallel within A and B.

    Full matrices are used only to implement explicit intra-/inter-space
    products. The returned jets have band axes first, then (a[,d[,e]]).
    This is a new helper, not a copy of upstream Dcov/DerDcov.
    """
    def __init__(self, data_K, order, external_terms):
        self.data = data_K
        self.order = order
        self.ham = [data_K.covariant('Ham', commader=r)
                    for r in range(1, order + 2)]
        self.aa = ([data_K.covariant('AA', commader=r)
                    for r in range(order + 1)] if external_terms else [])

    def evaluate(self, ik, inn, out):
        indices = np.r_[inn, out].astype(int)
        na, nb = len(inn), len(out)
        size = na + nb
        energy = self.data.E_K[ik, indices]
        cross = np.zeros((size, size), dtype=bool)
        cross[:na, na:] = True
        cross[na:, :na] = True
        inverse = self.data.dEig_inv[ik][indices][:, indices]
        if np.any(inverse[cross] == 0):
            raise ValueError('BCP requires a resolved nonzero gap between inn and out; '
                             'keep degenerate bands in the same subspace.')
        inv = np.where(cross, inverse, 0.)
        # Reuse WB's energy-gap cutoff, without silently zeroing cross-gap poles.
        assert np.allclose(inv[cross] * (energy[:, None]-energy[None, :])[cross], 1.)
        def off(x):
            return np.where(cross, x, 0.)
        def diagonal(x):
            return np.where(cross, 0., x)
        def raw(formula):
            return formula.nn(ik, indices, np.array([], dtype=int)).copy()
        v = raw(self.ham[0])
        d0 = -v * inv[:, :, None]
        h1 = np.stack([diagonal(v[:, :, d]) for d in range(3)], axis=-1)
        d1 = np.zeros((size, size, 3, 3), dtype=complex)
        d2 = np.zeros((size, size, 3, 3, 3), dtype=complex)
        if self.order:
            w = raw(self.ham[1])
            v1 = np.empty_like(w)
            for a in range(3):
                for d in range(3):
                    v1[:, :, a, d] = w[:, :, a, d] + _commutator(v[:, :, a], d0[:, :, d])
                    rhs = -v1[:, :, a, d] - _commutator(h1[:, :, d], d0[:, :, a])
                    d1[:, :, a, d] = rhs * inv
        if self.order == 2:
            h2 = np.empty_like(w)
            for d in range(3):
                for e in range(3):
                    h2[:, :, d, e] = diagonal(v1[:, :, d, e])
            w3 = raw(self.ham[2])
            for a in range(3):
                for d in range(3):
                    for e in range(3):
                        v2 = (w3[:, :, a, d, e]
                              + _commutator(w[:, :, a, d], d0[:, :, e])
                              + _commutator(w[:, :, a, e], d0[:, :, d])
                              + _commutator(_commutator(v[:, :, a], d0[:, :, e]), d0[:, :, d])
                              + _commutator(v[:, :, a], d1[:, :, d, e]))
                        rhs = (-v2 - _commutator(h2[:, :, d, e], d0[:, :, a])
                               - _commutator(h1[:, :, d], d1[:, :, a, e])
                               - _commutator(h1[:, :, e], d1[:, :, a, d]))
                        d2[:, :, a, d, e] = rhs * inv
        sources = {'D': [1j*d0, 1j*d1, 1j*d2][:self.order+1]}
        if self.aa:
            a0 = raw(self.aa[0])
            aj = [np.stack([off(a0[:, :, a]) for a in range(3)], axis=-1)]
            if self.order:
                acomma = raw(self.aa[1])
                a1 = np.empty_like(acomma)
                for a in range(3):
                    for d in range(3):
                        a1[:, :, a, d] = off(acomma[:, :, a, d] + _commutator(a0[:, :, a], d0[:, :, d]))
                aj.append(a1)
            if self.order == 2:
                acomma2 = raw(self.aa[2])
                a2 = np.empty_like(acomma2)
                for a in range(3):
                    for d in range(3):
                        for e in range(3):
                            a2[:, :, a, d, e] = off(
                                acomma2[:, :, a, d, e]
                                + _commutator(acomma[:, :, a, d], d0[:, :, e])
                                + _commutator(acomma[:, :, a, e], d0[:, :, d])
                                + _commutator(_commutator(a0[:, :, a], d0[:, :, e]), d0[:, :, d])
                                + _commutator(a0[:, :, a], d1[:, :, d, e]))
                aj.append(a2)
            sources['A'] = aj
        result = {}
        for key, m in sources.items():
            n = [m[0] * inv[:, :, None]]
            if self.order:
                n1 = np.empty_like(m[1])
                for a in range(3):
                    for d in range(3):
                        n1[:, :, a, d] = (m[1][:, :, a, d]
                            - _commutator(h1[:, :, d], n[0][:, :, a])) * inv
                n.append(n1)
            if self.order == 2:
                n2 = np.empty_like(m[2])
                for a in range(3):
                    for d in range(3):
                        for e in range(3):
                            n2[:, :, a, d, e] = (m[2][:, :, a, d, e]
                                - _commutator(h2[:, :, d, e], n[0][:, :, a])
                                - _commutator(h1[:, :, d], n[1][:, :, a, e])
                                - _commutator(h1[:, :, e], n[1][:, :, a, d])) * inv
                n.append(n2)
            result[key] = (m, n)
        return result


def _bcp_product(x, y, order):
    """Ordered derivative of X_AB^a Y_BA^b, with explicit Cartesian axes."""
    if order == 0:
        return np.einsum('mla,lnb->mnab', x[0], y[0])
    if order == 1:
        return (np.einsum('mlad,lnb->mnabd', x[1], y[0])
                + np.einsum('mla,lnbd->mnabd', x[0], y[1]))
    return (np.einsum('mlade,lnb->mnabde', x[2], y[0])
            + np.einsum('mlad,lnbe->mnabde', x[1], y[1])
            + np.einsum('mlae,lnbd->mnabde', x[1], y[1])
            + np.einsum('mla,lnbde->mnabde', x[0], y[2]))
