import numpy as np
from ..utility import alpha_A, beta_A, cached_einsum, delta_f
from .formula import Formula_ln, Matrix_ln, Matrix_GenDer_ln, FormulaProduct, FormulaSum, DeltaProduct
from ..symmetry.point_symmetry import transform_ident, transform_odd


class Identity(Formula_ln):

    def __init__(self, data_K=None, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 0
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        return np.eye(len(inn))

    def ln(self, ik, inn, out):
        return np.zeros((len(out), len(inn)))


class Eavln(Matrix_ln):
    """ be careful : this is not a covariant matrix"""

    def __init__(self, data_K):
        super().__init__(0.5 * (data_K.E_K[:, :, None] + data_K.E_K[:, None, :]))
        self.ndim = 0
        self.transformTR = transform_ident
        self.transformInv = transform_ident


class DEinv_ln(Matrix_ln):
    """DEinv_ln.matrix[ik, m, n] = 1 / (E_mk - E_nk)"""

    def __init__(self, data_K):
        super().__init__(data_K.dEig_inv)

    def nn(self, ik, inn, out):
        raise NotImplementedError("dEinv_ln should not be called within inner states")


class Dcov(Matrix_ln):

    def __init__(self, data_K):
        super().__init__(data_K.D_H)

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")


class DerDcov(Dcov):

    def __init__(self, data_K):
        self.W = data_K.covariant('Ham', commader=2)
        self.V = data_K.covariant('Ham', gender=1) # data_K.covariant('Ham', gender=1) = data_K.covariant('Ham', commader=1)
        self.D = data_K.Dcov
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        summ = self.W.ln(ik, inn, out)
        tmp = cached_einsum("lpb,pnd->lnbd", self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        tmp = -cached_einsum("lmb,mnd->lnbd", self.D.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None]
        return summ

#
######################################

class Mcov(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 1
        self.D = Dcov(data_K)
        if self.external_terms:
            self.A = data_K.covariant('AA')
    
    def ln(self, ik, inn, out):
        summ = np.zeros((len(out), len(inn), 3), dtype=complex)
        if self.internal_terms:
            summ += 1j * self.D.ln(ik, inn, out)
        if self.external_terms:
            summ += self.A.ln(ik, inn, out)
        return summ
        
    def nn(self, ik, inn, out):
        raise ValueError("Mln should not be called within inner states")


class Ncov(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 1
        self.M = Mcov(data_K, **parameters)
        self.dEinv = DEinv_ln(data_K)
    
    def ln(self, ik, inn, out):
        return self.M.ln(ik, inn, out) * (self.dEinv.ln(ik, inn, out)[:, :, None])
    
    def nn(self, ik, inn, out):
        raise ValueError("Nln should not be called within inner states")


class DV_ln(Formula_ln):
    # V^c_l - V^c_n

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 1
        self.V = data_K.covariant('Ham', commader=1)
    
    def ln(self, ik, inn, out):
        V_l = cached_einsum("llc->lc", self.V.ll(ik, inn, out).real)
        V_n = cached_einsum("nnc->nc", self.V.nn(ik, inn, inn).real)
        deltaV_ln = V_l[:, None, :] - V_n[None, :, :]
        return deltaV_ln
    
    def nn(self, ik, inn, out):
        raise ValueError("DVln should not be called within inner states")


class DerMcov(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 2
        self.dD = DerDcov(data_K)
        if self.external_terms:
            self.dA = data_K.covariant('AA', gender=1)
    
    def ln(self, ik, inn, out):
        summ = np.zeros((len(out), len(inn), 3, 3), dtype=complex)
        if self.internal_terms:
            summ += 1j * self.dD.ln(ik, inn, out)
        if self.external_terms:
            summ += self.dA.ln(ik, inn, out)
        return summ
    
    def nn(self, ik, inn, out):
        raise ValueError("DerMcovln should not be called within inner states")


class DerNcov(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 2
        self.dM = DerMcov(data_K, **parameters)
        self.N = Ncov(data_K, **parameters)
        self.DV = DV_ln(data_K)
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        summ = self.dM.ln(ik, inn, out)
        summ += -cached_einsum('lna,lnd->lnad', self.N.ln(ik, inn, out), self.DV.ln(ik, inn, out))
        summ *= self.dEinv.ln(ik, inn, out)[:, :, None, None]
        return summ
    
    def nn(self, ik, inn, out):
        raise ValueError("DerNcovln should not be called within inner states")


class DInvMass(Formula_ln):
    # V^d:e_l - V^d:e_n

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 2
        self.dV = InvMass(data_K)
    
    def ln(self, ik, inn, out):
        dV_l = cached_einsum("llcd->lcd", self.dV.ll(ik, inn, out))
        dV_n = cached_einsum("nncd->ncd", self.dV.nn(ik, inn, inn))
        DdV_ln = dV_l[:, None, :, :] - dV_n[None, :, :, :]
        return DdV_ln
    
    def nn(self, ik, inn, out):
        raise ValueError("DInvMassln should not be called within inner states")


class Der2Mcov(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 3
        self.ddD = Der2Dcov(data_K)
        if self.external_terms:
            self.ddA = Der2A(data_K)
    
    def ln(self, ik, inn, out):
        summ = np.zeros((len(out), len(inn), 3, 3, 3), dtype=complex)
        if self.internal_terms:
            summ += 1j * self.ddD.ln(ik, inn, out)
        if self.external_terms:
            summ += self.ddA.ln(ik, inn, out)
        return summ

    def nn(self, ik, inn, out):
        raise ValueError("Der2Mcovln should not be called within inner states")


class Der2Ncov(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 3
        self.ddM = Der2Mcov(data_K, **parameters)
        self.N = Ncov(data_K, **parameters)
        self.DInvMass = DInvMass(data_K)
        self.dN = DerNcov(data_K, **parameters)
        self.DV = DV_ln(data_K)
        self.dEinv = DEinv_ln(data_K)
    
    def ln(self, ik, inn, out):
        summ = self.ddM.ln(ik, inn, out)
        summ += -cached_einsum('lna,lnde->lnade', self.N.ln(ik, inn, out), self.DInvMass.ln(ik, inn, out))
        temp = -cached_einsum('lnad,lne->lnade', self.dN.ln(ik, inn, out), self.DV.ln(ik, inn, out))
        summ += temp + temp.swapaxes(3, 4)
        summ *= self.dEinv.ln(ik, inn, out)[:, :, None, None, None]
        return summ
    
    def nn(self, ik, inn, out):
        raise ValueError("Der2Ncovln should not be called within inner states")

##########################################

class Der2Dcov(Formula_ln):

    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.WV = DerWln(data_K)   # \overline{W}^{bc:d}
        self.dV = InvMass(data_K)  # \overline{V}^{b:d}
        self.V = data_K.covariant('Ham', commader=1)
        self.D = Dcov(data_K)
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        summ = self.WV.ln(ik, inn, out)
        summ += cached_einsum("lpbe,pnd->lnbde", self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lpde,pnb->lnbde", self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lpe,pnbd->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("lpd,pnbe->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("lpb,pnde->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += -cached_einsum("lmde,mnb->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmbd,mne->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmbe,mnd->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmb,mnde->lnbde", self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))
        summ += -cached_einsum("lmd,mnbe->lnbde", self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))

        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None, None]
        return summ

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")


class Der2Dcov_chain(Formula_ln):
    '''
    Another implementation of Der2Dcov, but using the chain rule.
    '''
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.WV = DerWln(data_K)   # \overline{W}^{bc:d}
        self.dV = InvMass(data_K)  # \overline{V}^{b:d}
        self.V = data_K.covariant('Ham', commader=1)
        self.W = data_K.covariant('Ham', commader=2)
        self.D = Dcov(data_K)
        self.dEinv = DEinv_ln(data_K)
        

    def ln(self, ik, inn, out):
        summ = self.WV.ln(ik, inn, out)
        summ += cached_einsum("lpbe,pnd->lnbde", self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lpde,pnb->lnbde", self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lpb,pnde->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("lpd,pnbe->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += -cached_einsum("lmbe,mnd->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmde,mnb->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmb,mnde->lnbde", self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))
        summ += -cached_einsum("lmd,mnbe->lnbde", self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))
        Vn = cached_einsum("nnc->nc", self.V.nn(ik, inn, inn).real)
        Vl = cached_einsum("llc->lc", self.V.ll(ik, out, out).real)
        deltaV_ln = Vl[:, None, :] - Vn[None, :, :]
        temp = deltaV_ln * ( self.dEinv.ln(ik, inn, out)[:, :, None] )
        summ += -cached_einsum("lnbd,lne->lnbde", self.W.ln(ik, inn, out), temp)
        prod = cached_einsum("lmd,mnb->lndb", self.D.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += cached_einsum("lndb,lne->lnbde", prod, temp)
        prod = cached_einsum("lmb,mnd->lnbd", self.D.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += cached_einsum("lnbd,lne->lnbde", prod, temp)
        prod = -cached_einsum("lpb,pnd->lnbd", self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lnbd,lne->lnbde", prod, temp)
        prod = -cached_einsum("lpd,pnb->lndb", self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lndb,lne->lnbde", prod, temp)
        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None, None]
        return summ

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")


# TODO Der2A,B,O,H,S can be merged to one class.


class Der2A(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.A = data_K.covariant('AA')
        self.dA = data_K.covariant('AA', gender=1)
        self.Abar_de = Matrix_GenDer_ln(data_K.covariant('AA', commader=1), data_K.covariant('AA', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Abar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.A.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dA.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.A.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dA.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        summ = self.Abar_de.ln(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.ln(ik, inn, out), self.A.nn(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.ln(ik, inn, out), self.dA.nn(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.A.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dA.ll(ik, inn, out), self.D.ln(ik, inn, out))
        return summ


class Der2B(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.B = data_K.covariant('BB')
        self.dB = data_K.covariant('BB', gender=1)
        self.Bbar_de = Matrix_GenDer_ln(data_K.covariant('BB', commader=1), data_K.covariant('BB', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Bbar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.B.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dB.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.B.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dB.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        summ = self.Bbar_de.ln(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.ln(ik, inn, out), self.B.nn(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.ln(ik, inn, out), self.dB.nn(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.B.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dB.ll(ik, inn, out), self.D.ln(ik, inn, out))
        return summ


class Der2O(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.O = data_K.covariant('OO')
        self.dO = data_K.covariant('OO', gender=1)
        self.Obar_de = Matrix_GenDer_ln(data_K.covariant('OO', commader=1), data_K.covariant('OO', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Obar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.O.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dO.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.O.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dO.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Der2H(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.H = data_K.covariant('CC')
        self.dH = data_K.covariant('CC', gender=1)
        self.Hbar_de = Matrix_GenDer_ln(data_K.covariant('CC', commader=1), data_K.covariant('CC', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Hbar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.H.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dH.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.H.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dH.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()



class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=2), data_K.Dcov)
        self.transformTR = transform_ident
        self.transformInv = transform_ident


class DerWln(Matrix_GenDer_ln):
    r""" :math:`\overline{W}^{bc:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', 2), data_K.covariant('Ham', 3), data_K.Dcov)
        self.transformTR = transform_odd
        self.transformInv = transform_odd


#############################
#   Third derivative of  E  #
#############################


class Der3E(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.V = data_K.covariant('Ham', commader=1)
        self.D = data_K.Dcov
        self.dV = InvMass(data_K)
        self.dD = DerDcov(data_K)
        self.dW = DerWln(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        summ += 1 * self.dW.nn(ik, inn, out)
        summ += 1 * cached_einsum("mlac,lnb->mnabc", self.dV.nl(ik, inn, out), self.D.ln(ik, inn, out))
        summ += 1 * cached_einsum("mla,lnbc->mnabc", self.V.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += -1 * cached_einsum("mlbc,lna->mnabc", self.dD.nl(ik, inn, out), self.V.ln(ik, inn, out))
        summ += -1 * cached_einsum("mlb,lnac->mnabc", self.D.nl(ik, inn, out), self.dV.ln(ik, inn, out))

        # TODO: alternatively: add factor 0.5 to first term, remove 4th and 5th, and ad line below.
        # Should give the same, I think, but needs to be tested
        # summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


##############################
#   Fourth derivative of  E  #
#   Add by Jiayu Li          #
##############################

class DerYln(Matrix_GenDer_ln):
    r""" :math:`\overline{Y}^{abc:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', 3), data_K.covariant('Ham', 4), data_K.Dcov)
        self.transformTR = transform_ident
        self.transformInv = transform_ident

class Der2Wnn(Formula_ln):
    r""" :math:`\overline{W}^{ab:cd}`"""

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dY = DerYln(data_K)
        self.dW = DerWln(data_K)
        self.W = data_K.covariant('Ham', commader=2)
        self.D = Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.ndim = 4
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = self.dY.nn(ik, inn, out)
        summ += cached_einsum('mlabd,lnc->mnabcd', self.dW.nl(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum('mlab,lncd->mnabcd', self.W.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ -= cached_einsum('mlcd,lnab->mnabcd', self.dD.nl(ik, inn, out), self.W.ln(ik, inn, out))
        summ -= cached_einsum('mlc,lnabd->mnabcd', self.D.nl(ik, inn, out), self.dW.ln(ik, inn, out))
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Der2Vln(Formula_ln):
    r""" :math:`\overline{V}^{a:cd}`"""

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dW = DerWln(data_K)
        self.dV = InvMass(data_K)
        self.D = Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd
    
    def ln(self, ik, inn, out):
        summ = self.dW.ln(ik, inn, out)
        summ += cached_einsum('lpad,pnc->lnacd', self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum('lpa,pncd->lnacd', self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ -= cached_einsum('lmcd,mna->lnacd', self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ -= cached_einsum('lmc,mnad->lnacd', self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))
        return summ
    
    def nn(self, ik, inn, out):
        raise NotImplementedError()


class Der4E(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddW = Der2Wnn(data_K)
        self.ddV = Der2Vln(data_K)
        self.D = Dcov(data_K)
        self.dV = InvMass(data_K)
        self.dD = DerDcov(data_K)
        self.ddD = Der2Dcov(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.ndim = 4
        self.transformTR = transform_ident
        self.transformInv = transform_ident
    
    def nn(self, ik, inn, out):
        summ = self.ddW.nn(ik, inn, out)
        summ += cached_einsum('mlacd,lnb->mnabcd', self.ddV.nl(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum('mlac,lnbd->mnabcd', self.dV.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum('mlad,lnbc->mnabcd', self.dV.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum('mla,lnbcd->mnabcd', self.V.nl(ik, inn, out), self.ddD.ln(ik, inn, out))
        summ -= cached_einsum('mlbcd,lna->mnabcd', self.ddD.nl(ik, inn, out), self.V.ln(ik, inn, out))
        summ -= cached_einsum('mlbc,lnad->mnabcd', self.dD.nl(ik, inn, out), self.dV.ln(ik, inn, out))
        summ -= cached_einsum('mlbd,lnac->mnabcd', self.dD.nl(ik, inn, out), self.dV.ln(ik, inn, out))
        summ -= cached_einsum('mlb,lnacd->mnabcd', self.D.nl(ik, inn, out), self.ddV.ln(ik, inn, out))
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   Berry curvature    #
########################


class Omega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.O = data_K.covariant('OO')

        self.ndim = 1
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.O.nn(ik, inn, out)
            summ += -1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.A.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.A.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.A.nn(ik, inn, out)[:, :, alpha_A],
                self.A.nn(ik, inn, out)[:, :, beta_A])

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

########################
#   Quantum metric     #
#   Add by Jiayu Li    #
########################

class Metric(Formula_ln):

    def __init__(self, data_K, **parameters):
        r"""  :math:`(1/2) * \tildeF _{ab} + \tildeF _{ba}` """
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.F = data_K.covariant('FF')

        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)

        if self.internal_terms:
            summ += - (1/2) * cached_einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out),
                self.D.ln(ik, inn, out))
        
        if self.external_terms:
            summ += (1/2) * self.F.nn(ik, inn, out)
            summ += (1j/2) * cached_einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out),
                self.A.ln(ik, inn, out))
            summ += (1j/2) * cached_einsum(
                "mla,lnb->mnab",
                self.A.nl(ik, inn, out),
                self.D.ln(ik, inn, out))
            summ += -(1/2) * cached_einsum(
                "mla,lnb->mnab",
                self.A.nn(ik, inn, out),
                self.A.nn(ik, inn, out))

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class tildeFab(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.F = data_K.covariant('FF')
        
        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_ident
    
    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)

        if self.external_terms:
            summ += -1 * np.einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out),
                self.D.ln(ik, inn, out))
        
        if self.external_terms:
            summ += self.F.nn(ik, inn, out)
            summ += 1j * np.einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out),
                self.A.ln(ik, inn, out))
            summ += 1j * np.einsum(
                "mla,lnb->mnab",
                self.A.nl(ik, inn, out),
                self.D.ln(ik, inn, out))
            summ += -1 * np.einsum(
                "mla,lnb->mnab",
                self.A.nn(ik, inn, out),
                self.A.nn(ik, inn, out))
        
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class QuantumMetric(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov
        self.dEinv = DEinv_ln(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')

        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)

        if self.internal_terms:
            summ += -1 * np.einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out),
                self.D.ln(ik, inn, out))
            
        if self.external_terms:
            summ += np.einsum(
                "mla,lnb->mnab",
                self.A.nl(ik, inn, out),
                self.A.ln(ik, inn, out))
            summ += 1j * np.einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out),
                self.A.ln(ik, inn, out))
            summ += 1j * np.einsum(
                "mla,lnb->mnab",
                self.A.nl(ik, inn, out),
                self.D.ln(ik, inn, out))
        
        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

########################
#   Berry connection   #
#     polarizability   #
#   Add by Jiayu Li    #
########################

class BCP(Formula_ln):
    """
    the symmetric version (v2.0)
    """
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov
        self.dEinv = DEinv_ln(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')

        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)

        if self.internal_terms:
            M_nl = 1j * self.D.nl(ik, inn, out)
            M_ln = 1j * self.D.ln(ik, inn, out)

            if self.external_terms:
                M_nl += self.A.nl(ik, inn, out)
                M_ln += self.A.ln(ik, inn, out)
            
            N_nl = M_nl * self.dEinv.nl(ik, inn, out)[:, :, None]
            N_ln = M_ln * self.dEinv.ln(ik, inn, out)[:, :, None]

            summ += cached_einsum(
                "mla,lnb->mnab", N_nl, M_ln)
            summ -= cached_einsum(
                "mla,lnb->mnab", M_nl, N_ln)

        summ += summ.swapaxes(0, 1).conj()
        summ = summ / 4  # a factor 2 for symmetrization of ab and the other 2 for mn
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()



########################
#   derivative of      #
#   Berry curvature    #
########################


class DerOmega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.dO = data_K.covariant('OO', gender=1)
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.external_terms:
            summ += 0.5 * self.dO.nn(ik, inn, out)

        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            if self.internal_terms:
                summ += -1j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dD.ln(ik, inn, out)[:, :, b])
                pass

            if self.external_terms:
                summ += -1 * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum(
                    "mlcd,lnc->mncd",
                    self.dD.nl(ik, inn, out)[:, :, a, :],
                    self.A.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.A.nn(ik, inn, out)[:, :, a],
                    self.dA.nn(ik, inn, out)[:, :, b, :])
                pass

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

# Add by Jiayu Li

########################
#   derivative of      #
#   Metric             #
########################

class DerMetric(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
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
            pass
        
        if self.external_terms:
            summ += 0.5 * self.dF.nn(ik, inn, out)
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
            summ += -0.5 * cached_einsum(
                "mlad,lnb->mnabd",
                self.dA.nl(ik, inn, out),
                self.A.ln(ik, inn, out))
            summ += -0.5 * cached_einsum(
                "mla,lnbd->mnabd",
                self.A.nl(ik, inn, out),
                self.dA.ln(ik, inn, out))
            pass

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

class DerBCP(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.dEinv = DEinv_ln(data_K)
        self.V = data_K.covariant('Ham', gender=1)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd
    
    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.internal_terms:
            M_nl = 1j * self.D.nl(ik, inn, out)
            M_ln = 1j * self.D.ln(ik, inn, out)
            dM_nl = 1j * self.dD.nl(ik, inn, out)
            dM_ln = 1j * self.dD.ln(ik, inn, out)

            if self.external_terms:
                M_nl += self.A.nl(ik, inn, out)
                M_ln += self.A.ln(ik, inn, out)
                dM_nl += self.dA.nl(ik, inn, out)
                dM_ln += self.dA.ln(ik, inn, out)
                pass
            
            N_nl = M_nl * self.dEinv.nl(ik, inn, out)[:, :, None]
            N_ln = M_ln * self.dEinv.ln(ik, inn, out)[:, :, None]
            dN_nl = dM_nl * self.dEinv.nl(ik, inn, out)[:, :, None, None]
            dN_ln = dM_ln * self.dEinv.ln(ik, inn, out)[:, :, None, None]

            Vn = cached_einsum("nnc->nc", self.V.nn(ik, inn, inn).real)
            Vl = cached_einsum("llc->lc", self.V.ll(ik, out, out).real)
            deltaV_nl = Vn[:, None, :] - Vl[None, :, :]
            deltaV_ln = Vl[:, None, :] - Vn[None, :, :]
            # dEinv2_nl = self.dEinv.nl(ik, inn, out) ** 2
            # dEinv2_ln = self.dEinv.ln(ik, inn, out) ** 2
            dN_nl -= cached_einsum("nla,nld->nlad", N_nl, deltaV_nl * (self.dEinv.nl(ik, inn, out)[:, :, None]) )
            dN_ln -= cached_einsum("lna,lnd->lnad", N_ln, deltaV_ln * (self.dEinv.ln(ik, inn, out)[:, :, None]) )
            # dN_nl -= M_nl * dV_nl * (dEinv2_nl[:, :, None])
            # dN_ln -= M_ln * dV_ln * (dEinv2_ln[:, :, None])

            summ += cached_einsum("mlad,lnb->mnabd", dN_nl, M_ln)
            summ += cached_einsum("mla,lnbd->mnabd", N_nl, dM_ln)
            summ -= cached_einsum("mlad,lnb->mnabd", dM_nl, N_ln)
            summ -= cached_einsum("mla,lnbd->mnabd", M_nl, dN_ln)

        summ += summ.swapaxes(0, 1).conj()
        summ = summ / 4 # 2 for symmetrization of ab and the other 2 for mn
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()

class DerBCP_test(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.M = Mcov(data_K)
        self.dM = DerMcov(data_K)
        self.N = Ncov(data_K)
        self.dN = DerNcov(data_K)
        self.dEinv = DEinv_ln(data_K)

        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd
    
    def nn(self, ik, inn, out):
        summ = cached_einsum('mlad,lnb->mnabd', self.dN.nl(ik, inn, out), self.M.ln(ik, inn, out))
        summ += cached_einsum('mla,lnbd->mnabd', self.N.nl(ik, inn, out), self.dM.ln(ik, inn, out))
        summ += -cached_einsum('mlad,lnb->mnabd', self.dM.nl(ik, inn, out), self.N.ln(ik, inn, out))
        summ += -cached_einsum('mla,lnbd->mnabd', self.M.nl(ik, inn, out), self.dN.ln(ik, inn, out))

        summ += summ.swapaxes(0, 1).conj()
        summ = summ / 4 # 2 for symmetrization of ab and the other 2 for mn
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()



###############################
###  second derivative of  ####
###  Berry curvature       ####
###############################


class Der2Omega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddD = Der2Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.ddA = Der2A(data_K)
            self.ddO = Der2O(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.external_terms:
            summ += 0.5 * self.ddO.nn(ik, inn, out)

        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            if self.internal_terms:
                summ += -1j * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a], self.dD.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a], self.ddD.ln(ik, inn, out)[:, :, b])
                pass

            if self.external_terms:
                summ += -1 * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a], self.dA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a], self.ddA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum("mlcde,lnc->mncde", self.ddD.nl(ik, inn, out)[:, :, a, :],
                        self.A.ln(ik, inn, out)[:, :, b])
                summ += -1 * s * cached_einsum("mlcd,lnce->mncde", self.dD.nl(ik, inn, out)[:, :, a, :],
                        self.dA.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mlce,lncd->mncde", self.dA.nn(ik, inn, out)[:, :, a], self.dA.nn(ik, inn, out)[:, :, b, :])
                summ += -1j * s * cached_einsum("mlc,lncde->mncde", self.A.nn(ik, inn, out)[:, :, a], self.ddA.nn(ik, inn, out)[:, :, b, :])
                pass

        summ += summ.swapaxes(0, 1).conj()
        return summ


    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Der2Omega_chain(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddD = Der2Dcov_chain(data_K)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.ddA = Der2A(data_K)
            self.ddO = Der2O(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.external_terms:
            summ += 0.5 * self.ddO.nn(ik, inn, out)

        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            if self.internal_terms:
                summ += -1j * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a], self.dD.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a], self.ddD.ln(ik, inn, out)[:, :, b])
                pass

            if self.external_terms:
                summ += -1 * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a], self.dA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a], self.ddA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum("mlcde,lnc->mncde", self.ddD.nl(ik, inn, out)[:, :, a, :],
                        self.A.ln(ik, inn, out)[:, :, b])
                summ += -1 * s * cached_einsum("mlcd,lnce->mncde", self.dD.nl(ik, inn, out)[:, :, a, :],
                        self.dA.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mlce,lncd->mncde", self.dA.nn(ik, inn, out)[:, :, a], self.dA.nn(ik, inn, out)[:, :, b, :])
                summ += -1j * s * cached_einsum("mlc,lncde->mncde", self.A.nn(ik, inn, out)[:, :, a], self.ddA.nn(ik, inn, out)[:, :, b, :])
                pass

        summ += summ.swapaxes(0, 1).conj()
        return summ


    def ln(self, ik, inn, out):
        raise NotImplementedError()

# Add by Jiayu Li
########################
#   second derivative of  #
#   BCP                   #
########################


class Der2BCP(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddD = Der2Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.dEinv = DEinv_ln(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.dV = InvMass(data_K)
        # self.dV = data_K.covariant('Ham', commader=2)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.ddA = Der2A(data_K)
        
        self.ndim = 4
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3, 3), dtype=complex)
        if self.internal_terms:
            M_nl = 1j * self.D.nl(ik, inn, out)
            M_ln = 1j * self.D.ln(ik, inn, out)
            dM_nl = 1j * self.dD.nl(ik, inn, out)
            dM_ln = 1j * self.dD.ln(ik, inn, out)
            ddM_nl = 1j * self.ddD.nl(ik, inn, out)
            ddM_ln = 1j * self.ddD.ln(ik, inn, out)

            if self.external_terms:
                M_nl += self.A.nl(ik, inn, out)
                M_ln += self.A.ln(ik, inn, out)
                dM_nl += self.dA.nl(ik, inn, out)
                dM_ln += self.dA.ln(ik, inn, out)
                ddM_nl += self.ddA.nl(ik, inn, out)
                ddM_ln += self.ddA.ln(ik, inn, out)
                pass
            
            N_nl = M_nl * self.dEinv.nl(ik, inn, out)[:, :, None]
            N_ln = M_ln * self.dEinv.ln(ik, inn, out)[:, :, None]
            dN_nl = dM_nl * self.dEinv.nl(ik, inn, out)[:, :, None, None]
            dN_ln = dM_ln * self.dEinv.ln(ik, inn, out)[:, :, None, None]
            ddN_nl = ddM_nl * self.dEinv.nl(ik, inn, out)[:, :, None, None, None]
            ddN_ln = ddM_ln * self.dEinv.ln(ik, inn, out)[:, :, None, None, None]

            Vn = cached_einsum("nnc->nc", self.V.nn(ik, inn, inn))
            Vl = cached_einsum("llc->lc", self.V.ll(ik, out, out))
            deltaV_nl = Vn[:, None, :] - Vl[None, :, :]
            deltaV_ln = Vl[:, None, :] - Vn[None, :, :]
            dVn = cached_einsum("nncd->ncd", self.dV.nn(ik, inn, inn))
            dVl = cached_einsum("llcd->lcd", self.dV.ll(ik, out, out))
            deltadV_nl = dVn[:, None, :, :] - dVl[None, :, :, :]
            deltadV_ln = dVl[:, None, :, :] - dVn[None, :, :, :]

            dN_nl -= cached_einsum("nla,nld->nlad", N_nl, deltaV_nl * (self.dEinv.nl(ik, inn, out)[:, :, None]) )
            dN_ln -= cached_einsum("lna,lnd->lnad", N_ln, deltaV_ln * (self.dEinv.ln(ik, inn, out)[:, :, None]) )

            ddN_nl -= cached_einsum("nlae,nld->nlade", dN_nl, deltaV_nl * (self.dEinv.nl(ik, inn, out)[:, :, None]))
            ddN_ln -= cached_einsum("lnae,lnd->lnade", dN_ln, deltaV_ln * (self.dEinv.ln(ik, inn, out)[:, :, None]))
            ddN_nl -= cached_einsum("nlad,nle->nlade", dN_nl, deltaV_nl * (self.dEinv.nl(ik, inn, out)[:, :, None]))
            ddN_ln -= cached_einsum("lnad,lne->lnade", dN_ln, deltaV_ln * (self.dEinv.ln(ik, inn, out)[:, :, None]))
            ddN_nl -= cached_einsum("nla,nlde->nlade", N_nl, deltadV_nl * (self.dEinv.nl(ik, inn, out)[:, :, None, None]))
            ddN_ln -= cached_einsum("lna,lnde->lnade", N_ln, deltadV_ln * (self.dEinv.ln(ik, inn, out)[:, :, None, None]))

            summ += cached_einsum("mlade,lnb->mnabde", ddN_nl, M_ln)
            summ += cached_einsum("mlad,lnbe->mnabde", dN_nl, dM_ln)
            summ += cached_einsum("mlae,lnbd->mnabde", dN_nl, dM_ln)
            summ += cached_einsum("mla,lnbde->mnabde", N_nl, ddM_ln)
            summ -= cached_einsum("mlade,lnb->mnabde", ddM_nl, N_ln)
            summ -= cached_einsum("mlad,lnbe->mnabde", dM_nl, dN_ln)
            summ -= cached_einsum("mlae,lnbd->mnabde", dM_nl, dN_ln)
            summ -= cached_einsum("mla,lnbde->mnabde", M_nl, ddN_ln)

        summ += summ.swapaxes(0, 1).conj()
        summ = summ / 4
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()

class Der2BCP_test(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.M = Mcov(data_K, **parameters)
        self.N = Ncov(data_K, **parameters)
        self.dM = DerMcov(data_K, **parameters)
        self.dN = DerNcov(data_K, **parameters)
        self.ddM = Der2Mcov(data_K, **parameters)
        self.ddN = Der2Ncov(data_K, **parameters)

        self.ndim = 4
        self.transformTR = transform_ident
        self.transformInv = transform_ident
    
    def nn(self, ik, inn, out):
        summ = cached_einsum('mlade,lnb->mnabde', self.ddN.nl(ik, inn, out), self.M.ln(ik, inn, out))
        summ += cached_einsum('mlad,lnbe->mnabde', self.dN.nl(ik, inn, out), self.dM.ln(ik, inn, out))
        summ += cached_einsum('mlae,lnbd->mnabde', self.dN.nl(ik, inn, out), self.dM.ln(ik, inn, out))
        summ += cached_einsum('mla,lnbde->mnabde', self.N.nl(ik, inn, out), self.ddM.ln(ik, inn, out))
        summ += -cached_einsum('mlade,lnb->mnabde', self.ddM.nl(ik, inn, out), self.N.ln(ik, inn, out))
        summ += -cached_einsum('mlad,lnbe->mnabde', self.dM.nl(ik, inn, out), self.dN.ln(ik, inn, out))
        summ += -cached_einsum('mlae,lnbd->mnabde', self.dM.nl(ik, inn, out), self.dN.ln(ik, inn, out))
        summ += -cached_einsum('mla,lnbde->mnabde', self.M.nl(ik, inn, out), self.ddN.ln(ik, inn, out))

        summ /= 4
        summ += summ.swapaxes(0, 1).conj()
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Hamiltonian(Matrix_ln):

    def __init__(self, data_K):
        v = data_K.covariant('Ham', gender=0)
        self.__dict__.update(v.__dict__)


class Velocity(Matrix_ln):

    def __init__(self, data_K, external_terms=False):
        v = data_K.covariant('Ham', gender=1)
        self.__dict__.update(v.__dict__)
        if external_terms:
            self.matrix = self.matrix + 1j * data_K.Xbar('AA') * (
                data_K.E_K[:, :, None, None] - data_K.E_K[:, None, :, None])


class Spin(Matrix_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS')
        self.__dict__.update(s.__dict__)


class DerSpin(Matrix_GenDer_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS', gender=1)
        self.__dict__.update(s.__dict__)


class Der2Spin(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.S = data_K.covariant('SS')
        self.dS = data_K.covariant('SS', gender=1)
        self.Sbar_de = Matrix_GenDer_ln(data_K.covariant('SS', commader=1), data_K.covariant('SS', commader=2),
                    Dcov(data_K))
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = self.Sbar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.S.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dS.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.S.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dS.nl(ik, inn, out), self.D.ln(ik, inn, out))

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   orbital moment     #
########################


class Morb_H(Formula_ln):

    def __init__(self, data_K, **parameters):
        r"""  :math:`\varepcilon_{abc} \langle \partial_a u | H | \partial_b \rangle` """
        super().__init__(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.B = data_K.covariant('BB')
            self.C = data_K.covariant('CC')
        self.D = data_K.Dcov
        self.E = data_K.E_K
        self.ndim = 1
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A] * self.E[ik][out][None, :, None],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.C.nn(ik, inn, out)
            summ += -1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.B.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.B.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.A.nn(ik, inn, out)[:, :, alpha_A] * self.E[ik][inn][None, :, None],
                self.A.nn(ik, inn, out)[:, :, beta_A])
        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Morb_Hpm(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        r""" Morb_H  +- (En+Em)/2 * Omega """
        super().__init__(data_K, **parameters)
        self.H = Morb_H(data_K, **parameters)
        self.sign = sign
        if self.sign != 0:
            self.O = Omega(data_K, **parameters)
            self.Eav = Eavln(data_K)
        self.ndim = 1
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        res = self.H.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * self.Eav.nn(ik, inn, out)[:, :, None] * self.O.nn(ik, inn, out)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class morb(Morb_Hpm):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)


########################
#   derivative of      #
#   orbital moment     #
########################

class DerMorb_H(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.E = data_K.E_K
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB', gender=1)
            self.dH = data_K.covariant('CC', gender=1)
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -2j * cached_einsum(
                "mpc,pld,lnc->mncd",
                self.D.nl(ik, inn, out)[:, :, alpha_A], self.V.ll(ik, inn, out),
                self.D.ln(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.E[ik][out][:, None, None, None] * self.dD.ln(ik, inn, out)[:, :, b])
        if self.external_terms:
            summ += 1 * self.dH.nn(ik, inn, out)
            summ += -2j * cached_einsum(
                "mpc,pld,lnc->mncd",
                self.A.nn(ik, inn, out)[:, :, alpha_A], self.V.nn(ik, inn, out),
                self.A.nn(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.A.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None],
                    self.dA.nn(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum(
                    "mlc,lncd->mncd", (self.B.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2).conj(),
                    self.dD.ln(ik, inn, out)[:, :, b, :])

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class DerMorb(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        super().__init__(data_K, **parameters)
        self.dermorb_H = DerMorb_H(data_K, **parameters)
        self.sign = sign
        if self.sign != 0:
            self.V = data_K.covariant('Ham', commader=1)
            self.dO = DerOmega(data_K, **parameters)
            self.O = Omega(data_K, **parameters)
            self.E = data_K.E_K
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        res = self.dermorb_H.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * cached_einsum("mlc,lnd->mncd", self.O.nn(ik, inn, out), self.V.nn(ik, inn, out))
            res += self.sign * self.E[ik][inn][:, None, None, None] * self.dO.nn(ik, inn, out)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Dermorb(DerMorb):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)


##############################
###  second derivative of ####
###   orbital moment      ####
##############################

class Der2Morb_H(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddD = Der2Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.dV = InvMass(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.E = data_K.E_K
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.ddA = Der2A(data_K)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB', gender=1)
            self.ddB = Der2B(data_K)
            self.dH = data_K.covariant('CC', gender=1)
            self.ddH = Der2H(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident
    # TODO merge term if possible.

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -2j * cached_einsum("mpc,plde,lnc->mncde", self.D.nl(ik, inn, out)[:, :, alpha_A],
                    self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -1j * s * cached_einsum("mpce,pld,lnc->mncde", self.dD.nl(ik, inn, out)[:, :, a, :],
                        self.V.ll(ik, inn, out), self.D.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mpc,pld,lnce->mncde", self.D.nl(ik, inn, out)[:, :, a],
                        self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out)[:, :, b, :])
                summ += -2j * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a, :],
                    self.E[ik][out][:, None, None, None] * self.dD.ln(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a],
                    self.E[ik][out][:, None, None, None, None] * self.ddD.ln(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mpc,ple,lncd->mncde", self.D.nl(ik, inn, out)[:, :, a],
                    self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out)[:, :, b])
        if self.external_terms:
            summ += 1 * self.ddH.nn(ik, inn, out)
            summ += -2j * cached_einsum("mpc,plde,lnc->mncde", self.A.nn(ik, inn, out)[:, :, alpha_A],
                    self.dV.nn(ik, inn, out), self.A.nn(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -1j * cached_einsum("mpce,pld,lnc->mncde", self.dA.nn(ik, inn, out)[:, :, a],
                        self.V.nn(ik, inn, out), self.A.nn(ik, inn, out)[:, :, b])
                summ += -1j * cached_einsum("mpc,pld,lnce->mncde", self.A.nn(ik, inn, out)[:, :, a],
                        self.V.nn(ik, inn, out), self.dA.nn(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlce,lncd->mncde",
                        self.dA.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None, None], self.dA.nn(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlc,lncde->mncde",
                        self.A.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None], self.ddA.nn(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlc,ple,lncd->mncde", self.A.nn(ik, inn, out)[:, :, a],
                        self.V.nn(ik, inn, out), self.dA.nn(ik, inn, out)[:, :, b])
                summ += -2 * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a],
                        self.dB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a],
                        self.ddB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum("mlce,lncd->mncde", (self.dB.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2, 3).conj(),
                        self.dD.ln(ik, inn, out)[:, :, b])
                summ += -2 * s * cached_einsum("mlc,lncde->mncde", (self.B.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2).conj(),
                        self.ddD.ln(ik, inn, out)[:, :, b])

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Der2Morb(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        super().__init__(data_K, **parameters)
        self.der2morb_H = Der2Morb_H(data_K, **parameters)
        self.sign = sign
        if self.sign != 0:
            self.V = data_K.covariant('Ham', commader=1)
            self.dV = InvMass(data_K)
            self.dO = DerOmega(data_K, **parameters)
            self.ddO = Der2Omega(data_K, **parameters)
            self.O = Omega(data_K, **parameters)
            self.E = data_K.E_K
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        res = self.der2morb_H.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * cached_einsum("mlce,lnd->mncde", self.dO.nn(ik, inn, out), self.V.nn(ik, inn, out))
            res += self.sign * cached_einsum("mlc,lnde->mncde", self.O.nn(ik, inn, out), self.dV.nn(ik, inn, out))
            res += self.sign * cached_einsum("mle,lncd->mncde", self.V.nn(ik, inn, out), self.dO.nn(ik, inn, out))
            res += self.sign * self.E[ik][inn][:, None, None, None, None] * self.ddO.nn(ik, inn, out)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Der2morb(Der2Morb):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)


# Add by Jiayu Li
#########################################
#   orbital magnetic quadrupole moment  #
#########################################

class Qorb_IMD(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.Der3E = Der3E(data_K, **parameters)
        self.ndim = 2
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            summ += (1/12) * s * self.Der3E.nn(ik, inn, out)[:, :, :, a, b]
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()

class Qorb_QMD(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dg = DerMetric(data_K, **parameters)
        self.ndim = 2
        self.transformTR = transform_odd
        self.transformInv = transform_odd
        
    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            summ += (1/3) * s * self.dg.nn(ik, inn, out)[:, :, :, a, b]
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()
    
    @property
    def additive(self):
        return False

class Qorb_QMDen(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dg_res = Qorb_QMD(data_K, **parameters)
        self.Eav = Eavln(data_K)
        self.ndim = 2
        self.transformTR = transform_odd
        self.transformInv = transform_odd
        
    @property
    def additive(self):
        return False
    
    def nn(self, ik, inn, out):
        res = self.dg_res.nn(ik, inn, out)
        res *= self.Eav.nn(ik, inn, out)[:, :, None, None]
        return res
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()
    

## testing 
## ndim = 3, Fermi sea

class Qorb_QMD_ndim3(Formula_ln):
    """Using Gao-Xiao convention
    """
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dg = DerMetric(data_K, **parameters)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd
        
    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        summ += (-1/3) *self.dg.nn(ik, inn, out)
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()
    
    @property
    def additive(self):
        return False

class Qorb_QMDen_ndim3(Formula_ln):
    """Using Gao-Xiao convention
    """
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dg_res = Qorb_QMD_ndim3(data_K, **parameters)
        self.Eav = Eavln(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd
        
    @property
    def additive(self):
        return False
    
    def nn(self, ik, inn, out):
        res = self.dg_res.nn(ik, inn, out)
        res *= self.Eav.nn(ik, inn, out)[:, :, None, None, None]
        return res
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()

class Qorb_IMD_ndim3(Formula_ln):
    r""":math:`Q_IMD_{ilj} = \partial_i \langle u_n | \partial_{lj} H | u_n \rangle`
    """
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        # DerWln :math:`\overline{W}^{lj:i}`
        self.DerWln = DerWln(data_K, **parameters)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd
    
    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        temp = (-1/12) * self.DerWln.nn(ik, inn, out) # mnlji
        summ += cached_einsum('mnlji->mnilj', temp) # mnlji -> mnilj
        # summ += np.transpose(temp, axes=(0, 1, 4, 2, 3)) # mnlji -> mnilj
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

class Qorb_MassVel_ndim3(Formula_ln):
    r""":math:`Q_IMD_{ilj} = - \langle u_n | \partial_{lj} H | u_n \rangle * v_n_i`
    """
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.MassVel = MassVel(data_K, **parameters)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd
    
    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        temp = (1/12) * self.MassVel.nn(ik, inn, out) # mnlji
        summ += cached_einsum('mnlji->mnilj', temp) # mnlji -> mnilj
        # summ += np.transpose(temp, axes=(0, 1, 4, 2, 3))
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()
        


########################
#   spin transport     #
########################

def _spin_velocity_einsum_opt(C, A, B):
    # Optimized version of C += cached_einsum('knls,klma->knmas', A, B). Used in shc_B_H.
    nk = C.shape[0]
    nw = C.shape[1]
    for ik in range(nk):
        # Performing C[ik] += cached_einsum('nls,lma->nmas', A[ik], B[ik])
        tmp_a = np.swapaxes(A[ik], 1, 2)  # nls -> nsl
        tmp_a = np.reshape(tmp_a, (nw * 3, nw))  # nsl -> (ns)l
        tmp_b = np.reshape(B[ik], (nw, nw * 3))  # lma -> l(ma)
        tmp_c = tmp_a @ tmp_b  # (ns)l, l(ma) -> (ns)(ma)
        tmp_c = np.reshape(tmp_c, (nw, 3, nw, 3))  # (ns)(ma) -> nsma
        C[ik] += np.transpose(tmp_c, (0, 2, 3, 1))  # nsma -> nmas


class SpinVelocity(Matrix_ln):
    """spin current matrix elements. SpinVelocity.matrix[ik, m, n, a, s] = <u_mk|{v^a S^s}|u_nk> / 2"""

    def __init__(self, data_K, spin_current_type, external_terms=True):
        if spin_current_type == "simple":
            # tight-binding case
            super().__init__(self._J_H_simple(data_K, external_terms=external_terms))
        elif spin_current_type == "qiao":
            # J. Qiao et al PRB (2018)
            super().__init__(self._J_H_qiao(data_K, external_terms=external_terms))
        elif spin_current_type == "ryoo":
            # J. H. Ryoo et al PRB (2019)
            super().__init__(self._J_H_ryoo(data_K, external_terms=external_terms))
        else:
            raise ValueError(f"spin_current_type must be `qiao` or `ryoo` or `simple`, not {spin_current_type}")
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def _J_H_simple(self, data_K, external_terms=True):
        # Spin current operator, J. Qiao et al PRB (2019)
        # J_H[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        S = data_K.Xbar('SS')
        V = Velocity(data_K, external_terms=external_terms).matrix
        J = cached_einsum("klms,kmna->klnas", S, V)
        return (J + J.swapaxes(1, 2).conj()) / 2

    def _J_H_qiao(self, data_K, external_terms=True):
        if not external_terms:
            raise NotImplementedError(
                "spin Hall qiao without external terms is not implemented yet. Use `SHC_type='simple'`")
        # Spin current operator, J. Qiao et al PRB (2019)
        # J_H_qiao[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        SS_H = data_K.Xbar('SS')
        SH_H = data_K.Xbar("SH")
        shc_K_H = -1j * data_K.Xbar("SR")
        _spin_velocity_einsum_opt(shc_K_H, SS_H, data_K.D_H)
        shc_L_H = -1j * data_K._R_to_k_H(data_K.get_R_mat('SHR'), hermitian=False)
        _spin_velocity_einsum_opt(shc_L_H, SH_H, data_K.D_H)
        J = (
            data_K.delE_K[:, None, :, :, None] * SS_H[:, :, :, None, :] +
            data_K.E_K[:, None, :, None, None] * shc_K_H[:, :, :, :, :] - shc_L_H)
        return (J + J.swapaxes(1, 2).conj()) / 2

    def _J_H_ryoo(self, data_K, external_terms=True):
        if not external_terms:
            raise NotImplementedError(
                "spin Hall ryoo without external terms is not implemented yet. Use `SHC_type='simple'`")
        # Spin current operator, J. H. Ryoo et al PRB (2019)
        # J_H_ryoo[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        SA_H = data_K.Xbar("SA")
        SHA_H = data_K.Xbar("SHA")
        J = -1j * (data_K.E_K[:, None, :, None, None] * SA_H - SHA_H)
        _spin_velocity_einsum_opt(J, data_K.Xbar('SS'), data_K.Xbar('Ham', 1))
        return (J + J.swapaxes(1, 2).conj()) / 2


class SpinOmega(Formula_ln):
    """spin Berry curvature"""

    def __init__(self, data_K, spin_current_type="ryoo", **parameters):
        super().__init__(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
        self.D = data_K.Dcov
        self.J = SpinVelocity(data_K, spin_current_type, external_terms=self.external_terms)
        self.dEinv = DEinv_ln(data_K)
        self.ndim = 3
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)

        # v_over_de[l,n,b] = v[l,n,b] / (e[n] - e[l]) = D[l,n,b] - 1j * A[l,n,b]
        v_over_de = self.D.ln(ik, inn, out)
        if self.external_terms:
            v_over_de += - 1j * self.A.ln(ik, inn, out)

        # j_over_de[m,l,a,s] = j[m,l,a,s] / (e[m] - e[l])
        j_over_de = self.J.nl(ik, inn, out) * self.dEinv.nl(ik, inn, out)[:, :, None, None]

        summ += -2 * cached_einsum("mlas,lnb->mnabs", j_over_de, v_over_de).imag

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


####################################
#                                  #
#    Some Prooducts                #
#                                  #
####################################

class VelOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Omega(data_K, **kwargs_formula)], name='VelOmega')


class VelHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Morb_Hpm(data_K, sign=+1, **kwargs_formula)],
                         name='VelHplus')


class VelSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Spin(data_K)], name='VelSpin')


class VelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1)], name='VelVel')

# add by Jiayu Li
class VelMetric(FormulaProduct):
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Metric(data_K, **kwargs_formula)], name='VelMetric')

class MetricVel(FormulaProduct):
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Metric(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='MetricVel')

class VelDerMetric(FormulaProduct):
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), DerMetric(data_K, **kwargs_formula)], name='VelDerMetric')

class VelDerOmega(FormulaProduct):
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), DerOmega(data_K, **kwargs_formula)], name='VelDerOmega')

class VelBCP(FormulaProduct):
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), BCP(data_K, **kwargs_formula)], name='VelBCP')

class VelDerBCP(FormulaProduct):
    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), DerBCP(data_K, **kwargs_formula)], name='VelDerBCP')




class VelVelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1),
                          data_K.covariant('Ham', commader=1)], name='VelVelVel')


class MassVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), data_K.covariant('Ham', commader=1)], name='MassVel')


class MassMass(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), InvMass(data_K)], name='MassMass')


class VelMassVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), InvMass(data_K),
                          data_K.covariant('Ham', commader=1)], name='VelMassVel')


class OmegaS(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Spin(data_K)], name='SpinOmega')


class OmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='OmegaOmega')


class OmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)],
                         name='OmegaHplus')


class emcha_surf(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        velocity = data_K.covariant('Ham', commader=1)
        formula1 = FormulaProduct([InvMass(data_K), Omega(data_K, **kwargs_formula), velocity],
            name='mass-berry-vel (apus)(psua) ([au]bpbs) ([us]abbp)')
        formula2 = FormulaProduct([velocity, DerOmega(data_K, **kwargs_formula), velocity],
            name='v-derberry-vel (aups) ([au]bbps) ([us]abpb)')
        tmp = FormulaSum([formula2, formula1], [1, 1], ['aups', 'apus'])
        super().__init__([
            tmp,
            DeltaProduct(delta_f, formula1, 'us,MLabbp->MLaups'),
            DeltaProduct(delta_f, tmp, 'au,MLbbps->MLaups'),
            DeltaProduct(delta_f, formula2, 'us,MLabpb->MLaups'),
            formula2,
            # romove antisymmatric part
            # DeltaProduct(Levi_Civita
            #    DeltaProduct(Levi_Civita,tmp,'pta,MLxtbs->MLaxpbs'),
            #    'xub,MLaxpbs->MLaups')
        ],
            # [2,-2,-1,1,-1,-1],['aups','aups','aups','aups','aups','aups'])
            [2, -2, -1, 1, -1], ['aups', 'aups', 'aups', 'aups', 'aups'])


class NLDrude_Z_spin(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Spin(data_K)], name='Der3ESpin')
        term2 = FormulaProduct([Der2Spin(data_K), data_K.covariant('Ham', commader=1)], name='Der2SpinVel')
        super().__init__([term1, term2], [-1, 1], ['apsu', 'uaps'])


class NLDrude_Z_orb_Hplus(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='Der3EHplus')
        term2 = FormulaProduct([Der2Morb(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='Der2HplusVel')
        super().__init__([term1, term2], [-1, 1], ['apsu', 'uaps'])


class NLDrude_Z_orb_Omega(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Omega(data_K, **kwargs_formula)], name='Der3EOmega')
        term2 = FormulaProduct([Der2Omega(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='Der2OmegaVel')
        super().__init__([term1, term2], [-1, 1], ['apsu', 'uaps'])
