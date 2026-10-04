"""Run with: python -m unittest discover -s calculators_collect/tests -v.

Requires a complete WannierBerri 1.7/1.8 environment. Actual upstream base,
DerDcov, data grouping, integration and tabulation classes are exercised.
"""
import itertools
import unittest
from types import SimpleNamespace
import numpy as np
from wannierberri.formula.formula import Matrix_ln, Matrix_GenDer_ln
from wannierberri.formula.covariant import Dcov
from wannierberri.formula import covariant_basic as basic
from wannierberri.data_K.data_K import Data_K, get_transform_TR, get_transform_Inv
from calculators_collect.formula import covariant as f
from calculators_collect.calculators import static, tabulate
from calculators_collect.system.needed_data import metric_required_matrices, validate_metric_data


class Model:
    """Consistent H^W(k) and moving basis S(k)=exp[-i f(k)T]."""
    get_bands_in_range_groups = Data_K.get_bands_in_range_groups
    get_bands_in_range_groups_ik = Data_K.get_bands_in_range_groups_ik

    def __init__(self, x, h0, v, w, t, c, b, rotation=None):
        h = h0 + np.einsum('a,ija->ij', x, v) + .5*np.einsum('a,b,ijab->ij', x, x, w)
        e, u = np.linalg.eigh(h)
        if rotation is not None:
            u = u @ rotation
        def rotate(z):
            return np.einsum('im,ij...,jn->mn...', u.conj(), z, u)[None]
        grad = c+b@x
        aa = np.einsum('a,ij->ija', grad, t)
        da = np.einsum('ad,ij->ijad', b, t)
        ff = np.einsum('a,b,ij->ijab', grad, grad, t@t)
        df = np.einsum('ad,b,ij->ijabd', b, grad, t@t)+np.einsum('a,bd,ij->ijabd', grad, b, t@t)
        self.raw = {('Ham',1):rotate(v+np.einsum('ijab,b->ija',w,x)), ('Ham',2):rotate(w),
                    ('AA',0):rotate(aa), ('AA',1):rotate(da), ('FF',0):rotate(ff), ('FF',1):rotate(df)}
        gap=e[:,None]-e[None,:]; inv=np.zeros_like(gap)
        np.divide(1.,gap,out=inv,where=abs(gap)>1e-7)
        self.E_K,self.dEig_inv=e[None],inv[None]
        self.D_H=-self.raw['Ham',1]*inv[None,:,:,None]
        self.Dcov=Dcov(self)
        self.force_internal_terms_only=False
        self.num_wann=len(e); self.nk=1; self.cell_volume=2.
    def covariant(self,name,commader=0,gender=0):
        if gender and name=='Ham':return self.covariant('Ham',commader=1)
        if gender:return Matrix_GenDer_ln(self.covariant(name),self.covariant(name,commader=1),self.Dcov)
        return Matrix_ln(self.raw[name,commader], transformTR=get_transform_TR(name,commader), transformInv=get_transform_Inv(name,commader))


def parameters():
    rng=np.random.default_rng(1707)
    def herm(shape):
        a=rng.normal(size=shape)+1j*rng.normal(size=shape)
        return .5*(a+a.swapaxes(0,1).conj())
    w=.13*herm((4,4,3,3));w=.5*(w+w.swapaxes(2,3))
    return (np.diag([-3.,-1.,1.4,3.7]), .25*herm((4,4,3)), w, .3*herm((4,4)),
            np.array([.6,-.3,.5]),np.array([[.4,.1,-.2],[.1,-.3,.15],[-.2,.15,.25]]))


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.par=parameters();self.x=np.array([.17,-.21,.13]);self.data=Model(self.x,*self.par)
    def parts(self,active):
        return np.array(active,dtype=int),np.setdiff1d(np.arange(4),active)
    def close(self,a,b,atol=2e-10):
        np.testing.assert_allclose(a,b,rtol=2e-8,atol=atol)
    def test_metric_against_physical_state_geometry(self):
        # For this complete moving frame, M connects derivatives of physical states.
        m=self.data.raw['AA',0][0]+1j*self.data.D_H[0]
        for active in ([0],[0,1],[],[0,1,2,3]):
            a,b=self.parts(active)
            q=np.einsum('mla,lnb->mnab',m[a][:,b],m[b][:,a])
            g=.5*(q+q.swapaxes(2,3))
            self.close(f.Metric(self.data).nn(0,a,b),g)
            if len(a):self.assertGreaterEqual(np.linalg.eigvalsh(np.trace(g,axis1=0,axis2=1).real).min(),-1e-12)
    def test_derivative_all_switches_and_group_sizes(self):
        for flags in itertools.product((False,True),repeat=3):
            kw=dict(zip(('internal_terms','external_terms','cross_terms'),flags))
            for active in ([0],[0,1]):
                a,b=self.parts(active);h=1e-4
                dg=f.DerMetric(self.data,**kw).trace(0,a,b)
                fd=np.empty((3,3,3))
                for d in range(3):
                    dx=np.eye(3)[d]*h
                    fd[:,:,d]=(f.Metric(Model(self.x+dx,*self.par),**kw).trace(0,a,b)-
                               f.Metric(Model(self.x-dx,*self.par),**kw).trace(0,a,b))/(2*h)
                with self.subTest(flags=flags,active=active):self.close(dg,fd,atol=2e-9)
    def test_basic_reference_and_symmetries(self):
        for active in ([0],[0,1],[],[0,1,2,3]):
            a,b=self.parts(active)
            for ext in (False,True):
                for cls,reference in ((f.Metric,basic.tildeFab),(f.DerMetric,basic.tildeFab_d)):
                    z=cls(self.data,external_terms=ext).nn(0,a,b)
                    ref=reference(self.data,external_terms=ext).nn(0,a,b)
                    self.close(z,.5*(ref+ref.swapaxes(2,3)))
                    self.close(z,z.swapaxes(0,1).conj());self.close(z,z.swapaxes(2,3))
    def test_exact_degeneracy_covariance_and_finite_difference(self):
        par=list(self.par);par[0]=np.diag([-2.,-2.,1.4,3.7]);x=np.zeros(3)
        rng=np.random.default_rng(9);w=np.eye(4,dtype=complex)
        w[:2,:2]=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        d0=Model(x,*par);dr=Model(x,*par,rotation=w);a,b=self.parts([0,1])
        for cls in (f.Metric,f.DerMetric):
            z=cls(d0).nn(0,a,b)
            self.close(cls(dr).nn(0,a,b),np.einsum('im,ij...,jn->mn...',w[:2,:2].conj(),z,w[:2,:2]))
        h=1e-4;fd=np.stack([(f.Metric(Model(x+np.eye(3)[d]*h,*par)).trace(0,a,b)-f.Metric(Model(x-np.eye(3)[d]*h,*par)).trace(0,a,b))/(2*h) for d in range(3)],axis=-1)
        self.close(f.DerMetric(d0).trace(0,a,b),fd,atol=2e-9)
    def test_two_band_regression(self):
        par=(np.diag([-1.,1.]),np.zeros((2,2,3)),np.zeros((2,2,3,3)),np.array([[0.,1.],[1.,0.]]),np.zeros(3),np.diag([1.,0.,0.]))
        data=Model(np.array([.7,0.,0.]),*par)
        self.assertAlmostEqual(f.Metric(data).trace(0,[0],[1])[0,0],.49)
        self.assertAlmostEqual(f.DerMetric(data).trace(0,[0],[1])[0,0,0],1.4)
        self.close(f.DerMetric(data).trace(0,[0,1],[]),np.zeros((3,3,3)))
    def test_product_axes(self):
        a,b=self.parts([0,1]);v=self.data.raw['Ham',1][0][a][:,a]
        g=f.Metric(self.data).nn(0,a,b);dg=f.DerMetric(self.data).nn(0,a,b)
        for cls,ref in ((f.VelMetric,np.einsum('mkc,knab->mncab',v,g)),(f.MetricVel,np.einsum('mkab,knc->mnabc',g,v)),(f.VelDerMetric,np.einsum('mkc,knabd->mncabd',v,dg))):
            self.close(cls(self.data).nn(0,a,b),ref)
        self.close(f.VelMetric(self.data).trace(0,a,b),f.MetricVel(self.data).trace(0,a,b).transpose(2,0,1))
    def test_switches_and_required_data(self):
        self.assertEqual(metric_required_matrices(),{'Ham','AA','FF'})
        self.assertEqual(metric_required_matrices(external_terms=False),{'Ham'})
        self.assertEqual(metric_required_matrices(force_internal_terms_only=True),{'Ham'})
        d=self.data;d.system=SimpleNamespace(has_R_mat=lambda key:key in {'Ham','AA'})
        with self.assertRaisesRegex(ValueError,'FF'):f.Metric(d)
        with self.assertRaisesRegex(ValueError,'FF'):f.DerMetric(d)
        d.force_internal_terms_only=True
        a,b=self.parts([0]);self.close(f.DerMetric(d).nn(0,a,b),f.DerMetric(d,external_terms=False).nn(0,a,b))
        validate_metric_data(d)
    def occupied_trace(self, data, formula, mu):
        # Direct selected-subspace evaluation, independent of prefix differences.
        active=np.flatnonzero(data.E_K[0]<mu)
        other=np.setdiff1d(np.arange(data.num_wann),active)
        return formula(data).trace(0,active,other)/data.cell_volume
    def test_static_occupied_subspace_and_window_invariance(self):
        wide=np.linspace(-5.,6.,111)+1.234567e-6
        narrow=wide[35:91]
        for calculator,formula in ((static.Metric,f.Metric),(static.MetricDipole_FermiSea,f.DerMetric)):
            actual=calculator(Efermi=wide)(self.data).data
            expected=np.array([self.occupied_trace(self.data,formula,mu) for mu in wide])
            self.close(actual,expected)
            self.close(calculator(Efermi=narrow)(self.data).data,actual[35:91])
    def test_static_filled_sea_is_full_subspace(self):
        data=self.data;ef=np.array([5.,6.])
        for calculator,formula in ((static.Metric,f.Metric),(static.MetricDipole_FermiSea,f.DerMetric)):
            actual=calculator(Efermi=ef)(data).data
            band_sum=sum(formula(data).trace(0,*self.parts([n])) for n in range(4))/data.cell_volume
            self.assertGreater(np.max(abs(band_sum)),1e-5)
            expected=self.occupied_trace(data,formula,ef[0])
            self.close(expected,np.zeros_like(expected))
            self.close(actual,np.stack([expected,expected]))
    def test_static_tetra_occupied_subspace(self):
        # Actual tetra branch, controlled exact weights for two occupied groups.
        # This tests trace semantics, not the tetra interpolation algorithm.
        data=self.data
        for groups,active in (({(0,1):np.ones(2),(1,2):np.ones(2)},[0,1]),
                              ({(0,2):np.ones(2)},[0,1]),
                              ({(0,4):np.ones(2)},[0,1,2,3])):
            data.tetraWeights=SimpleNamespace(weights_all_band_groups=lambda ef,**kw: [groups])
            for calculator,formula in ((static.Metric,f.Metric),(static.MetricDipole_FermiSea,f.DerMetric)):
                actual=calculator(Efermi=np.array([0.,.1]),tetra=True)(data).data
                expected=formula(data).trace(0,*self.parts(active))/data.cell_volume
                self.close(actual,np.stack([expected,expected]))
    def test_surface_sign_and_axes(self):
        ef=np.linspace(-4,5,901)+1.234567e-6
        actual=static.MetricQuadrupole_FermiSurf(Efermi=ef)(self.data).data
        de=ef[1]-ef[0];extended=np.r_[ef[0]-de,ef,ef[-1]+de]
        # Independent occupation selection at each mu, then positive d/dmu.
        sea=np.array([self.occupied_trace(self.data,f.VelDerMetric,mu) for mu in extended])
        self.close(actual,(sea[2:]-sea[:-2])/(2*de))
        self.assertEqual(actual.shape,(len(ef),3,3,3,3))
        self.assertGreater(np.max(abs(actual)),1e-5)
        self.close(actual.sum(axis=0)*de,sea[-1]-sea[0])
    def test_tabulators_and_degenerate_group_average(self):
        par=list(self.par);par[0]=np.diag([-2.,-2.,1.4,3.7]);data=Model(np.zeros(3),*par)
        for tab,formula in ((tabulate.Metric,f.Metric),(tabulate.DerMetric,f.DerMetric)):
            actual=tab(ibands=[0,1])(data).data
            expected=formula(data).trace(0,*self.parts([0,1]))/2
            self.close(actual[0],np.stack([expected,expected]))
    def test_static_degenerate_occupied_subspace(self):
        par=list(self.par);par[0]=np.diag([-2.,-2.,1.4,3.7]);data=Model(np.zeros(3),*par)
        ef=np.linspace(-3.,5.,81)+1.234567e-6
        for calculator,formula in ((static.Metric,f.Metric),(static.MetricDipole_FermiSea,f.DerMetric)):
            expected=np.array([self.occupied_trace(data,formula,mu) for mu in ef])
            self.close(calculator(Efermi=ef)(data).data,expected)
    def test_two_band_occupied_metric_and_derivative(self):
        par=(np.diag([-1.,1.]),np.zeros((2,2,3)),np.zeros((2,2,3,3)),np.array([[0.,1.],[1.,0.]]),np.zeros(3),np.diag([1.,0.,0.]))
        data=Model(np.array([.7,0.,0.]),*par);ef=np.array([-2.,0.,2.])
        for calculator,component,value in ((static.Metric,(0,0),.49),
                                          (static.MetricDipole_FermiSea,(0,0,0),1.4)):
            actual=calculator(Efermi=ef)(data).data
            self.close(actual[(slice(None),)+component],np.array([0.,value,0.])/data.cell_volume)
    def test_subspace_formula_is_not_additive(self):
        for cls in (f.Metric,f.DerMetric,f.VelMetric,f.MetricVel,f.VelDerMetric):
            self.assertFalse(cls(self.data).additive)
        for calculator,formula in ((static.Metric,f.Metric),(static.MetricDipole_FermiSea,f.DerMetric),
                                   (static.MetricQuadrupole_FermiSurf,f.VelDerMetric)):
            cal=calculator(Efermi=np.array([0.,1.]))
            self.assertIs(cal.Formula,formula)
            self.assertFalse(cal.Formula(self.data).additive)

if __name__=='__main__':unittest.main()
