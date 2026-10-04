"""BCP tests against spectral sums, analytic limits and covariant finite differences."""
import itertools
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.linalg import polar
from test_metric import Model, parameters
from calculators_collect.formula import covariant as f
from calculators_collect.formula.covariant_basic import _BCPConnections
from calculators_collect.calculators import static, tabulate
from calculators_collect.system.needed_data import bcp_required_matrices


class BCPModel(Model):
    """The same quadratic H and moving frame; exact higher comma derivatives."""
    def covariant(self, name, commader=0, gender=0):
        if (name,commader) not in self.raw and ((name=='Ham' and commader==3) or (name=='AA' and commader==2)):
            self.raw[name,commader]=np.zeros((1,self.num_wann,self.num_wann)+(3,)*(commader+(name=='AA')),complex)
        return super().covariant(name,commader,gender)


class BCPTests(unittest.TestCase):
    def setUp(self):
        self.par=parameters();self.x=np.array([.17,-.21,.13]);self.data=BCPModel(self.x,*self.par)
    def parts(self,a):
        return np.array(a,dtype=int),np.setdiff1d(np.arange(4),a)
    def close(self,a,b,atol=2e-10):
        np.testing.assert_allclose(a,b,rtol=2e-8,atol=atol)
    def test_spectral_matrix_and_normalization(self):
        # Independent explicit sums over l, preserving both energy denominators.
        d=self.data;e=d.E_K[0];connection=d.raw['AA',0][0]+1j*d.D_H[0]
        for active in ([0],[0,1],[0,2],[],list(range(4))):
            a,b=self.parts(active);expected=np.zeros((len(a),len(a),3,3),complex)
            for im,m in enumerate(a):
                for jn,n in enumerate(a):
                    for l in b:
                        q=np.outer(connection[m,l],connection[l,n])
                        expected[im,jn]+=.25*(q+q.T)*(1/(e[m]-e[l])+1/(e[n]-e[l]))
            self.close(f.BCP(d).nn(0,a,b),expected)
    def test_first_second_derivatives_all_switches(self):
        h=1e-4
        for flags in itertools.product((False,True),repeat=3):
            kw=dict(zip(('internal_terms','external_terms','cross_terms'),flags))
            for active in ([0],[0,1],[0,2]):
                a,b=self.parts(active)
                for formula,lower in [(f.DerBCP,f.BCP),(f.Der2BCP,f.DerBCP)]:
                    fd=[]
                    for c in range(3):
                        dx=np.eye(3)[c]*h
                        dp=BCPModel(self.x+dx,*self.par);dm=BCPModel(self.x-dx,*self.par)
                        fd.append((lower(dp,**kw).trace(0,a,b)-lower(dm,**kw).trace(0,a,b))/(2*h))
                    with self.subTest(flags=flags,active=active,formula=formula.__name__):
                        self.close(formula(self.data,**kw).trace(0,a,b),np.stack(fd,-1),atol=2e-9)
    def test_second_derivative_direct_bcp_difference(self):
        a,b=self.parts([0,1]);h=3e-4;g=f.BCP(self.data).trace(0,a,b)
        dd=f.Der2BCP(self.data).trace(0,a,b)
        for d in range(3):
            for e in range(3):
                def value(shift):return f.BCP(BCPModel(self.x+shift,*self.par)).trace(0,a,b)
                xd=np.eye(3)[d]*h;xe=np.eye(3)[e]*h
                fd=(value(xd)+value(-xd)-2*g)/h**2 if d==e else (value(xd+xe)-value(xd-xe)-value(-xd+xe)+value(-xd-xe))/(4*h*h)
                self.close(dd[:,:,d,e],fd,atol=4e-8)
        self.close(dd,dd.swapaxes(-1,-2))
    def test_full_matrix_parallel_transport_derivatives(self):
        a,b=self.parts([0,1]);h=1e-5
        def vectors(x):
            h0,v,w,*_=self.par
            return np.linalg.eigh(h0+np.einsum('a,ija->ij',x,v)+.5*np.einsum('a,b,ijab->ij',x,x,w))[1]
        u0=vectors(self.x)
        def aligned(x,formula):
            u=vectors(x);rotation=polar(u[:,a].conj().T@u0[:,a])[0]
            z=formula(BCPModel(x,*self.par)).nn(0,a,b)
            return np.einsum('im,ij...,jn->mn...',rotation.conj(),z,rotation)
        for formula,lower in [(f.DerBCP,f.BCP),(f.Der2BCP,f.DerBCP)]:
            fd=np.stack([(aligned(self.x+np.eye(3)[c]*h,lower)-aligned(self.x-np.eye(3)[c]*h,lower))/(2*h) for c in range(3)],-1)
            self.close(formula(self.data).nn(0,a,b),fd,atol=3e-9)
    def test_trace_additivity_all_subsets(self):
        for formula in [f.BCP,f.DerBCP,f.Der2BCP]:
            single=[formula(self.data).trace(0,*self.parts([n])) for n in range(4)]
            self.assertTrue(formula(self.data).additive)
            for mask in range(16):
                a,b=self.parts([n for n in range(4) if mask & (1<<n)])
                expected=sum((single[n] for n in a),np.zeros((3,)*formula(self.data).ndim))
                self.close(formula(self.data).trace(0,a,b),expected)
    def test_degenerate_gauge_covariance(self):
        par=list(self.par);par[0]=np.diag([-2.,-2.,1.4,3.7]);x=np.zeros(3)
        rng=np.random.default_rng(812);r=np.eye(4,dtype=complex)
        r[:2,:2]=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        d0=BCPModel(x,*par);dr=BCPModel(x,*par,rotation=r);a,b=self.parts([0,1])
        for formula in [f.BCP,f.DerBCP,f.Der2BCP]:
            z=formula(d0).nn(0,a,b)
            self.close(formula(dr).nn(0,a,b),np.einsum('im,ij...,jn->mn...',r[:2,:2].conj(),z,r[:2,:2]))
            self.close(z,z.swapaxes(0,1).conj());self.close(z,z.swapaxes(2,3))
        with self.assertRaisesRegex(ValueError,'gap'):f.BCP(d0).trace(0,[0],[1,2,3])
    def test_analytic_two_band_and_term_switches(self):
        par=(np.diag([-1.,1.]),np.zeros((2,2,3)),np.zeros((2,2,3,3)),np.array([[0.,1.],[1.,0.]]),np.zeros(3),np.diag([1.,0.,0.]))
        d=BCPModel(np.array([.7,0.,0.]),*par)
        for formula,value in [(f.BCP,-.245),(f.DerBCP,-.7),(f.Der2BCP,-1.)]:
            z=formula(d,internal_terms=False,cross_terms=False).trace(0,[0],[1])
            self.assertAlmostEqual(z[(0,)*z.ndim],value)
            self.close(formula(d).trace(0,[0,1],[]),np.zeros_like(z))
            self.close(formula(d,external_terms=False).trace(0,[0],[1]),np.zeros_like(z))
    def test_velocity_products_and_nonadditivity(self):
        a,b=self.parts([0,1]);v=self.data.raw['Ham',1][0][a][:,a]
        for formula,base in [(f.VelBCP,f.BCP),(f.VelDerBCP,f.DerBCP)]:
            self.close(formula(self.data).nn(0,a,b),np.einsum('mkc,kn...->mnc...',v,base(self.data).nn(0,a,b)))
            self.assertFalse(formula(self.data).additive)
            bands=sum(formula(self.data).trace(0,*self.parts([n])) for n in range(4))
            self.assertGreater(np.max(abs(bands)),1e-5)
            self.close(formula(self.data).trace(0,*self.parts(range(4))),np.zeros_like(bands))
    def test_static_integrals_and_surface_weights(self):
        ef=np.linspace(-5,6,111)+1.234567e-6;h=ef[1]-ef[0]
        pairs=[(static.BCPD_FermiSea,f.DerBCP,0),(static.BCPQ_FermiSea,f.Der2BCP,0),
               (static.BCPD_FermiSurf,f.VelBCP,1),(static.BCPQ_FermiSurf,f.VelDerBCP,1)]
        for calculator,formula,order in pairs:
            def trace(mu):
                a,b=self.parts(np.flatnonzero(self.data.E_K[0]<mu))
                return formula(self.data).trace(0,a,b)/self.data.cell_volume
            expected=np.array([trace(mu) for mu in ef]) if not order else np.array([(trace(mu+h)-trace(mu-h))/(2*h) for mu in ef])
            self.close(calculator(Efermi=ef)(self.data).data,expected)
            self.close(calculator(Efermi=ef,constant_factor=2.3)(self.data).data,2.3*expected)
            self.close(calculator(Efermi=ef[30:90])(self.data).data,expected[30:90])
    def test_tabulators(self):
        par=list(self.par);par[0]=np.diag([-2.,-2.,1.4,3.7]);d=BCPModel(np.zeros(3),*par)
        for tab,formula in [(tabulate.BCP,f.BCP),(tabulate.DerBCP,f.DerBCP),(tabulate.BCPQuadrupole,f.Der2BCP)]:
            z=tab(ibands=[0,1])(d).data[0];expected=formula(d).trace(0,*self.parts([0,1]))/2
            self.close(z,np.stack([expected,expected]))
    def test_data_requirements(self):
        self.assertEqual(bcp_required_matrices(),{'Ham','AA'})
        self.assertEqual(bcp_required_matrices(external_terms=False),{'Ham'})
        d=self.data;d.system=SimpleNamespace(has_R_mat=lambda name:name=='Ham')
        for formula in [f.BCP,f.DerBCP,f.Der2BCP]:
            with self.assertRaisesRegex(ValueError,'AA'):formula(d)
        d.force_internal_terms_only=True
        self.close(f.BCP(d).trace(0,*self.parts([0])),f.BCP(d,external_terms=False).trace(0,*self.parts([0])))

if __name__=='__main__':unittest.main()
