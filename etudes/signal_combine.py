"""
A (perhaps vain) attempt to create a class to contain all signals
and subsequent signal functions for a single pulsar added to a PTA model.
The class is registered as a Pytree to work alongside JAX transformations,
but I'm skeptical about the ability to unroll the natural looping structure
of PTA signal modeling.

For now, try not to worry about unrolling the loops pertaining to some
N_{psr} PTA.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.tree_util import register_pytree_node_class

from etudes.etudes.wn_signals import WN_Signal
from etudes.etudes.deterministic import CW_Signal
from etudes.etudes.gp_signals import (ECORR_GP_Signal, Timing_Model,
                                      RN_Signal, Common_GW_Signal)

@register_pytree_node_class
class _EtudesPTA(object):
    """
    Multi-pulsar signal object. Creates a list of CombinedSignal_1psr
    objects for every pulsar input. This will be looped over for
    likelihood calculations

    TODO: Find a way to appropriately loop through likelihood
    function such that it can be written in terms of JAX primitives
    and break the natural looping over individual pulsars.
    """
    def __init__(self, psrs, signalcollections, param_names=None, fix_wn=True):
        self.psrs = psrs
        self.param_names = param_names
        self.signalcollections = signalcollections

        self.fix_wn = fix_wn

        # Fixed vs varied white noise determines the likelihood function
        if fix_wn:
            self.ll_fn = self.ll_fn_fixwn
        else:
            self.ll_fn = self.ll_fn_varywn

    def _map_params(self, params):
        ret = {}
        for i, p in enumerate(self.param_names):
            ret[p] = params[i]
        return ret

    @jax.jit
    def ll_fn_fixwn(self, xs):
        params = self._map_params(xs)

        loglike = 0

        for sc in self.signalcollections:
            TNr = sc.get_TNr(params)
            rNr_logdet = sc.get_rNr_logdet(params)
            phiinv, logdet_phi = sc.get_phiinv_logdet(params)

            loglike += -0.5 * rNr_logdet

            Sigma = sc.TNT + (jnp.diag(phiinv))
            cf = jsl.cho_factor(Sigma)
            expval = jsl.cho_solve(cf, TNr)
            
            logdet_sigma = jnp.sum(2 * jnp.log(jnp.diag(cf[0])))
            loglike += 0.5 * (jnp.dot(TNr, expval) - logdet_sigma - logdet_phi)

        return loglike
    
    @jax.jit
    def ll_fn_varywn(self, xs):
        params = self._map_params(xs)

        loglike = 0

        for sc in self.signalcollections:
            TNr = sc.get_TNr(params)
            TNT = sc.get_TNT(params)
            rNr_logdet = sc.get_rNr_logdet(params)
            phiinv, logdet_phi = sc.get_phiinv_logdet(params)

            loglike += -0.5 * rNr_logdet

            Sigma = TNT + (jnp.diag(phiinv))
            cf = jsl.cho_factor(Sigma)
            expval = jsl.cho_solve(cf, TNr)
            
            logdet_sigma = jnp.sum(2 * jnp.log(jnp.diag(cf[0])))
            loglike += 0.5 * (jnp.dot(TNr, expval) - logdet_sigma - logdet_phi)

        return loglike
    
    @jax.jit
    def get_loglikelihood(self, xs):
        return self.ll_fn(xs)
    
    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psrs, self.signalcollections,
                    self.param_names, self.fix_wn)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)    


@register_pytree_node_class
class _Etudes1PsrSignal(object):
    """
    Single-pulsar concatenated signal object. All potential signals
    should (key word: should) be included here as potential inputs.

    TODO: Find a way to appropriately loop through all given signal
    functions such that it can be written in terms of JAX primitives.
    """
    def __init__(self, T=None, TNT=None, psr=None,
                 has_wn=True, has_basis_ecorr=True,
                 has_rn=True, has_tm=True, has_gwb=True, has_cw=True,
                 Umat=None, ecorr_weights=None, Fmat=None, Ffreqs=None,
                 efac=True, equad=True, fix_wn=True, fix_wn_vals=None,
                 rn_comps=30, gwb_comps=5, tref=0):
        self.psr = psr

        self.has_wn = has_wn
        self.has_basis_ecorr = has_basis_ecorr
        self.has_rn = has_rn
        self.has_tm = has_tm
        self.has_gwb = has_gwb
        self.has_cw = has_cw

        self.Umat = Umat
        self.ecorr_weights = ecorr_weights
        self.Fmat = Fmat
        self.Ffreqs = Ffreqs

        self.efac = efac
        self.equad = equad
        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        self.rn_comps = rn_comps
        self.gwb_comps = gwb_comps
        self.tref = tref

        self._init_model(psr, has_wn=has_wn, has_basis_ecorr=has_basis_ecorr,
                         has_tm=has_tm, has_rn=has_rn, has_gwb=has_gwb, has_cw=has_cw,
                         Umat=Umat, ecorr_weights=ecorr_weights, Fmat=Fmat, Ffreqs=Ffreqs,
                         efac=efac, equad=equad, fix_wn=fix_wn, fix_wn_vals=fix_wn_vals,
                         rn_comps=rn_comps, gwb_comps=gwb_comps, tref=tref)
        self.T, self.TNT = self._init_basis(Umat=Umat, Fmat=Fmat)
        self._init_get_delay(has_cw=has_cw)
    
    def _init_basis(self, Umat=None, Fmat=None, fix_wn_vals=None):
        T = self.psr.Mmat
        if Fmat is not None:
            T = jnp.concatenate([Fmat, T], axis=1)
        if Umat is not None:
            T = jnp.concatenate([Umat, T], axis=1)
        
        ndiag = self.wn_signal.get_ndiag(fix_wn_vals)[:, None]
        mult = T / ndiag
        TNT = jnp.dot(T.T, mult)
        return T, TNT
    
    def _init_get_delay(self, has_cw=True):
        """
        get_delay functional form depends on whether or not a
        CW signal is present in the model.
        """
        if has_cw:
            self.get_delay = self._get_delay_cw
        else:
            self.get_delay = self._get_delay_nocw


    def _init_model(self, psr, has_wn=True, has_basis_ecorr=False,
                    has_tm=True, has_rn=True, has_gwb=True, has_cw=True,
                    Umat=None, ecorr_weights=None, Fmat=None, Ffreqs=None,
                    efac=True, equad=True, fix_wn=True, fix_wn_vals=None,
                    rn_comps=30, gwb_comps=5, tref=0):
        if has_wn:
            self.wn_signal = WN_Signal(psr, efac=efac, equad=equad,
                                       fix_wn=fix_wn, fix_wn_vals=fix_wn_vals)
        if has_basis_ecorr:
            self.basis_ecorr_signal = ECORR_GP_Signal(psr, Umat=Umat, weights=ecorr_weights)
        if has_tm:
            self.tm_signal = Timing_Model(psr)
        if has_rn:
            self.rn_signal = RN_Signal(psr, Fmat=Fmat, Ffreqs=Ffreqs, ncomps=rn_comps)
        if has_gwb:
            self.gwb_signal = Common_GW_Signal(psr, Fmat=Fmat[:,:2*gwb_comps], Ffreqs=Ffreqs[:2*gwb_comps], ncomps=gwb_comps)
        if has_cw:
            self.cw_signal = CW_Signal(psr, tref=tref)
    
    @jax.jit
    def _get_delay_cw(self, pars):
        return self.wn_signal.get_delay(pars) + self.cw_signal.get_delay(pars)
    
    @jax.jit
    def _get_delay_nocw(self, pars):
        return self.wn_signal.get_delay(pars)

    @jax.jit
    def get_ndiag(self, pars):
        return self.wn_signal.get_ndiag(pars)
    
    @jax.jit
    def get_phi(self, pars):
        #ecorr_phi = self.basis_ecorr_signal.get_phi(pars)
        rn_phi = self.rn_signal.get_phi(pars)
        gw_phi = self.gwb_signal.get_phi(pars)
        rn_phi = rn_phi.at[:2*self.gwb_comps].add(gw_phi)
        tm_phi = self.tm_signal.get_phi(pars)
        return jnp.concatenate([rn_phi, tm_phi])
    
    @jax.jit
    def get_phiinv_logdet(self, pars):
        phi = self.get_phi(pars)
        return 1.0 / phi, jnp.sum(jnp.log(phi))
    
    @jax.jit
    def get_detres(self, pars):
        return self.psr.residuals - self.get_delay(pars)

    @jax.jit
    def get_TNr(self, pars):
        mult = self.get_detres(pars) / self.get_ndiag(pars)
        return jnp.dot(self.T.T, mult)
    
    @jax.jit
    def get_TNT(self, pars):
        mult = self.T / self.get_ndiag(pars)[:, None]
        return jnp.dot(self.T.T, mult)
    
    @jax.jit
    def get_rNr_logdet(self, pars):
        r = self.get_detres(pars)
        N = self.get_ndiag(pars)
        mult = r / N
        ret = jnp.dot(r.T, mult)
        logdet = jnp.sum(jnp.log(N))
        return ret + logdet
    
    @jax.jit
    def get_ll(self, pars):
        loglike = 0

        TNr = self.get_TNr(pars, T=self.T)
        phiinv, logdet_phi = self.get_phiinv_logdet(pars)
        TNT = self.get_TNT(pars)
        rNr_logdet = self.get_rNr_logdet(pars)

        loglike += -0.5 * rNr_logdet

        Sigma = TNT + jnp.diag(phiinv)
        cf = jsl.cho_factor(Sigma)
        expval = jsl.cho_solve(cf, TNr)

        logdet_sigma = jnp.sum(2 * jnp.log(jnp.diag(cf[0])))
        loglike = loglike + 0.5 * (jnp.dot(TNr, expval) - logdet_sigma - logdet_phi)

        return loglike
    
    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (self.T, self.TNT), (self.psr, self.has_wn, self.has_basis_ecorr,
                    self.has_rn, self.has_tm, self.has_gwb, self.has_cw,
                    self.Umat, self.ecorr_weights, self.Fmat, self.Ffreqs,
                    self.efac, self.equad, self.fix_wn, self.fix_wn_vals,
                    self.rn_comps, self.gwb_comps, self.tref)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

