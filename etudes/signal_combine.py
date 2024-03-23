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
                                      RN_Signal)

@register_pytree_node_class
class EtudesPTA(object):
    """
    Multi-pulsar signal object. Creates a list of CombinedSignal_1psr
    objects for every pulsar input. This will be looped over for
    likelihood calculations

    TODO: Find a way to appropriately loop through likelihood
    function such that it can be written in terms of JAX primitives
    and break the natural looping over individual pulsars.
    """
    def __init__(self, psrs, has_wn=True, has_basis_ecorr=True,
               has_rn=True, has_tm=True, has_cw=True, param_names=None,
               Umats=None, ecorr_weights=None, Fmats=None, Ffreqs=None,
               efac=True, equad=True, fix_wn=True, fix_wn_vals=None, TNTs=None):
        self.psrs = psrs
        self.param_names = param_names

        self.has_wn = has_wn
        self.has_basis_ecorr = has_basis_ecorr
        self.has_rn = has_rn
        self.has_tm = has_tm
        self.has_cw = has_cw

        if not Umats:
            self.Umats = [None] * len(psrs)
        else:
            self.Umats = Umats
        if not ecorr_weights:
            self.ecorr_weights = [None] * len(psrs)
        else:
            self.ecorr_weights = ecorr_weights
        if not Fmats:
            self.Fmats = [None] * len(psrs)
        else:
            self.Fmats = Fmats
        if not Ffreqs:
            self.Ffreqs = [None] * len(psrs)
        else:
            self.Ffreqs = Ffreqs

        self.efac = efac
        self.equad = equad
        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        if not self.param_names:
            self._name_params()

        # Create individual pulsar signal objects
        self.signalcollections = []
        for psr, Umat, ecw, Fmat, Ff in zip(self.psrs, self.Umats,
                                            self.ecorr_weights, self.Fmats,
                                            self.Ffreqs):
            self.signalcollections.append(Etudes1PsrSignal(psr, has_wn=has_wn,
                                                           has_basis_ecorr=has_basis_ecorr,
                                                           has_rn=has_rn, has_tm=has_tm,
                                                           has_cw=has_cw, Umat=Umat, ecorr_weights=ecw,
                                                           Fmat=Fmat, Ffreqs=Ff, efac=efac, equad=equad,
                                                           fix_wn=fix_wn, fix_wn_vals=fix_wn_vals))

        # Fixed vs varied white noise determines the likelihood function
        if fix_wn:
            self.TNTs = [sc.set_TNT() for sc in self.signalcollections]
            self.ll_fn = self.ll_fn_fixwn
        else:
            self.ll_fn = self.ll_fn_varywn
    
    def _name_params(self):
        """
        Populate parameter namespace given included signals.

        (Currently no backend selection on white noise... will add later)
        """
        pars = []

        for psr in self.psrs:
            if self.has_wn and not self.fix_wn:
                if self.efac:
                    pars.append('{}_efac'.format(psr.name))
                if self.equad:
                    pars.append('{}_log10_t2equad'.format(psr.name))
            if self.has_rn:
                    pars.append('{}_rn_gamma'.format(psr.name))
                    pars.append('{}_rn_log10_A'.format(psr.name))
        
        # If searching for continuous wave signal
        if self.has_cw:
            cw_pars = ['cw_cosinc', 'cw_costheta', 'cw_log10_Mc',
                       'cw_log10_fgw', 'cw_log10_h', 'cw_phase0',
                       'cw_phi', 'cw_psi']
            pars.extend(cw_pars)
        
        self.param_names = pars

    def _map_params(self, params):
        ret = {}
        for i, p in enumerate(self.param_names):
            ret[p] = params[i]
        return ret
    
    @jax.jit
    def ll_fn_fixwn(self, xs):
        params = self._map_params(xs)

        loglike = 0

        for i, sc in enumerate(self.signalcollections):
            TNr = sc.get_TNr(params)
            TNT = self.TNTs[i]
            rNr_logdet = sc.get_rNr_logdet(params)
            phiinv, logdet_phi = sc.get_phiinv_logdet(params)

            loglike += -0.5 * rNr_logdet

            Sigma = TNT + (jnp.diag(phiinv) if phiinv.ndim == 1 else phiinv)
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

            Sigma = TNT + (jnp.diag(phiinv) if phiinv.ndim == 1 else phiinv)
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
        return (), (self.psrs, self.param_names, self.has_wn, self.has_basis_ecorr,
                    self.has_rn, self.has_tm, self.has_cw, self.param_names,
                    self.Umats, self.ecorr_weights, self.Fmats, self.Ffreqs,
                    self.efac, self.equad, self.fix_wn, self.fix_wn_vals,
                    self.TNTs)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
        


@register_pytree_node_class
class Etudes1PsrSignal(object):
    """
    Single-pulsar concatenated signal object. All potential signals
    should (key word: should) be included here as potential inputs.

    TODO: Find a way to appropriately loop through all given signal
    functions such that it can be written in terms of JAX primitives.
    """
    def __init__(self, psr, has_wn=True, has_basis_ecorr=True,
                 has_rn=True, has_tm=True, has_cw=True,
                 Umat=None, ecorr_weights=None, Fmat=None, Ffreqs=None,
                 efac=True, equad=True, fix_wn=True, fix_wn_vals=None):
        self.psr = psr

        self.has_wn = has_wn
        self.has_basis_ecorr = has_basis_ecorr
        self.has_rn = has_rn
        self.has_tm = has_tm
        self.has_cw = has_cw

        self.Umat = Umat
        self.ecorr_weights = ecorr_weights
        self.Fmat = Fmat
        self.Ffreqs = Ffreqs

        self.efac = efac
        self.equad = equad
        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        self._init_model(Umat=Umat, ecorr_weights=ecorr_weights, Fmat=Fmat, Ffreqs=Ffreqs,
                         fix_wn=fix_wn, fix_wn_vals=fix_wn_vals)
        self.T = self._init_basis(Umat=Umat, Fmat=Fmat)

        self._init_get_delay(has_cw=has_cw)
    
    def _init_basis(self, Umat=None, Fmat=None):
        T = self.psr.Mmat
        if Fmat is not None:
            T = jnp.concatenate([Fmat, T], axis=1)
        if Umat is not None:
            T = jnp.concatenate([Umat, T], axis=1)
        return T
    
    def _init_get_delay(self, has_cw=True):
        """
        get_delay functional form depends on whether or not a
        CW signal is present in the model.
        """
        if has_cw:
            self.get_delay = self._get_delay_cw
        else:
            self.get_delay = self._get_delay_nocw


    def _init_model(self, Umat=None, ecorr_weights=None, Fmat=None, Ffreqs=None,
                    fix_wn=True, fix_wn_vals=None):
        if self.has_wn:
            self.wn_signal = WN_Signal(self.psr, fix_wn=fix_wn, fix_wn_vals=fix_wn_vals)
        if self.has_basis_ecorr:
            self.basis_ecorr_signal = ECORR_GP_Signal(self.psr, Umat=Umat, weights=ecorr_weights)
        if self.has_tm:
            self.tm_signal = Timing_Model(self.psr)
        if self.has_rn:
            self.rn_signal = RN_Signal(self.psr, Fmat=Fmat, Ffreqs=Ffreqs)
        if self.has_cw:
            self.cw_signal = CW_Signal(self.psr)
    
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
    def set_TNT(self):
        return self.get_TNT(self.fix_wn_vals)

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

        TNr = self.get_TNr(pars)
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
        return (), (self.psr, self.has_wn, self.has_basis_ecorr,
                    self.has_rn, self.has_tm, self.has_cw,
                    self.Umat, self.ecorr_weights, self.Fmat, self.Ffreqs,
                    self.efac, self.equad, self.fix_wn, self.fix_wn_vals)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

