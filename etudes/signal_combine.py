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
class CombinedSignal_1psr(object):
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

        self.has_wn=has_wn
        self.has_basis_ecorr=has_basis_ecorr
        self.has_rn = has_rn
        self.has_tm = has_tm
        self.has_cw=has_cw

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
    
    def _init_basis(self, Umat=None, Fmat=None):
        T = self.psr.Mmat
        if Fmat is not None:
            T = jnp.concatenate([Fmat, T], axis=1)
        if Umat is not None:
            T = jnp.concatenate([Umat, T], axis=1)
        return T

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
    def get_delay(self, pars):
        return self.wn_signal.get_delay(pars) + self.cw_signal.get_delay(pars)

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
        return self.psr._residuals - self.get_delay(pars)
    
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

