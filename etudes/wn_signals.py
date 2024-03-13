"""
Contains functions related to modeling of white noise (WN) signals.
This includes WN signals common to PTA analyses such as
EFAC and EQUAD. Correlated WN signals such as ECORR require a bit more
care to be JAX-compatible and will be added later.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class WN_Signal(object):
    """
    Class for single-pulsar white noise signals.
    """

    def __init__(self, psr, efac=True, equad=True, fix_wn=True, fix_wn_vals=None):
        self.psr = psr

        self.has_efac = efac
        self.has_equad = equad

        self.efacname = '{}_efac'.format(psr.name)
        self.equadname = '{}_log10_t2equad'.format(psr.name)

        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        if fix_wn:
            self._init_fix_ndiag(efac=self.has_efac, equad=self.has_equad, fix_wn_vals=fix_wn_vals)
        else:
            self._init_vary_ndiag(efac=self.has_efac, equad=self.has_equad)
    
    def _init_vary_ndiag(self, efac=True, equad=True):
        if efac and equad:
            self._get_ndiag = self._combined_ndiag
        elif efac and not equad:
            self._get_ndiag = self._efac_ndiag
        elif not efac and equad:
            self._get_ndiag = self._equad_ndiag
        else:
            return # TODO: add case for no WN signal
    
    def _init_fix_ndiag(self, efac=True, equad=True, fix_wn_vals=None):
        if efac and equad:
            self._ndiag = self._combined_ndiag(fix_wn_vals)
        elif efac and not equad:
            self._ndiag = self._efac_ndiag(fix_wn_vals)
        elif not efac and equad:
            self._ndiag = self._equad_ndiag(fix_wn_vals)
        else:
            return
        
        # Hopefully this caches ndiag and doesn't recompute it each time
        self._get_ndiag = self._return_fix_ndiag

    def _return_fix_ndiag(self, pars):
        return self._ndiag

    def _efac_ndiag(self, pars):
        return pars[self.efacname]**2 * self.psr.toaerrs**2
    
    def _equad_ndiag(self, pars):
        return self.psr.toaerrs**2 + 10**(2 * pars[self.equadname])
    
    def _combined_ndiag(self, pars):
        return pars[self.efacname]**2 * (self.psr.toaerrs**2 + 10**(2 * pars[self.equadname]))
    
    @jax.jit
    def get_ndiag(self, pars):
        return self._get_ndiag(pars)
    
    @jax.jit
    def get_delay(self, pars):
        return jnp.zeros_like(self.psr.toas)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psr, self.has_efac, self.has_equad, self.fix_wn, self.fix_wn_vals)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
