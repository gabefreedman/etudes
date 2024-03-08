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

    def __init__(self, psr, efac=True, equad=True):
        self.psr = psr

        self.has_efac = efac
        self.has_equad = equad

        self.efacname = '{}_efac'.format(psr.name)
        self.equadname = '{}_log10_t2equad'.format(psr.name)

        self._init_ndiag(efac=self.has_efac, equad=self.has_equad)
    
    def _init_ndiag(self, efac=True, equad=True):
        if efac and equad:
            self._get_ndiag = self._combined_ndiag
        elif efac and not equad:
            self._get_ndiag = self._efac_ndiag
        elif not efac and equad:
            self._get_ndiag = self._equad_ndiag
        else:
            return # TODO: add error case for no WN signal

    def _efac_ndiag(self, pars):
        return pars[self.efacname]**2 * self.psr.toaerrs**2
    
    def _equad_ndiag(self, pars):
        return self.psr.toaerrs**2 + 10**(2 * pars[self.equadname])
    
    def _combined_ndiag(self, pars):
        return pars[self.efacname]**2 * (self.psr.toaerrs**2 + 10**(2 * pars[self.equadname]))
    
    @jax.jit
    def get_ndiag(self, pars):
        return self._get_ndiag(pars)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psr, self.has_efac, self.has_equad)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
