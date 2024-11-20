"""
Contains functions related to modeling of white noise (WN) signals.
This includes WN signals common to PTA analyses such as
EFAC and EQUAD. Correlated WN signals such as ECORR require a bit more
care to be JAX-compatible and will be added later.
"""

# need numpy import for selection funcions (np.unique)
import numpy as np

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

        self.efacname = "{}_efac".format(psr.name)
        self.equadname = "{}_log10_t2equad".format(psr.name)

        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        if fix_wn:
            self._init_fix_ndiag(
                efac=self.has_efac, equad=self.has_equad, fix_wn_vals=fix_wn_vals
            )
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
            return  # TODO: add case for no WN signal

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
        return pars[self.efacname] ** 2 * self.psr.toaerrs**2

    def _equad_ndiag(self, pars):
        return self.psr.toaerrs**2 + 10 ** (2 * pars[self.equadname])

    def _combined_ndiag(self, pars):
        return pars[self.efacname] ** 2 * (
            self.psr.toaerrs**2 + 10 ** (2 * pars[self.equadname])
        )

    @jax.jit
    def get_ndiag(self, pars):
        return self._get_ndiag(pars)

    @jax.jit
    def get_delay(self, pars):
        return jnp.zeros_like(self.psr.toas)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (
            self.psr,
            self.has_efac,
            self.has_equad,
            self.fix_wn,
            self.fix_wn_vals,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


@register_pytree_node_class
class WN_Signal_selec(object):
    """
    White noise signal class for models where noise is broken
    up by backend for each pulsar
    """

    def __init__(
        self, psr=None, efac=True, equad=True, fix_wn=True, fix_wn_vals=None, **kwargs
    ):
        self.psr = psr

        self.has_efac = efac
        self.has_equad = equad

        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        self._select_by_backend(psr, fix_wn_vals)

        if fix_wn:
            self._init_fix_ndiag(
                efac=self.has_efac, equad=self.has_equad, fix_wn_vals=fix_wn_vals
            )
        else:
            self._init_vary_ndiag(efac=self.has_efac, equad=self.has_equad)

    def _select_by_backend(self, psr, fix_wn_vals):
        backends = np.unique(psr.backend_flags)
        self.masks = jnp.ones((backends.shape[0], psr.toas.shape[0]))
        self.efacs = jnp.zeros(backends.shape[0])
        self.equads = jnp.zeros(backends.shape[0])

        for i, val in enumerate(backends):
            mask_bool = psr.backend_flags == val
            self.masks = self.masks.at[i, :].set(
                np.ma.array(
                    np.ones(psr.toas.shape[0]), mask=~mask_bool, fill_value=0.0
                ).filled()
            )
            self.efacs = self.efacs.at[i].set(
                fix_wn_vals["_".join([psr.name, val, "efac"])]
            )
            self.equads = self.equads.at[i].set(
                fix_wn_vals["_".join([psr.name, val, "log10_t2equad"])]
            )
        return

    def _init_fix_ndiag(self, efac=True, equad=True, fix_wn_vals=None):
        if efac and equad:
            self._ndiag = (
                self.efacs**2
                @ self.masks
                * (self.psr.toaerrs**2 + 10 ** (2 * self.equads) @ self.masks)
            )
        elif efac and not equad:
            self._ndiag = self.efacs**2 @ self.masks * self.psr.toaerrs**2
        elif not efac and equad:
            self._ndiag = self.psr.toaerrs**2 + 10 ** (2 * self.equads) @ self.masks
        else:
            pass
        # this `should` cache ndiag and not recompute it each time
        self._get_ndiag = self._return_fix_ndiag
        return

    def _init_vary_ndiag(self):
        # TODO: write vary white-noise case
        return

    def _return_fix_ndiag(self, pars):
        return self._ndiag

    @jax.jit
    def get_ndiag(self, pars):
        return self._get_ndiag(pars)

    @jax.jit
    def get_delay(self, pars):
        return jnp.zeros_like(self.psr.toas)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (
            self.psr,
            self.has_efac,
            self.has_equad,
            self.fix_wn,
            self.fix_wn_vals,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
