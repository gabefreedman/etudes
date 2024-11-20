"""
Contains functions related to modeling of Gaussian Process (GP) signals.
This includes intrinsic-pulsar red noise signals, a GP signal
representation of ECORR, and common-process red noise signals such
as the GWB (Correlations currently not implemented)
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import etudes.constants as const

jax.config.update("jax_enable_x64", True)


def create_fourierdesignmatrix_red(toas, nmodes=30):
    N = jnp.size(toas)
    F = jnp.zeros((N, 2 * nmodes))
    Tspan = jnp.max(toas) - jnp.min(toas)

    f = 1.0 * jnp.arange(1, nmodes + 1) / Tspan
    Ffreqs = jnp.repeat(f, 2)

    F = F.at[:, ::2].set(jnp.sin(2 * jnp.pi * toas[:, None] * f[None, :]))
    F = F.at[:, 1::2].set(jnp.cos(2 * jnp.pi * toas[:, None] * f[None, :]))

    return F, Ffreqs


def create_quantization_matrix(toas, dt=1, nmin=2):
    isort = jnp.argsort(toas)

    bucket_ref = [toas[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if toas[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas[i])
            bucket_ind.append([i])

    bucket_ind2 = [ind for ind in bucket_ind if len(ind) >= nmin]

    U = jnp.zeros((len(toas), len(bucket_ind2)), "d")
    for i, l in enumerate(bucket_ind2):
        U = U.at[l, i].set(1)

    weights = jnp.ones(U.shape[1])

    return U, weights


def create_quant_matrix_selec(psr):
    # loops through create_quantization matrix for instance where there
    # is backend selection on ECORR. This is to extract a for loop outside
    # of anything that would be JIT-compiled

    # need the full basis matrix (U) for all toas
    # need the per-backend basis weights
    Umat, _ = create_quantization_matrix(psr.toas)
    weights = []

    backends = np.unique(psr.backend_flags)
    for val in backends:
        mask = psr.backend_flags == val
        weights.append(create_quantization_matrix(psr.toas[jnp.nonzero(mask)])[1])

    return Umat, weights


@register_pytree_node_class
class ECORR_GP_Signal(object):
    """
    Class for correlated white noise signal modeled with GPs
    """

    def __init__(
        self, psr, Umat=None, weights=None, fix_wn=True, fix_wn_vals=True, **kwargs
    ):
        self.psr = psr

        self.ecorrname = "{}_basis_ecorr_log10_ecorr".format(psr.name)

        self.Umat = Umat
        self.weights = weights

        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

    def _init_basis(self, psr):
        # TODO: this recomputes at every step, should only be at initialization
        self.Umat, self.weights = create_quantization_matrix(psr.toas)

    def _ecorr_prior(self, pars):
        return self.weights * 10 ** (2 * pars[self.ecorrname])

    @jax.jit
    def get_phi(self, pars):
        return self._ecorr_prior(pars)

    @jax.jit
    def get_delay(self, pars):
        return jnp.zeros_like(self.psr.toas)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psr, self.Umat, self.weights, self.fix_wn, self.fix_wn_vals)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


@register_pytree_node_class
class ECORR_GP_Signal_selec(object):
    """
    Correlated white noise signal modeled with GPs and split
    by backend
    """

    def __init__(
        self, psr, Umat=None, weights=None, fix_wn=True, fix_wn_vals=True, **kwargs
    ):
        self.psr = psr

        self.Umat = Umat
        self.weights = weights

        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        self._select_by_backend(psr, fix_wn_vals)

        if fix_wn:
            self._init_fix_phi(psr)
        else:
            self._init_vary_phi(psr)

    def _select_by_backend(self, psr, fix_wn_vals):
        backends = np.unique(psr.backend_flags)
        self.masks = jnp.ones((backends.shape[0], psr.toas.shape[0]))
        self.ecorrs = jnp.zeros(backends.shape[0])

        for i, val in enumerate(backends):
            mask_bool = psr.backend_flags == val
            self.masks = self.masks.at[i, :].set(
                np.ma.array(
                    np.ones(psr.toas.shape[0]), mask=~mask_bool, fill_value=0.0
                ).filled()
            )
            self.ecorrs = self.ecorrs.at[i].set(
                fix_wn_vals["_".join([psr.name, "basis", "ecorr", val, "log10_ecorr"])]
            )
        return

    def _init_fix_phi(self, psr):
        phislcs = []
        for i, ecorr in enumerate(self.ecorrs):
            phislcs.append(self.weights[i] * 10 ** (2 * ecorr))

        self._phi = jnp.concatenate(phislcs)
        self._get_phi = self._return_fix_phi

    def _init_vary_phi(self, psr):
        # TODO: write case for varying ECORR
        return

    def _return_fix_phi(self, pars):
        return self._phi

    @jax.jit
    def get_phi(self, pars):
        # need to figure out why calling self._get_phi recompiles
        return self._get_phi(pars)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psr, self.Umat, self.weights, self.fix_wn, self.fix_wn_vals)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)


@register_pytree_node_class
class RN_Signal(object):
    """
    Class for intrinsic-pulsar red noise signals
    (or uncorrelated common signals... they're modeled
    the same way)
    """

    def __init__(self, psr, Fmat=None, Ffreqs=None, ncomps=30):
        self.psr = psr
        self.ncomps = ncomps

        self.rn_A_name = "{}_rn_log10_A".format(psr.name)
        self.rn_gamma_name = "{}_rn_gamma".format(psr.name)

        if isinstance(Fmat, jax.Array) and isinstance(Ffreqs, jax.Array):
            self.Fmat = Fmat
            self.Ffreqs = Ffreqs
        else:
            self.Fmat, self.Ffreqs = create_fourierdesignmatrix_red(
                psr.toas, nmodes=ncomps
            )

    def _init_basis(self, psr):
        return create_fourierdesignmatrix_red(psr.toas)

    def _powerlaw(self, pars):
        df = jnp.diff(jnp.concatenate((jnp.array([0]), self.Ffreqs[::2])))
        return (
            self.Ffreqs ** (-pars[self.rn_gamma_name])
            * (10 ** pars[self.rn_A_name]) ** 2
            / 12.0
            / jnp.pi**2
            * const.fyr ** (pars[self.rn_gamma_name] - 3)
            * jnp.repeat(df, 2)
        )

    @jax.jit
    def get_phi(self, pars):
        return self._powerlaw(pars)

    @jax.jit
    def get_delay(self, pars):
        return jnp.zeros_like(self.psr.toas)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (
            self.Fmat,
            self.Ffreqs,
        ), (self.psr, self.ncomps)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        psr, ncomps = aux_data
        (
            Fmat,
            Ffreqs,
        ) = children
        return cls(psr, Fmat, Ffreqs, ncomps)


@register_pytree_node_class
class Common_GW_Signal(object):
    """
    Class for common-process red noise signal
    """

    def __init__(self, psr, Fmat=None, Ffreqs=None, ncomps=5):
        self.psr = psr
        self.ncomps = ncomps

        self.gw_A_name = "gamma_gw"
        self.gw_gamma_name = "log10_A_gw"

        if isinstance(Fmat, jax.Array) and isinstance(Ffreqs, jax.Array):
            self.Fmat = Fmat
            self.Ffreqs = Ffreqs
        else:
            self.Fmat, self.Ffreqs = create_fourierdesignmatrix_red(
                psr.toas, nmodes=ncomps
            )

    def _init_basis(self, psr):
        return create_fourierdesignmatrix_red(psr.toas)

    def _powerlaw(self, pars):
        df = jnp.diff(jnp.concatenate((jnp.array([0]), self.Ffreqs[::2])))
        return (
            self.Ffreqs ** (-pars[self.gw_gamma_name])
            * (10 ** pars[self.gw_A_name]) ** 2
            / 12.0
            / jnp.pi**2
            * const.fyr ** (pars[self.gw_gamma_name] - 3)
            * jnp.repeat(df, 2)
        )

    @jax.jit
    def get_phi(self, pars):
        return self._powerlaw(pars)

    @jax.jit
    def get_delay(self, pars):
        return jnp.zeros_like(self.psr.toas)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (
            self.Fmat,
            self.Ffreqs,
        ), (self.psr, self.ncomps)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        psr, ncomps = aux_data
        (
            Fmat,
            Ffreqs,
        ) = children
        return cls(psr, Fmat, Ffreqs, ncomps)


@register_pytree_node_class
class Timing_Model(object):
    """
    Class for linearized timing model
    """

    def __init__(self, psr):
        self.psr = psr

        self.Mmat = psr.Mmat
        self.weights = jnp.ones(psr.Mmat.shape[1])

    def _tm_prior(self, pars):
        return self.weights * 1e40

    @jax.jit
    def get_phi(self, pars):
        return self._tm_prior(pars)

    @jax.jit
    def get_delay(self, pars):
        return jnp.zeros_like(self.psr.toas)

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psr,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
