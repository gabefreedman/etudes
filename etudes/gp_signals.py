# -*- coding: utf-8 -*-
"""PyTree objects and helper functions for modeling Gaussian Process (GP)
signals. This module includes intrinsic-pulsar red noise signals, a GP
signal representation of ECORR, and common-process red noise signals such
as the GWB (Correlations currently not implemented).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import etudes.constants as const


def create_fourierdesignmatrix_red(toas, nmodes=30):
    """Construct the Fourier design matrix and associated basis
    frequencies, as defined by eq. 11 of Lentati et al. (2013).

    :param toas: Vector of times of arrival [s]
    :type toas: array-like
    :param nmodes: Number of Fourier modes to use for constructing basis
    :type nmodes: int, optional

    :return: (F, Ffreqs), where F is the Fourier design matrix and Ffreqs
        are the sampling basis frequencies
    :rtype: tuple
    """
    N = jnp.size(toas)
    F = jnp.zeros((N, 2 * nmodes))
    Tspan = jnp.max(toas) - jnp.min(toas)

    f = 1.0 * jnp.arange(1, nmodes + 1) / Tspan
    Ffreqs = jnp.repeat(f, 2)

    F = F.at[:, ::2].set(jnp.sin(2 * jnp.pi * toas[:, None] * f[None, :]))
    F = F.at[:, 1::2].set(jnp.cos(2 * jnp.pi * toas[:, None] * f[None, :]))

    return F, Ffreqs


def create_quantization_matrix(toas, dt=1, nmin=2):
    """Map TOAs to observing epochs via a quantization matrix.

    :param toas: Vector of times of arrival [s]
    :type toas: array-like
    :param dt: Maximum time difference between epochs [s]
    :type dt: float, optional
    :param nmin: Minimum number of TOAs per epoch
    :type nmin: int, optional

    :return: (U, weights), where U is the quantization matrix
        and weights are the basis weights for each epoch
        (by default all weights are 1)
    :rtype: tuple
    """
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
    """Create quantization matrix for models with ECORR signal
    is broken up by receiver backend.

    :param psr: single Pulsar object
    :dtype psr: `enterprise.pulsar.Pulsar`

    :return: (Umat, weights), where Umat is the quantization matrix
        and weights the basis weights.
    :rtype: tuple
    """
    Umat, _ = create_quantization_matrix(psr.toas)
    weights = []

    backends = np.unique(psr.backend_flags)
    for val in backends:
        mask = psr.backend_flags == val
        weights.append(create_quantization_matrix(psr.toas[jnp.nonzero(mask)])[1])

    return Umat, weights


@register_pytree_node_class
class ECORR_GP_Signal(object):
    """Class for correlated white noise signals (ECORR)
    modeled with GPs. Main output function is `get_phi`.
    If Umat and weights are not supplied they will be
    calculated at initialization (and may be recompiled
    each time `get_phi` is run).

    :param psr: A Pulsar object
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param Umat: Quantization matrix for input pulsar's TOAs
    :type Umat: array-like, optional
    :param weights: Basis weights for each epoch
    :type weights: array-like, optional
    :param fix_wn: Flag for fixing ECORR values at constants
    :type fix_wn: bool, optional
    :param fix_wn_vals: Dictionary of ECORR values to use when fix_wn=True
    :type fix_wn_vals: dict, optional
    """

    def __init__(
        self, psr, Umat=None, weights=None, fix_wn=True, fix_wn_vals=True, **kwargs
    ):
        self.psr = psr
        self.ecorrname = "{}_basis_ecorr_log10_ecorr".format(psr.name)

        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        if isinstance(Umat, jax.Array) and isinstance(weights, jax.Array):
            self.Umat = Umat
            self.weights = weights
        else:
            self.Umat, self.weights = create_quantization_matrix(psr.toas)

    def _ecorr_prior(self, pars):
        """Calculate portion of GP signal prior covariance matrix
        for ECORR signal.

        :param pars: Dictionary of parameter values
        :type pars: dict

        :return: ECORR prior covariance matrix
        :rtype: array-like
        """
        return self.weights * 10 ** (2 * pars[self.ecorrname])

    @jax.jit
    def get_phi(self, pars):
        """Wrapper to return prior covariance matrix for ECORR signal.

        :param pars: Dictionary of parameter values
        :type pars: dict

        :return: ECORR prior covariance matrix
        :rtype: array-like
        """
        return self._ecorr_prior(pars)

    @jax.jit
    def get_delay(self, pars):
        """Stochastic signal, get_delay returns array of zeros.

        :param pars: Dictionary of parameter values
        :type pars: dict
        """
        return jnp.zeros_like(self.psr.toas)

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psr, self.Umat, self.weights, self.fix_wn, self.fix_wn_vals)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        return cls(*children, *aux_data)


@register_pytree_node_class
class ECORR_GP_Signal_selec(object):
    """Class for correlated white noise signals (ECORR)
    modeled with GPs. Main output function is `get_phi`.
    ECORR signal parameters are split up by receiver backend.

    :param psr: A Pulsar object
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param Umat: Quantization matrix for input pulsar's TOAs
    :type Umat: array-like, optional
    :param weights: Basis weights for each epoch
    :type weights: array-like, optional
    :param fix_wn: Flag for fixing ECORR values at constants
    :type fix_wn: bool, optional
    :param fix_wn_vals: Dictionary of ECORR values to use when fix_wn=True
    :type fix_wn_vals: dict, optional
    """

    def __init__(
        self, psr, Umat=None, weights=None, fix_wn=True, fix_wn_vals=True, **kwargs
    ):
        self.psr = psr

        self.fix_wn = fix_wn
        self.fix_wn_vals = fix_wn_vals

        self._select_by_backend(psr, fix_wn_vals)

        if isinstance(Umat, jax.Array) and isinstance(weights, jax.Array):
            self.Umat = Umat
            self.weights = weights
        else:
            self.Umat, self.weights = create_quant_matrix_selec(psr)

        if fix_wn:
            self._init_fix_phi()
        else:
            self._init_vary_phi()

    def _select_by_backend(self, psr, fix_wn_vals):
        """Divide up ECORR signal parameters by receiver backend.
        This function defines the self.ecorrs attribute.

        :param psr: A Pulsar object
        :type psr: :class:`enterprise.pulsar.Pulsar`
        :param fix_wn_vals: Dictionary of ECORR values (split by backend)
        :type fix_wn_vals: dict

        :return: None
        """
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

    def _init_fix_phi(self):
        """Initialize the ECORR `get_phi` function for fixed ECORR
        values.
        """
        phislcs = []
        for i, ecorr in enumerate(self.ecorrs):
            phislcs.append(self.weights[i] * 10 ** (2 * ecorr))

        self._phi = jnp.concatenate(phislcs)
        self._get_phi = self._return_fix_phi

    def _init_vary_phi(self):
        # TODO: write case for varying ECORR
        return

    def _return_fix_phi(self, pars):
        """Wrapper to return fixed phi matrix."""
        return self._phi

    @jax.jit
    def get_phi(self, pars):
        """Wrapper to return prior covariance matrix for ECORR
        signal. Valid for fixed or varied ECORR parameters.
        """
        # note: calling self._get_phi recompiles
        return self._get_phi(pars)

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psr, self.Umat, self.weights, self.fix_wn, self.fix_wn_vals)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        return cls(*children, *aux_data)


@register_pytree_node_class
class RN_Signal(object):
    """Class for intrinsic-pulsar red noise signals.
    Main output is `get_phi`. If Fmat and Ffreqs are
    not supplied they will calculated at initialization,
    and then possibly recompiled every subsequent time.

    :param psr: A Pulsar object
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param Fmat: Fourier design matrix
    :type Fmat: array-like, optional
    :param Ffreqs: Fourier basis frequencies
    :type Ffreqs: array-like, optional
    :param ncomps: Number of Fourier modes to use
    :type ncomps: int, optional
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

    def _powerlaw(self, pars):
        """Power law prior for red noise signals.

        :param pars: Dictionary of parameter values
        :type pars: dict

        :return: red noise prior covariance from power law
        :rtype: array-like
        """
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
        """Wrapper to return prior covariance matrix
        for intrinsic red noise signal. Currently only implmented with
        power law prior.

        :param pars: Dictionary of parameter values
        :type pars: dict

        :return: red noise prior covariance matrix
        :rtype: array-like
        """
        return self._powerlaw(pars)

    @jax.jit
    def get_delay(self, pars):
        """Stochastic signal, get_delay returns array of zeros.

        :param pars: Dictionary of parameter values
        :type pars: dict
        """
        return jnp.zeros_like(self.psr.toas)

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (
            self.Fmat,
            self.Ffreqs,
        ), (self.psr, self.ncomps)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        psr, ncomps = aux_data
        (
            Fmat,
            Ffreqs,
        ) = children
        return cls(psr, Fmat, Ffreqs, ncomps)


@register_pytree_node_class
class Common_GW_Signal(object):
    """Class for common process (common across all
    pulsars) red noise signals. Main output is
    `get_phi`. If Fmat and Ffreqs are not supplied
    they will calculated at initialization, and then
    possibly recompiled every subsequent time.

    :param psr: A Pulsar object
    :type psr: :class:`enterprise.pulsar.Pulsar`
    :param Fmat: Fourier design matrix
    :type Fmat: array-like, optional
    :param Ffreqs: Fourier basis frequencies
    :type Ffreqs: array-like, optional
    :param ncomps: Number of Fourier modes to use
        for common basis
    :type ncomps: int, optional
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

    def _powerlaw(self, pars):
        """Power law prior for red noise signals.

        :param pars: Dictionary of parameter values
        :type pars: dict

        :return: red noise prior covariance from power law
        :rtype: array-like
        """
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
        """Wrapper to return prior covariance matrix
        for common process red noise signal. Currently
        only implmented with power law prior.

        :param pars: Dictionary of parameter values
        :type pars: dict

        :return: red noise prior covariance matrix
        :rtype: array-like
        """
        return self._powerlaw(pars)

    @jax.jit
    def get_delay(self, pars):
        """Stochastic signal, get_delay returns array of zeros.

        :param pars: Dictionary of parameter values
        :type pars: dict
        """
        return jnp.zeros_like(self.psr.toas)

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (
            self.Fmat,
            self.Ffreqs,
        ), (self.psr, self.ncomps)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method for reconstructing custom PyTree"""
        psr, ncomps = aux_data
        (
            Fmat,
            Ffreqs,
        ) = children
        return cls(psr, Fmat, Ffreqs, ncomps)


@register_pytree_node_class
class Timing_Model(object):
    """Class for linearized timing model modeled as
    a GP process.

    :param psr: A Pulsar object
    :type psr: :class:`enterprise.pulsar.Pulsar`
    """

    def __init__(self, psr):
        self.psr = psr

        self.Mmat = psr.Mmat
        self.weights = jnp.ones(psr.Mmat.shape[1])

    def _tm_prior(self, pars):
        """Fixed prior covariance for timing model signal."""
        return self.weights * 1e40

    @jax.jit
    def get_phi(self, pars):
        """Wrapper to return prior covariance matrix
        for timing model.

        :param pars: Dictionary of parameter values
        :type pars: dict

        :return: red noise prior covariance matrix
        :rtype: array-like
        """
        return self._tm_prior(pars)

    @jax.jit
    def get_delay(self, pars):
        """Stochastic signal, get_delay returns array of zeros.

        :param pars: Dictionary of parameter values
        :type pars: dict
        """
        return jnp.zeros_like(self.psr.toas)

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psr,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        "Method for reconstructing custom PyTree"
        return cls(*children, *aux_data)
