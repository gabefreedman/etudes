# -*- coding: utf-8 -*-
"""Module containing a PyTree object for collecting all signals for one
pulsar into a single class. I'm skeptical about the ability to unroll
the natural looping structure of PTA signal modeling, but this works for
limited examples.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.tree_util import register_pytree_node_class
from etudes.wn_signals import WN_Signal, WN_Signal_selec
from etudes.deterministic import CW_Signal
from etudes.gp_signals import (
    ECORR_GP_Signal,
    ECORR_GP_Signal_selec,
    Timing_Model,
    RN_Signal,
    Common_GW_Signal,
)


@register_pytree_node_class
class Etudes1PsrSignal(object):
    """Single-pulsar concatenated signal object. All potential signals
    should (key word: should) be included here as potential inputs. This
    includes white noise, intrinsic red noise, common process red noise,
    and deterministic signals.
    TODO: Find a way to appropriately loop through all given signal
    functions such that it can be written in terms of JAX primitives.

    :param T: GP basis matrix for all signals
    :type T: jax.Array
    :param TNT: matrix product :math:`T^{T}N^{-1}T` for basis matrix
        T and white noise covariance matrix N
    :type TNT: jax.Array
    :param psr: Pulsar object
    :type psr: `enterprise.pulsar.Pulsar`
    :param backend_select: Flag to separate white nosie parameters
        by receiving backend
    :type backend_select: bool, optional
    :param has_wn: Flag to include white noise signal
    :type has_wn: bool, optional
    :param has_basis_ecorr: Flag to include ECORR signal with GP basis model
    :type has_basis_ecorr: bool, optional
    :param has_rn: Flag to include intrinsic red noise signal
    :type has_rn: bool, optional
    :param has_tm: Flag to include timing model
    :type has_tm: bool, optional
    :param has_gwb: Flag to include common process red noise signal
    :type has_gwb: bool, optional
    :param has_cw: Flag to include deterministic continuous wave signal
    :type has_cw: bool, optional
    :param Umat: Basis matrix for ECORR signal
    :type Umat: jax.Array, optional
    :param ecorr_weights: Weights for ECORR GP basis
    :type ecorr_weights: jax.Array, optional
    :param Fmat: Basis matrix for red noise signals
    :type Fmat: jax.Array, optional
    :param Ffreqs: Fourier basis frequencies for red noise signals
    :type Ffreqs: jax.Array, optional
    :param efac: Flag to include EFAC parameters
    :type efac: bool, optional
    :param equad: Flag to include EQUAD parameters
    :type equad: bool, optional
    :param fix_wn: Flag to fix white noise parameters
    :type fix_wn: bool, optional
    :param fix_wn_vals: Dictionary of white noise values to
        use when fix_wn=True
    :type fix_wn_vals: dict, optional
    :param rn_comps: Number of Fourier components for intrinsic
        red noise model
    :type rn_comps: int, optional
    :param gwb_comps: Number of Fourier components for common process
        red noise model
    :type gwb_comps: int, optional
    :param tref: Reference time for continuous wave signal
    :type tref: float, optional
    """

    def __init__(
        self,
        T=None,
        TNT=None,
        psr=None,
        backend_select=False,
        has_wn=True,
        has_basis_ecorr=True,
        has_rn=True,
        has_tm=True,
        has_gwb=True,
        has_cw=True,
        Umat=None,
        ecorr_weights=None,
        Fmat=None,
        Ffreqs=None,
        efac=True,
        equad=True,
        fix_wn=True,
        fix_wn_vals=None,
        rn_comps=30,
        gwb_comps=5,
        tref=0,
    ):
        self.psr = psr
        self.backend_select = backend_select

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

        self._init_model(
            psr,
            has_wn=has_wn,
            has_basis_ecorr=has_basis_ecorr,
            has_tm=has_tm,
            has_rn=has_rn,
            has_gwb=has_gwb,
            has_cw=has_cw,
            Umat=Umat,
            ecorr_weights=ecorr_weights,
            Fmat=Fmat,
            Ffreqs=Ffreqs,
            efac=efac,
            equad=equad,
            fix_wn=fix_wn,
            fix_wn_vals=fix_wn_vals,
            rn_comps=rn_comps,
            gwb_comps=gwb_comps,
            tref=tref,
        )
        self.T, self.TNT = self._init_basis(Umat=Umat, Fmat=Fmat)
        self._init_get_delay(has_cw=has_cw)

    def _init_basis(self, Umat=None, Fmat=None, fix_wn_vals=None):
        """Initialize the basis matrix for all GP signals, including
        ECORR, intrinsic red noise, and common process red noise. Also
        calculate matrix product :math:`T^{T}N^{-1}T`

        :param Umat: Basis matrix for ECORR signal
        :type Umat: jax.Array, optional
        :param Fmat: Basis matrix for red noise signals
        :type Fmat: jax.Array, optional
        :param fix_wn_vals: Dictionary of white noise values
        :type fix_wn_vals: dict, optional

        :return: Basis matrix :math:`T` and matrix
            product :math:`T^{T}N^{-1}T`
        :rtype: tuple
        """
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
        """Initialize `get_delay` function. Functional
        form depends on whether or not a continuous wave signal
        is present in the model.

        :param has_cw: Flag to include continuous wave signal
        :type has_cw: bool, optional
        """
        if has_cw:
            self.get_delay = self._get_delay_cw
        else:
            self.get_delay = self._get_delay_nocw

    def _init_model(
        self,
        psr,
        backend_select=False,
        has_wn=True,
        has_basis_ecorr=False,
        has_tm=True,
        has_rn=True,
        has_gwb=True,
        has_cw=True,
        Umat=None,
        ecorr_weights=None,
        Fmat=None,
        Ffreqs=None,
        efac=True,
        equad=True,
        fix_wn=True,
        fix_wn_vals=None,
        rn_comps=30,
        gwb_comps=5,
        tref=0,
    ):
        """Initialize all signals for the single pulsar
        model, given all kwargs from Pytree initialization.

        :param psr: Pulsar object
        :type psr: `enterprise.pulsar.Pulsar`
        :param backend_select: Flag to separate white nosie parameters
            by receiving backend
        :type backend_select: bool, optional
        :param has_wn: Flag to include white noise signal
        :type has_wn: bool, optional
        :param has_basis_ecorr: Flag to include ECORR signal with GP basis model
        :type has_basis_ecorr: bool, optional
        :param has_tm: Flag to include timing model
        :type has_tm: bool, optional
        :param has_rn: Flag to include intrinsic red noise signal
        :type has_rn: bool, optional
        :param has_gwb: Flag to include common process red noise signal
        :type has_gwb: bool, optional
        :param has_cw: Flag to include deterministic continuous wave signal
        :type has_cw: bool, optional
        :param Umat: Basis matrix for ECORR signal
        :type Umat: jax.Array, optional
        :param ecorr_weights: Weights for ECORR GP basis
        :type ecorr_weights: jax.Array, optional
        :param Fmat: Basis matrix for red noise signals
        :type Fmat: jax.Array, optional
        :param Ffreqs: Fourier basis frequencies for red noise signals
        :type Ffreqs: jax.Array, optional
        :param efac: Flag to include EFAC parameters
        :type efac: bool, optional
        :param equad: Flag to include EQUAD parameters
        :type equad: bool, optional
        :param fix_wn: Flag to fix white noise parameters
        :type fix_wn: bool, optional
        :param fix_wn_vals: Dictionary of white noise values to
            use when fix_wn=True
        :type fix_wn_vals: dict, optional
        :param rn_comps: Number of Fourier components for intrinsic
            red noise model
        :type rn_comps: int, optional
        :param gwb_comps: Number of Fourier components for common process
            red noise model
        :type gwb_comps: int, optional
        :param tref: Reference time for continuous wave signal
        :type tref: float, optional
        """
        if has_wn:
            if backend_select:
                self.wn_signal = WN_Signal_selec(
                    psr, efac=efac, equad=equad, fix_wn=fix_wn, fix_wn_vals=fix_wn_vals
                )
            else:
                self.wn_signal = WN_Signal(
                    psr, efac=efac, equad=equad, fix_wn=fix_wn, fix_wn_vals=fix_wn_vals
                )
        if has_basis_ecorr:
            if backend_select:
                self.basis_ecorr_signal = ECORR_GP_Signal_selec(
                    psr,
                    Umat=Umat,
                    weights=ecorr_weights,
                    fix_wn=fix_wn,
                    fix_wn_vals=fix_wn_vals,
                )
            else:
                self.basis_ecorr_signal = ECORR_GP_Signal(
                    psr,
                    Umat=Umat,
                    weights=ecorr_weights,
                    fix_wn=fix_wn,
                    fix_wn_vals=fix_wn_vals,
                )
        if has_tm:
            self.tm_signal = Timing_Model(psr)
        if has_rn:
            self.rn_signal = RN_Signal(psr, Fmat=Fmat, Ffreqs=Ffreqs, ncomps=rn_comps)
        if has_gwb:
            self.gwb_signal = Common_GW_Signal(
                psr,
                Fmat=Fmat[:, : 2 * gwb_comps],
                Ffreqs=Ffreqs[: 2 * gwb_comps],
                ncomps=gwb_comps,
            )
        if has_cw:
            self.cw_signal = CW_Signal(psr, tref=tref)

    def _init_phi_fn(self, has_basis_ecorr=False, has_rn=True, has_gwb=True):
        """Initialize the `get_phi` function. Functional form depends
        on what GP signals are present.

        :param has_basis_ecorr: Flag to include ECORR signal
        :type has_basis_ecorr: bool, optional
        :param has_rn: Flag to include intrinsic red noise signal
        :type has_rn: bool, optional
        :param has_gwb: Flag to include common process red noise signal
        :type has_gwb: bool, optional
        """
        # get_phi changes form based on presence of ECORR, red-noise signal,
        # common-process signals... this will be a long bunch of if/else statements
        if has_basis_ecorr and has_rn and has_gwb:
            self._get_phi = self.get_phi_full
        elif has_basis_ecorr and has_rn and not has_gwb:
            self._get_phi = self.get_phi_ecorr_rn
        elif not has_basis_ecorr and has_rn and has_gwb:
            self._get_phi = self.get_phi_rn_gwb
        elif not has_basis_ecorr and has_rn and not has_gwb:
            self._get_phi = self.get_phi_rn_only
        return

    @jax.jit
    def _get_delay_cw(self, pars):
        """Calculate total delay from all deterministic signals
        in instance where model contains a continuous wave.

        :param pars: Input parameter values
        :type pars: dict

        :return: Deterministic signal delay
        :rtype: jax.Array
        """
        return self.wn_signal.get_delay(pars) + self.cw_signal.get_delay(pars)

    @jax.jit
    def _get_delay_nocw(self, pars):
        """Calculate total delay when no continuous wave signal is
        in the model (should return array of 0).

        :param pars: Input parameter values
        :type pars: dict

        :return: Deterministic signal delay
        :rtype: jax.Array
        """
        return self.wn_signal.get_delay(pars)

    @jax.jit
    def get_ndiag(self, pars):
        """Calculate white noise covariance matrix.

        :param pars: Input parameter values
        :type pars: dict

        :return: White noise covariance matrix
        :rtype: jax.Array
        """
        return self.wn_signal.get_ndiag(pars)

    # Different function variations for get_phi
    def get_phi_full(self, pars):
        """Phi matrix calculation for GP ECORR, intrinsic red noise,
        common process red noise, and timing model signals.

        :param pars: Input parameter values
        :type pars: dict

        :return: Phi matrix
        :rtype: jax.Array
        """
        ecorr_phi = self.basis_ecorr_signal.get_phi(pars)
        rn_phi = self.rn_signal.get_phi(pars)
        gw_phi = self.gwb_signal.get_phi(pars)
        rn_phi = rn_phi.at[: 2 * self.gwb_comps].add(gw_phi)
        tm_phi = self.tm_signal.get_phi(pars)
        return jnp.concatenate([ecorr_phi, rn_phi, tm_phi])

    def get_phi_rn_only(self, pars):
        """Phi matrix calculation for intrinsic red noise
        and timing model signals.

        :param pars: Input parameter values
        :type pars: dict

        :return: Phi matrix
        :rtype: jax.Array
        """
        rn_phi = self.rn_signal.get_phi(pars)
        tm_phi = self.tm_signal.get_phi(pars)
        return jnp.concatenate([rn_phi, tm_phi])

    def get_phi_rn_gwb(self, pars):
        """Phi matrix calculation for intrinsic red noise,
        common process red noise, and timing model signals.

        :param pars: Input parameter values
        :type pars: dict

        :return: Phi matrix
        :rtype: jax.Array
        """
        rn_phi = self.rn_signal.get_phi(pars)
        gw_phi = self.gwb_signal.get_phi(pars)
        rn_phi = rn_phi.at[: 2 * self.gwb_comps].add(gw_phi)
        tm_phi = self.tm_signal.get_phi(pars)
        return jnp.concatenate([rn_phi, tm_phi])

    def get_phi_ecorr_rn(self, pars):
        """Phi matrix calculation for GP ECORR, intrinsic red
        noise, and timing model signals.

        :param pars: Input parameter values
        :type pars: dict

        :return: Phi matrix
        :rtype: jax.Array
        """
        ecorr_phi = self.basis_ecorr_signal.get_phi(pars)
        rn_phi = self.rn_signal.get_phi(pars)
        tm_phi = self.tm_signal.get_phi(pars)
        return jnp.concatenate([ecorr_phi, rn_phi, tm_phi])

    @jax.jit
    def get_phi(self, pars):
        """Wrapper function to call the appropriate `get_phi`.

        :param pars: Input parameter values
        :type pars: dict

        :return: Phi matrix
        :rtype: jax.Array
        """
        return self._get_phi(pars)

    @jax.jit
    def get_phiinv_logdet(self, pars):
        """Calculate the inverse of the phi matrix and
        its log determinant.

        :param pars: Input parameter values
        :type pars: dict

        :return: (Inverse of phi matrix, log determinant of phi)
        :rtype: tuple
        """
        phi = self.get_phi(pars)
        return 1.0 / phi, jnp.sum(jnp.log(phi))

    @jax.jit
    def get_detres(self, pars):
        """Calculate deterministic signal delay and subtract
        from pulsar residuals.

        :param pars: Input parameter values
        :type pars: dict

        :return: Subtracted residuals
        :rtype: jax.Array
        """
        return self.psr.residuals - self.get_delay(pars)

    @jax.jit
    def get_TNr(self, pars):
        """Calculate matrix product :math:`T^{T}N^{-1}r`, where T is the
        GP basis matrix, N the white noise covariance matrix, and r
        the subtracted residuals.

        :param pars: Input parameter values
        :type pars: dict

        :return: :math:`T^{T}N^{-1}r`
        :rtype: jax.Array
        """
        mult = self.get_detres(pars) / self.get_ndiag(pars)
        return jnp.dot(self.T.T, mult)

    @jax.jit
    def get_TNT(self, pars):
        """Calculate matrix product :math:`T^{T}N^{-1}T`, where T is the
        GP basis matrix, N the white noise covariance matrix.

        :param pars: Input parameter values
        :type pars: dict

        :return: :math:`T^{T}N^{-1}T`
        :rtype: jax.Array
        """
        mult = self.T / self.get_ndiag(pars)[:, None]
        return jnp.dot(self.T.T, mult)

    @jax.jit
    def get_rNr_logdet(self, pars):
        """Calculate :math:`r^{T}N^{-1}r` and log determinant of N, where
        r is the subtracted residuals and N the white noise covariance
        matrix.

        :param pars: Input parameter values
        :type pars: dict

        :return: (:math:`r^{T}N^{-1}r`, log determinant of N)
        :rtype: tuple
        """
        r = self.get_detres(pars)
        N = self.get_ndiag(pars)
        mult = r / N
        ret = jnp.dot(r.T, mult)
        logdet = jnp.sum(jnp.log(N))
        return ret + logdet

    @jax.jit
    def get_ll(self, pars):
        """Calculate log likelihood given some input parameters.

        :param pars: Input parameter values
        :type pars: dict

        :return: Log likelihood
        :rtype: float
        """
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

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (self.T, self.TNT), (
            self.psr,
            self.backend_select,
            self.has_wn,
            self.has_basis_ecorr,
            self.has_rn,
            self.has_tm,
            self.has_gwb,
            self.has_cw,
            self.Umat,
            self.ecorr_weights,
            self.Fmat,
            self.Ffreqs,
            self.efac,
            self.equad,
            self.fix_wn,
            self.fix_wn_vals,
            self.rn_comps,
            self.gwb_comps,
            self.tref,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        "Method for reconstructing custom PyTree"
        return cls(*children, *aux_data)
