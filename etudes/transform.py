# -*- coding: utf-8 -*-
"""Module containing full PTA Pytree object as well as classes for coordinate
transformations on top of the base PTA. For now the only implemented
transformation is Interval, but there are plans to include a whitening
transformation class in the future.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class _EtudesPTA(object):
    """Multi-pulsar signal object. Takes in a list of Etudes1PsrSignal
    objects for every pulsar input. This will be looped over for
    likelihood calculations.
    TODO: Find a way to appropriately loop through likelihood
    function such that it can be written in terms of JAX primitives
    and break the natural looping over individual pulsars.

    :param psrs: List of pulsar objects
    :type psrs: list
    :param signalcollections: List of Etudes1PsrSignal objects
    :type signalcollections: list
    :param param_names: List of parameter names
    :type param_names: list, optional
    :param fix_wn: Flag to fix white noise in likelihood calculation
    :type fix_wn: bool, optional
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
        """Map parameter vector to dictionary of parameter names
        and values.

        :param params: Parameter vector
        :type params: jnp.Array

        :return: Dictionary of parameter names and values
        :rtype: dict
        """
        ret = {}
        for i, p in enumerate(self.param_names):
            ret[p] = params[i]
        return ret

    @jax.jit
    def ll_fn_fixwn(self, xs):
        """Log likelihood function for models with fixed white noise.

        :param xs: Parameter vector
        :type xs: jnp.Array

        :return: Log likelihood
        :rtype: float
        """
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
        """Log likelihood function for models with varied white noise.

        :param xs: Parameter vector
        :type xs: jnp.Array

        :return: Log likelihood
        :rtype: float
        """
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
        """Wrapper to call log likelihood function"""
        return self.ll_fn(xs)

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psrs, self.signalcollections, self.param_names, self.fix_wn)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        "Method for reconstructing custom PyTree"
        return cls(*children, *aux_data)


@register_pytree_node_class
class EtudesPTA(_EtudesPTA):
    """A wrapper class for the underlying _EtudesPTA class from the
    signal_combine.py file. It helps to keep most of the messy argument
    handling in a separate, hidden class, and only focus on the
    important information here.

    :param psrs: List of pulsar objects
    :type psrs: list
    :param signalcollections: List of Etudes1PsrSignal objects
    :type signalcollections: list
    :param params: List of Parameter objects
    :type params: list
    :param pmins: Lower prior bounds for all parameters
    :type pmins: jnp.Array, optional
    :param pmaxs: Upper prior bounds for all parameters
    :type pmaxs: jnp.Array, optional
    :param pavgs: Average values for all parameters' priors
    :type pavgs: jnp.Array, optional
    """

    def __init__(
        self,
        psrs,
        signalcollections,
        params,
        pmins=None,
        pmaxs=None,
        pavgs=None,
        **kwargs,
    ):
        super(EtudesPTA, self).__init__(psrs, signalcollections, **kwargs)

        self.params = params
        self.signalcollections = signalcollections
        self.param_names = [par.name for par in params]

        self.pmins = jnp.array([par.pmin for par in params])
        self.pmaxs = jnp.array([par.pmax for par in params])
        self.pavgs = (self.pmaxs + self.pmins) / 2

    @jax.jit
    def get_lnprior(self, xs):
        # For now this will just be 0 since all test
        # priors are uniform
        return 0

    @jax.jit
    def get_lnlikelihood(self, xs):
        """Wrapper to call log likelihood function"""
        return self.ll_fn(xs)

    @jax.jit
    def get_lnprob(self, xs):
        """Return log probability at input parameter vector.

        :param xs: Parameter vector
        :type xs: jnp.Array

        :return: Log probability
        :rtype: float
        """
        return self.get_lnlikelihood(xs) + self.get_lnprior(xs)

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (
            self.psrs,
            self.signalcollections,
            self.params,
            self.pmins,
            self.pmaxs,
            self.pavgs,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        "Method for reconstructing custom PyTree"
        return cls(*children, *aux_data)


@register_pytree_node_class
class Interval(EtudesPTA):
    """This class implements an interval coordinate transformation for model
    hyperparameters. It transforms the prior bounds for all parameters from
    some min and max value to positive and negative infinity.

    :param psrs: List of pulsar objects
    :type psrs: list
    :param signalcollections: List of Etudes1PsrSignal objects
    :type signalcollections: list
    :param params: List of Parameter objects
    :type params: list
    """

    def __init__(self, psrs, signalcollections, params, **kwargs):
        super(Interval, self).__init__(psrs, signalcollections, params, **kwargs)

        # Interval bounds are mins and maxes of parameter priors
        self.a = self.pmins
        self.b = self.pmaxs

    @jax.jit
    def forward(self, x):
        """Apply interval transformation to parameter vector.

        :param x: Parameter vector
        :type x: jnp.Array

        :return: Transformed parameter vector
        :rtype: jnp.Array
        """
        return jnp.log((x - self.a) / (self.b - x))

    @jax.jit
    def backward(self, p):
        """Undo the interval transformation.

        :param p: Transformed parameter vector
        :type p: jnp.Array

        :return: Parameter vector in original space
        :rtype: jnp.Array
        """
        return (self.b - self.a) * jnp.exp(p) / (1 + jnp.exp(p)) + self.a

    @jax.jit
    def dxdp(self, p):
        """Calculate Jacobian of interval transforamtion.

        :param p: Transformed parameter vector
        :type p: jnp.Array

        :return: Jacobian of interval transformation
        :rtype: jnp.Array
        """
        pp = jnp.atleast_2d(p)
        d = jnp.ones_like(pp)
        d = d.at[:].set((self.b - self.a) * jnp.exp(pp[:]) / (1 + jnp.exp(pp[:])) ** 2)
        return d.reshape(p.shape)

    @jax.jit
    def logjacobian(self, p):
        """Calculate Log of Jacobian evaluated at input
        parameter vector.

        :param p: Transformed parameter vector
        :type p: jnp.Array

        :return: Log of Jacobian evaluated at p
        :rtype: float
        """
        lj = jnp.sum(jnp.log(self.b - self.a) + p - 2 * jnp.log(1.0 + jnp.exp(p)))

        return lj

    @jax.jit
    def logjacobian_grad(self, p):
        """Calculate both the log of the Jacobian and its gradient, both
        evaluated at the input parameter vector.

        :param p: Transformed parameter vector
        :type p: jnp.Array

        :return: (Log of Jacobian, log Jacobian gradient evaluated at p)
        :rtype: tuple
        """
        lj = jnp.sum(jnp.log(self.b - self.a) + p - 2 * jnp.log(1.0 + jnp.exp(p)))

        lj_grad = jnp.zeros_like(p)
        lj_grad = (1 - jnp.exp(p)) / (1 + jnp.exp(p))
        return lj, lj_grad

    @jax.jit
    def get_lnprob(self, p):
        """Calculate log probability in transformed parameter space at
        input parameter vector. This function includes the additional
        component coming from the Jacobian
        (i.e., ret = lnlikelihood + lnprior + lnjacobian).

        :param p: Transformed parameter vector
        :type p: jnp.Array

        :return: Log probability
        :rtype: float
        """
        x = self.backward(p)
        ll = self.get_lnlikelihood(x)
        lp = self.get_lnprior(x)
        lj = self.logjacobian(p)

        return ll + lp + lj

    def tree_flatten(self):
        """Method for flattening custom PyTree"""
        return (), (self.psrs, self.signalcollections, self.params)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        "Method for reconstructing custom PyTree"
        return cls(*children, *aux_data)
