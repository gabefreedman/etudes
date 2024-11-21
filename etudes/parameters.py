# -*- coding: utf-8 -*-
"""Module with all parameter classes. These classes are
meant to be very lightweight containers of parameter
information such as names and prior bounds. No need to make it
more complicated than that.
"""

import numpy as np
import jax.numpy as jnp


class Parameter(object):
    """Base parameter class with minimum working functions.

    :param pmin: Parameter lower bound
    :type pmin: float
    :param pmax: Parameter upper bound
    :type pmax: float
    :param name: Optional name for parameter
    :type name: str, optional
    :param idx: Index in full parameter array for entire PTA model
    :type idx: int, optional
    """

    def __init__(self, pmin, pmax, name=None, idx=None):
        self.pmin = pmin
        self.pmax = pmax

        self.name = name
        self.idx = idx

    def get_prior(self):
        """Placeholder for specific prior function"""
        return 0

    def get_logprior(self):
        """Placeholder for specific log prior function"""
        return 0

    def sample(self):
        """Placeholder for sampling from prior"""
        return 0


class Uniform(Parameter):
    """Class for parameters with uniform distributions (i.e.,
    pdf(x) = 1/(pmax - pmin) for pmin <= x <= pmax).

    :param pmin: Parameter lower bound
    :type pmin: float
    :param pmax: Parameter upper bound
    :type pmax: float
    :param name: Optional name for parameter
    :type name: str, optional
    :param idx: Index in full parameter array for entire PTA model
    :type idx: int, optional
    """

    def __init__(self, pmin, pmax, name=None, idx=None):
        super(Uniform, self).__init__(pmin, pmax, name, idx)

    def get_prior(self, value):
        """Return prior probability for given value.

        :param value: Value at which to evaluate prior
        :type value: float

        :return: Prior probability evaluated at value
        :rtype: float
        """
        return ((value >= self.pmin) & (value <= self.pmax)) / (self.pmax - self.pmin)

    def get_lnprior(self, value):
        """Return log prior probability (0 for uniform distribution).

        :param value: Value at which to evaluate log prior
        :type value: float

        :return: 0 (log prior for all values in uniform distribution)
        :rtype: float
        """
        return 0.0

    def sample(self):
        """Sample from uniform distribution.

        :return: Random sample draw
        :rtype: float
        """
        return np.random.uniform(self.pmin, self.pmax)


class LinearExp(Parameter):
    """Class for parameters with log-uniform distributions (i.e.,
    pdf(x) ~ 10^x / (10^pmax - 10^pmin) for pmin <= x <= pmax).

    :param pmin: Parameter lower bound
    :type pmin: float
    :param pmax: Parameter upper bound
    :type pmax: float
    :param name: Optional name for parameter
    :type name: str, optional
    :param idx: Index in full parameter array for entire PTA model
    :type idx: int, optional
    """

    def __init__(self, pmin, pmax, name=None, idx=None):
        super(LinearExp, self).__init__(pmin, pmax, name, idx)

    def get_prior(self, value):
        """Return prior probability for given value.

        :param value: Value at which to evaluate prior
        :type value: float

        :return: Prior probability evaluated at value
        :rtype: float
        """
        return (
            ((self.pmin <= value) & (value <= self.pmax))
            * np.log(10)
            * 10**value
            / (10**self.pmax - 10**self.pmin)
        )

    def get_lnprior(self, value):
        """Return log prior probability.

        :param value: Value at which to evaluate log prior
        :type value: float

        :return: Log prior evaluated at value
        :rtype: float
        """
        return jnp.log(
            jnp.log(10) * 10**value / (10 ** (self.pmax) - 10 ** (self.pmin))
        )

    def sample(self):
        """Sample from log-uniform distribution."""
        return jnp.log10(np.random.uniform(10**self.pmin, 10**self.pmax))
