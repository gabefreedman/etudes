"""
Module to contain all parameter classes. These classes are
meant to be very lightweight containers of parameter
information such as names and prior bounds. No need to make it
more complicated than that.

TODO: Replace random numbers from numpy with JAX random number module
"""

import numpy as np

import jax
import jax.numpy as jnp

class Parameter(object):
    """
    Base parameter class
    """
    def __init__(self, pmin, pmax, name=None, idx=None):
        self.pmin = pmin
        self.pmax = pmax

        self.name = name

        # Optional index in full parameter index, may
        # be helpful if I try to include support for slicing
        self.idx = idx

    def get_prior(self):
        return 0
    
    def get_logprior(self):
        return 0
    
    def sample(self):
        return 0

class Uniform(Parameter):
    """
    Uniform distribution class
    """
    def __init__(self, pmin, pmax, name=None, idx=None):
        super(Uniform, self).__init__(pmin, pmax, name, idx)

    def get_prior(self, value):
        return ((value >= self.pmin) & (value <= self.pmax)) / (self.pmax - self.pmin)
    
    def get_lnprior(self, value):
        return 0
    
    def sample(self):
        return np.random.uniform(self.pmin, self.pmax)
    
class LinearExp(Parameter):
    """
    Log-uniform distribution class
    """
    def __init__(self, pmin, pmax, name=None, idx=None):
        super(LinearExp, self).__init__(pmin, pmax, name, idx)

    def get_prior(self, value):
        return ((self.pmin <= value) & (value <= self.pmax)) * np.log(10) * 10**value / (10**self.pmax - 10**self.pmin)
    
    def get_lnprior(self, value):
        return jnp.log(jnp.log(10) * 10**value / (10**(self.pmax) - 10**(self.pmin)))
    
    def sample(self):
        return jnp.log10(np.random.uniform(10**self.pmin, 10**self.pmax))
