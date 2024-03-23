"""
This module contains classes for coordinate transformations on top
of the base EtudesPTA object. For now the one primarily used is Interval,
but there are plans to include a whitening transformation class in the
future.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from etudes.etudes.signal_combine import _EtudesPTA

@register_pytree_node_class
class EtudesPTA(_EtudesPTA):
    """
    A wrapper class for the underlying _EtudesPTA class from the
    signal_combine.py file. I'm not sure if this is entirely necessary,
    but it helps to keep most of the messy argument handling in a separate,
    hidden class, and only focus on the important information here.
    """
    def __init__(self, psrs, signalcollections, params,
                 pmins=None, pmaxs=None, pavgs=None, **kwargs):
        super(EtudesPTA, self).__init__(psrs, signalcollections, **kwargs)
        
        self.params = params
        self.signalcollections = signalcollections
        self.param_names = [par.name for par in params]
        
        self.pmins = jnp.array([par.pmin for par in params])
        self.pmaxs = jnp.array([par.pmax for par in params])
        self.pavgs = (self.pmaxs + self.pmins) / 2
    
    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psrs, self.signalcollections, self.params,
                    self.pmins, self.pmaxs, self.pavgs)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

@register_pytree_node_class
class Interval(EtudesPTA):
    """This class implements an interval coordinate transformation for model
    hyperparameters. It transforms the prior bounds for all parameters from
    some min and max value to positive and negative infinity.
    """
    def __init__(self, psrs, signalcollections,
                 params, **kwargs):
        super(Interval, self).__init__(psrs, signalcollections, params, **kwargs)

        self.a = self.pmins
        self.b = self.pmaxs
    
    @jax.jit
    def forward(self, x):
        """
        Apply interval transformation to parameter vector.
        """
        return jnp.log((x - self.a) / (self.b - x))
    
    @jax.jit
    def backward(self, p):
        """
        Undo the interval transformation.
        """
        return (self.b - self.a) * jnp.exp(p) / (1 + jnp.exp(p)) + self.a
    
    @jax.jit
    def dxdp(self, p):
        """
        Jacobian of interval transforamtion
        """
        pp = jnp.atleast_2d(p)
        d = jnp.ones_like(pp)
        d = d.at[:].set((self.b-self.a)*jnp.exp(pp[:])/(1+jnp.exp(pp[:]))**2)
        return d.reshape(p.shape)
    
    @jax.jit
    def logjacobian(self, p):
        """
        Log of Jacobian evaluated at input parameter vector.
        """
        lj = jnp.sum(jnp.log(self.b-self.a) + p -
                    2*jnp.log(1.0+jnp.exp(p)))
        
        return lj
    
    @jax.jit
    def logjacobian_grad(self, p):
        """
        Return both the log of the Jacobian and its gradient, both
        evaluated at the input parameter vector
        """
        lj = jnp.sum(jnp.log(self.b-self.a) + p -
                    2*jnp.log(1.0+jnp.exp(p)))

        lj_grad = jnp.zeros_like(p)
        lj_grad = (1 - jnp.exp(p)) / (1 + jnp.exp(p))
        return lj, lj_grad

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psrs, self.signalcollections, self.params)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)