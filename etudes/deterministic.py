"""
Contains functions related to deterministic signal modeling.
In a (perhaps vain) attempt to keep all functions pure and appease the JAX overlords,
the version of the CW delay function (full frequency evolution vs phase
approximation) is defined at class initialization.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import etudes.constants as const


def create_gw_antenna_pattern(pos, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param pos: Unit vector from Earth to pulsar
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians

    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = jnp.array([jnp.sin(gwphi), -jnp.cos(gwphi), 0.0])
    n = jnp.array([-jnp.cos(gwtheta) * jnp.cos(gwphi), -jnp.cos(gwtheta) * jnp.sin(gwphi), jnp.sin(gwtheta)])
    omhat = jnp.array([-jnp.sin(gwtheta) * jnp.cos(gwphi), -jnp.sin(gwtheta) * jnp.sin(gwphi), -jnp.cos(gwtheta)])

    fplus = 0.5 * (jnp.dot(m, pos) ** 2 - jnp.dot(n, pos) ** 2) / (1 + jnp.dot(omhat, pos))
    fcross = (jnp.dot(m, pos) * jnp.dot(n, pos)) / (1 + jnp.dot(omhat, pos))
    cosMu = -jnp.dot(omhat, pos)

    return fplus, fcross, cosMu


@register_pytree_node_class
class CW_Signal(object):
    """
    Class for single-pulsar CW signals. Registered as a
    custom PyTree to allow jit-compilation of methods
    without continuously defining `self` as a static
    argument.
    """

    def __init__(self, psr, tref=0, evolve=True):
        self.psr = psr
        self.tref = tref
        self.evolve = evolve

        self._init_delay(evolve=self.evolve)
    
    def _init_delay(self, evolve=True):
        if evolve:
            self._freqevol_fn = self._full_evolve
        else:
            self._freqevol_fn = self._phase_approx
    
    def _full_evolve(self, w0, mc, toas, phase0):
        omega = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * toas)**(-3/8)
        phase = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))
        return omega, phase
    
    def _phase_approx(self, w0, mc, toas, phase0):
        omega = w0
        phase = phase0 + omega * toas
        return omega, phase

    def get_delay(self, pars, **kwargs):
        return self._get_delay(**pars, **kwargs)
    
    @jax.jit
    def _get_delay(self,
                cw_costheta=0, cw_phi=0, cw_cosinc=0,
                cw_log10_Mc=9, cw_log10_fgw=-8, cw_log10_h=-15,
                cw_phase0=0, cw_psi=0, **kwargs):
        """
        General CW signal delay function
        """

        # convert units to time
        mc = 10**cw_log10_Mc * const.Tsun
        fgw = 10**cw_log10_fgw
        gwtheta = jnp.arccos(cw_costheta)
        inc = jnp.arccos(cw_cosinc)

        dist = 2 * mc**(5/3) * (jnp.pi*fgw)**(2/3) / 10**cw_log10_h

        # get antenna pattern funcs and cosMu
        # write function to get pos from theta,phi
        fplus, fcross, _ = create_gw_antenna_pattern(self.psr._pos, gwtheta, cw_phi)

        # get pulsar time
        toas = self.psr.toas - self.tref

        # orbital frequency
        w0 = jnp.pi * fgw
        phase0 = cw_phase0 / 2  # convert GW to orbital phase

        # evolution
        omega, phase = self._freqevol_fn(w0, mc, toas, phase0)

        # define time dependent coefficients
        At = -0.5*jnp.sin(2*phase)*(3+jnp.cos(2*inc))
        Bt = 2*jnp.cos(2*phase)*jnp.cos(inc)

        # now define time dependent amplitudes
        alpha = mc**(5./3.)/(dist*omega**(1./3.))

        # define rplus and rcross
        rplus = alpha*(-At*jnp.cos(2*cw_psi)+Bt*jnp.sin(2*cw_psi))
        rcross = alpha*(At*jnp.sin(2*cw_psi)+Bt*jnp.cos(2*cw_psi))

        # residuals
        res = -fplus*rplus - fcross*rcross

        return res

    # Necessary flatten and unflatten methods to register class
    # as a PyTree
    def tree_flatten(self):
        return (), (self.psr, self.tref, self.evolve)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

