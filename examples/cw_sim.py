"""
Script for running a CW+GWB analysis with etudes. Assumes basic, non-backend-selected white noise...
at least until I fix a bug that allows for backend selection again.
"""

import pickle, glob, json, time
import logging, argparse
import numpy as np
from datetime import date

import jax
import jax.numpy as jnp

# I like using blackjax as my sampler, you can use whatever you like
import blackjax
from blackjax.util import run_inference_algorithm
from blackjax.progress_bar import progress_bar_scan

# etudes-specific imports
from etudes.parameters import Uniform
from etudes.transform import Interval
from etudes.signal_combine import Etudes1PsrSignal
from etudes.gp_signals import create_fourierdesignmatrix_red


# put together a basic model, can contain intrinsic pulsar red noise, a common process red-noise signal
# and a continuous wave signal
def make_model(psrs, has_rn=True, has_gwb=True, has_cw=True):
    params = []
    for psr in psrs:
        if has_rn:
            gamma = Uniform(0.0, 7.0, name='{}_rn_gamma'.format(psr.name))
            log10_A = Uniform(-18.0, -11.0, name='{}_rn_log10_A'.format(psr.name))

            params.extend([gamma, log10_A])
        
    if has_cw:
        cosinc = Uniform(-1.0, 1.0, name='cw_cosinc')
        costheta = Uniform(-1.0, 1.0, name='cw_costheta')
        log10_Mc = Uniform(7.0, 10.0, name='cw_log10_Mc')
        log10_fgw = Uniform(-9.0, -7.0, name='cw_log10_fgw')
        log10_h = Uniform(-18.0, -11.0, name='cw_log10_h')
        phase0 = Uniform(0, 2*np.pi, name='cw_phase0')
        phi = Uniform(0, 2*np.pi, name='cw_phi')
        psi = Uniform(0, np.pi, name='cw_psi')

        params.extend([cosinc, costheta, log10_Mc, log10_fgw, log10_h, phase0, phi, psi])
    
    if has_gwb:
        gamma_gw = Uniform(0.0, 7.0, name='gamma_gw')
        log10_Agw = Uniform(-18.0, -11.0, name='log10_A_gw')
        
        params.extend([gamma_gw, log10_Agw])
    
    # currently not used, but indexing the parameters may come in handy later
    # especially when dealing with non-conventional priors
    for i, p in enumerate(params):
        p.idx = i
    
    return params

# function to make all individual pulsar objects and store them in a list
def make_psr_objs(psrs, Fmats, Ffreqs, Ts, has_basis_ecorr=False, has_gwb=True, has_cw=True,
                  efac=True, equad=False, fix_wn=True, fix_wn_vals=None):
    psrobjs = []
    tref = np.max([p.toas.min() for p in psrs])
    for psr, Fmat, Ff, T in zip(psrs, Fmats, Ffreqs, Ts):
        psrobjs.append(Etudes1PsrSignal(psr=psr, has_basis_ecorr=has_basis_ecorr, has_gwb=has_gwb, has_cw=has_cw,
                                         has_tm=True, Fmat=Fmat, Ffreqs=Ff, T=T, efac=efac, equad=equad, fix_wn=fix_wn,
                                         fix_wn_vals=fix_wn_vals, tref=tref))
    
    return psrobjs

# main blackjax inference loop with progress bar for both NUTS adaptation and sampling
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def _one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state
    
    one_step = progress_bar_scan(num_samples)(_one_step)

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

# 
def main(resdir, nburnin=2_000, nsamples=10_000):
    """
    resdir: directory location for where you want the resulting chains to be store
    """

    # setup logging
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # here as check if we're using CPU or GPU (or whatever you intended)
    msg = jax.default_backend()
    logger.info(msg)

    # get RNG key
    rng_key = jax.random.PRNGKey(int(date.today().strftime("%Y%m%d")))

    # load data and noise dictionary (noise dict currently not used)
    datadir = ''
    picklefile = ''

    with open(picklefile, 'rb') as f:
        psrs = pickle.load(f)

    # here's where the noise dictionary is created
    # only valid when we're using simplified simulated data
    fix_wn_vals = {}
    for psr in psrs:
        fix_wn_vals['{}_efac'.format(psr.name)] = 1.0
    
    # create Fmats, Umats, and Ts if needed (all the basis matrices)
    # these all get created before pulsar creation so (I think) they can be cached
    Fmats, Ffreqs = [], []
    Ts = []

    for psr in psrs:
        T = psr.Mmat
        Fmat, Ffreq = create_fourierdesignmatrix_red(psr.toas)
        T = jnp.concatenate([Fmat, T], axis=1)
        Fmats.append(Fmat)
        Ffreqs.append(Ffreq)
        Ts.append(T)
    
    # create list of varying parameters in model
    params = make_model(psrs)

    # make list of parameter names and save as file
    param_names = [p.name for p in params]
    with open(resdir + 'params.txt', 'w') as f:
        f.write('\n'.join(param_names))
    
    # create individual pulsar objects
    psrobjs = make_psr_objs(psrs, Fmats, Ffreqs, Ts,
                            efac=True, equad=False,
                            fix_wn=True, fix_wn_vals=fix_wn_vals)
    
    # make interval object (interval being the coordinate transformation used)
    pta = Interval(psrs, psrobjs, params)
    logger.info('Created PTA object')

    # define logdensity function and starting value
    def logdensity_fn(params):
        return pta.get_lnprob(params)
    
    # Setup sampler and start adaptation warmup
    initial_position = pta.forward(pta.pavgs)
    
    # Do warmup adaptation
    start = time.time()
    
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn, progress_bar=True)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), x = warmup.run(warmup_key, initial_position, num_steps=nburnin)
    msg = 'Warmup adaptation time: {} s'.format(time.time() - start)
    logger.info(msg)
    
    # Sample
    start = time.time()

    kernel = blackjax.nuts(logdensity_fn, **parameters)
    states = run_inference_algorithm(sample_key, state, kernel, nsamples,
                                     progress_bar=True)
    msg = 'Sampling time: {} s'.format(time.time() - start)
    logger.info(msg)
    
    # transform chain back to original parameters
    chain = pta.backward(states[1].position)

    # Save chains
    with open(resdir + 'chains.txt', 'w') as f:
        np.savetxt(f, chain)
    
    return

# you can run this from the command line or bash script if you want
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('resdir', type=str, help='directory for storing chains')
    parser.add_argument('--nsamples', type=int, help='Number of NUTS samples to generate', default=10_000, required=False)
    parser.add_argument('--nburnin', type=int, help='Number of burnin adaptation samples for NUTS', default=2_000, required=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)
