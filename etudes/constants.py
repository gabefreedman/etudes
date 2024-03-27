"""
Useful constants. They're copied from constants.py in `enterprise`.
"""

import scipy.constants as sc

c = sc.speed_of_light
pc = sc.parsec
GMsun = 1.327124400e20
yr = sc.Julian_year
fyr = 1.0 / yr

kpc = pc * 1.0e3
Mpc = pc * 1.0e6

Tsun = GMsun / (c**3)