'''
stage_compute_filters.py
    filter stage
'''

import argparse
import numpy as np
import os
import pytempura
from falafel import utils as futils
from utils import EST_NORM_LIST, get_noise_dict_name, get_norm_name

print("Normalization")
parser = argparse.ArgumentParser(description='Norm stage')
parser.add_argument("--output-dir", type=str, help='Output directory to save stage norm')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')

args = parser.parse_args()

mlmax = args.mlmax
lmin = args.lmin
lmax = args.lmax

output_path = lambda x: os.path.join(args.output_dir, x)

# ucl = response cls, tcl = total cls # at this point doesnt have noise (nells = None)
# tcls = lensed, ucls = TgradT estimator normalization (grad = True)
ucls, tcls = futils.get_theory_dicts(nells=None, lmax=mlmax, grad=True)

filters = futils.get_theory_dicts_white_noise(args.filter_whitefwhm, args.filter_whiteamplitude, lmax=args.mlmax, grad=False)[1]
# tcls gets overwritten because filter stage has the correct cell (lensed) + noise
tcls['TT'] = filters['TT']
tcls['TE'] = filters['TE']
tcls['EE'] = filters['EE']
tcls['BB'] = filters['BB']

noise = {'TT': filters['TT'][:mlmax+1], 'EE': filters['EE'][:mlmax+1], 'BB': filters['BB'][:mlmax+1]} # Ignoring TE already

Als = pytempura.get_norms(EST_NORM_LIST, ucls, ucls, tcls, lmin, lmax, k_ellmax=mlmax, profile=None)

np.save(output_path(get_norm_name(args)), Als)
np.save(output_path(get_noise_dict_name(args)), noise)