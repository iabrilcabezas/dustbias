'''
auto_comparison.py
    calculate autospectra of foregrounds, compare to kappakappa
'''

import os
import numpy as np
import argparse
from solenspipe import get_qfunc
from solenspipe.utility import w_n
from sofind import DataModel
from pixell import enmap, curvedsky as cs
from falafel import utils as futils, qe
from orphics import maps
from utils import get_dust_name, get_norm_name, get_auto_name
from utils import get_px_frommask, get_noise_dict_name

print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--dust-type", type=str)
parser.add_argument("--dust-freq", type=float, default=95)
parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--skyfrac", type=str, default='GAL070')
parser.add_argument("--apodfact", type=str, default='3dg')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')
parser.add_argument("--est", type=str,default = 'TT', help='Estimator, one of TT,TE,EE,EB,TB,MV,MVPOL.') 

args = parser.parse_args()

args.est = args.est.upper()

odust_name = get_dust_name(args)
oname = get_auto_name(args)

output_dir = f'{args.output_dir}/../stage_auto/'
os.makedirs(output_dir, exist_ok=True)
output_path = lambda x: os.path.join(output_dir, x)

mask_options = {'skyfrac': args.skyfrac, 'apodfact': args.apodfact}
dm = DataModel.from_config(args.config_name)
mask = dm.read_mask(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)
args.wcs, args.shape = mask.wcs, mask.shape
px = get_px_frommask(args)

foreground_map = enmap.read_map(output_path(f'../stage_generate_fgs/{odust_name}.fits'))

ucls = futils.get_theory_dicts(lmax=args.mlmax, grad=True)[0]

# Convert the map to alms and compute the power spectrum
foreground_alms = cs.map2alm(foreground_map, lmax=args.mlmax)
foreground_cls = cs.alm2cl(foreground_alms) / w_n(mask, 2)

# Load the noise power spectrum and filter the alms
noise_dict = np.load(output_path(f'../stage_filter/{get_noise_dict_name(args)}'), allow_pickle=True).item()
Als = np.load(output_path(f'../stage_filter/{get_norm_name(args)}'), allow_pickle=True).item()

if args.est == 'TT':

    qfunc = get_qfunc(px, ucls, args.mlmax, args.est, Al1=Als[args.est], est2=None, Al2=None, R12=None)

    noise_cls_TT = noise_dict[args.est]

    ell = np.arange(len(noise_cls_TT))
    noise_interp_func = maps.interp(ell, noise_cls_TT)
    filter_func_T = 1. / noise_interp_func(ell)

    f_foreground_alms = qe.filter_alms(foreground_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)[0]
    f_alms = np.array([f_foreground_alms, np.zeros_like(f_foreground_alms), np.zeros_like(f_foreground_alms)])

    # Compute the 4-point function of the filtered alms
    foreground_4pt = qfunc(f_alms , f_alms)
    # Compute the power spectrum of the 4-point function, [0] for grad
    cls_4pt = cs.alm2cl(foreground_4pt[0]) / w_n(mask, 4)

    # Save the power spectrum to a file
    np.savetxt(output_path(oname), cls_4pt)


