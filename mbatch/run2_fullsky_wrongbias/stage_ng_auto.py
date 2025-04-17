
import os
import argparse
import numpy as np
from orphics import maps
from solenspipe import get_qfunc
from solenspipe.utility import w_n, smooth_cls
from sofind import DataModel
from pixell import enmap, curvedsky as cs
import falafel.utils as futils
from falafel import qe
import utils as autils
from utils import DUST_TYPES
import pytempura
# load data sim

# MPI ERROR HANDLER:
import mpi4py.rc
mpi4py.rc.threads = False
from auxiliary import MPI_error_handler

print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--est", type=str, default='TT', help='Estimator, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
parser.add_argument("--nside", type=int, default=2048, help="nside healpix maps")
parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='3dg')
parser.add_argument("--skyfrac", type=str, default='GAL070')
parser.add_argument("--meanfield", action='store_true', help='subtract meanfield')
parser.add_argument("--dust-type", type=str, default='gauss')

args = parser.parse_args()

output_path = lambda x: os.path.join(args.output_dir, x)

mask_options = {'skyfrac': args.skyfrac, 'apodfact': args.apodfact}
dm = DataModel.from_config(args.config_name)
mask = dm.read_mask(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)
args.wcs, args.shape = mask.wcs, mask.shape
px = autils.get_px_frommask(args)

if args.meanfield:
    mf_data = autils.read_meanfield(args)

ucls = futils.get_theory_dicts(lmax=args.mlmax, grad=True)[0]
# Load the noise power spectrum and filter the alms
noise_dict = np.load(output_path(f'../stage_filter/{autils.get_noise_dict_name(args)}'), allow_pickle=True).item()
Als = np.load(output_path(f'../stage_filter/{autils.get_norm_name(args)}'), allow_pickle=True).item()
# fit_scaled_params = np.load(output_path(f'../stage_data/{autils.get_name_fitparams()}'), allow_pickle=True).item()

cl_tot = noise_dict[args.est]

qfunc = get_qfunc(px, ucls, args.mlmax, args.est, Al1=Als[args.est], est2=None, Al2=None, R12=None)

noise_cls_TT = noise_dict[args.est]

ell = np.arange(len(noise_cls_TT))
noise_interp_func = maps.interp(ell, noise_cls_TT)
filter_func_T = 1. / noise_interp_func(ell)

# for dust_type in DUST_TYPES:

#     args.dust_type = dust_type
    # args.tilt, args.amplitude = fit_params['tilt'], fit_params['amplitude']

foreground_map = enmap.read_map(output_path(f'../stage_data/{autils.get_scaled_map_name(args.dust_type, sim_id=1000, fsky=args.skyfrac)}'))
foreground_alms = cs.map2alm(foreground_map, lmax=args.mlmax)

f_foreground_alms = qe.filter_alms(foreground_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)

f_alms = np.array([f_foreground_alms, np.zeros_like(f_foreground_alms), np.zeros_like(f_foreground_alms)])

# Compute the 4-point function of the filtered alms
foreground_4pt = qfunc(f_alms , f_alms)

if args.meanfield:

    xy_g0 = foreground_4pt[0] - mf_data['mf_grad_set0']
    xy_g1 = foreground_4pt[0] - mf_data['mf_grad_set1']

    # xy_c0 = foreground_4pt[1] - mf_data['mf_curl_set0']
    # xy_c1 = foreground_4pt[1] - mf_data['mf_curl_set0']

    # Compute the power spectrum of the 4-point function
    cls_4pt = cs.alm2cl(xy_g0, xy_g1) / w_n(mask, 4)
    cls_4pt_noise0 = cs.alm2cl(xy_g0) / w_n(mask, 4)
    
    # Save the power spectrum to a file
    np.savetxt(output_path(autils.get_auto_name(args, pl_tag=False, mf=True)), cls_4pt)
    np.savetxt(output_path(autils.get_auto_name(args, pl_tag=False, mf=True, tag='set00')), cls_4pt_noise0)

cls_4pt_nomf = cs.alm2cl(foreground_4pt[0]) / w_n(mask, 4)
np.savetxt(output_path(autils.get_auto_name(args, pl_tag=False, mf=False)), cls_4pt_nomf)

cl_fg = np.loadtxt(output_path(f'../stage_data/{autils.get_2pt_scaled_map_name(args.dust_type, sim_id=1000, fsky=args.skyfrac)}'))
cl_fg = smooth_cls(cl_fg, points=300)
cl_2pt_tcls = {args.est: cl_tot**2 / cl_fg}

Al_dust_N0_TT = pytempura.get_norms([args.est], ucls, ucls, cl_2pt_tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)
N0_TT_grad = Als[args.est][0]**2 / Al_dust_N0_TT[args.est][0]

np.savetxt(output_path(autils.get_N0_TT_name(args, pl_tag=False, tag='smoothed')), N0_TT_grad)