import numpy as np
import os
from pixell import enmap, curvedsky as cs
from solenspipe.utility import w_n, smooth_cls
from solenspipe import get_qfunc
from falafel import utils as futils, qe
from math import ceil
import pytempura
import argparse
from dustbias.mbatch import utils
from orphics import maps, stats, mpi, io

parser = argparse.ArgumentParser(description="New Reconstruction Code")
# parser.add_argument("--mask", type=str,  default="/rds/project/dirac_vol5/rds-dirac-dp002/AdvACT/DR6_products/masks/act_mask_fejer1_20220316_GAL060_rms_70.00_downgrade_3dg.fits",help="mask path")
parser.add_argument("--output-dir", type=str,  default=None,help="Output directory.")
parser.add_argument("--est", type=str, default='TT', help='Estimator, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
# parser.add_argument("--config-name", type=str, default='act_dr6v4')
# parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
# parser.add_argument("--apodfact", type=str, default='3dg')
parser.add_argument("--skyfrac", type=str, default='GAL070')
# parser.add_argument("--meanfield", action='store_true', help='subtract meanfield')
parser.add_argument("--dust-type", type=str, default='gauss')
parser.add_argument("--patches-fsky", type=str)
parser.add_argument("--data-path", type=str, default='mbatch_ng_240824')
parser.add_argument("--nsims-mf", type=int, default=50)
# parser.add_argument("--sims-start", type=int, default=0)
# parser.add_argument("--sims-end", type=int, default=10)

def get_auto_name(patch, mf=True, tag=None):
    if mf:
        return f'auto_patch{patch}_mf{tag}.txt'
    else:
        return f'auto_patch{patch}_nomf{tag}.txt'

args = parser.parse_args()
ell_array, lfac = utils.get_ell_arrays(args.lmax)

path = lambda x: os.path.join(args.output_dir, x)

# if not args.data_path.endswith('/'):
#     args.data_path += '/'

# data_path = lambda x: os.path.join(args.output_dir + f'{args.data_path}/', x)

data_path = lambda x: os.path.join(args.data_path, x)
mask_path = lambda x: os.path.join(args.output_dir + f'/../stage_local_masks_{args.patches_fsky}/', x)

npatches = {'0.01': 33, '0.001': 117}
npatches = npatches[args.patches_fsky]

# data
data_map = enmap.read_map(data_path(f'{utils.get_scaled_map_name(args.dust_type, sim_id=1000, fsky=args.skyfrac)}'))
args.wcs, args.shape = data_map.wcs, data_map.shape

filters = futils.get_theory_dicts_white_noise(args.filter_whitefwhm, args.filter_whiteamplitude, lmax=args.mlmax, grad=False)[1]
ucls, _ = futils.get_theory_dicts(nells=None, lmax=args.mlmax, grad=True)

Als = np.load(data_path(f'../stage_filter/{utils.get_norm_name(args)}'), allow_pickle=True).item()
noise_dict = np.load(data_path(f'../stage_filter/{utils.get_noise_dict_name(args)}'), allow_pickle=True).item()
cl_tot = filters[args.est]

noise_cls_TT = noise_dict[args.est]

ell = np.arange(len(noise_cls_TT))
noise_interp_func = maps.interp(ell, noise_cls_TT)
filter_func_T = 1. / noise_interp_func(ell)

local_masks = {}
for i in range(npatches):
    local_masks[i] = enmap.read_map(mask_path(f'mask_{i}.fits'))

# sims_ids = np.arange(args.sims_start, args.sims_end + 1)
# nsims = len(sims_ids)
# assert nsims != 0, 'No sims to process, check sims_start and sims_end'

comm,rank,my_tasks = mpi.distribute(npatches)
s = stats.Stats(comm)

for task in my_tasks:

    n = task # args.sims_start + task

    print(n)

    mask_project = enmap.project(local_masks[n], args.shape, wcs=args.wcs)
    px_local = qe.pixelization(shape=mask_project.shape,wcs=mask_project.wcs)

    data_map_local = data_map * mask_project

    local_data_alms = cs.map2alm(data_map_local, lmax=args.mlmax)
    local_data_cls = cs.alm2cl(local_data_alms) / w_n(mask_project,2)
    cl_fg = smooth_cls(local_data_cls, points=300)

    cl_2pt_tcls = {args.est: cl_tot**2 / local_data_cls}

    Al_dust_N0_TT = pytempura.get_norms([args.est], ucls, ucls, cl_2pt_tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)
    N0_TT_grad = Als[args.est][0]**2 / Al_dust_N0_TT[args.est][0]
    
    np.savetxt(path(get_auto_name(n, mf=False, tag='N0')), N0_TT_grad)