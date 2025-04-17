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
from orphics import maps

parser = argparse.ArgumentParser(description="New Reconstruction Code")
parser.add_argument("--mask", type=str,  default="/rds/project/dirac_vol5/rds-dirac-dp002/AdvACT/DR6_products/masks/act_mask_fejer1_20220316_GAL060_rms_70.00_downgrade_3dg.fits",help="mask path")
parser.add_argument("--output-dir", type=str,  default=None,help="Output directory.")
parser.add_argument("--width_ra", type=int,  default=15,help="Width of the RA patch.")
parser.add_argument("--width_dec", type=int,  default=10,help="Width of the DEC patch.")

parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
# parser.add_argument("--nside", type=int, default=2048, help="nside healpix maps")
parser.add_argument("--config-name", type=str, default='act_dr6v4')
# parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='3dg')
parser.add_argument("--skyfrac", type=str, default='GAL070')
parser.add_argument("--meanfield", action='store_true', help='subtract meanfield')
parser.add_argument("--dust-type", type=str, default='gauss')
parser.add_argument("--patches-fsky", type=str)
parser.add_argument("--data-path", type=str, default='mbatch_ng_240824')
parser.add_argument("--nsims-mf", type=int, default=50)
args = parser.parse_args()
path = lambda x: os.path.join(args.output_dir, x)

if not args.data_path.endswith('/'):
    args.data_path += '/'

ell_array, lfac = utils.get_ell_arrays(args.lmax)

data_path = lambda x: os.path.join(args.output_dir + f'{args.data_path}/', x)

mask_path = path('../stage_local_masks_{args.patches_fsky}/')

npatches = {'0.01': 33, '0.001': 117}
npatches = npatches[args.patches_fsky]

# data
data_map = enmap.read_map(data_path(f'stage_data/{utils.get_scaled_map_name(args.dust_type, sim_id=1000, fsky=args.skyfrac)}'))
args.wcs, args.shape = data_map.wcs, data_map.shape

filters = futils.get_theory_dicts_white_noise(args.filter_whitefwhm, args.filter_whiteamplitude, lmax=args.mlmax, grad=False)[1]
ucls, _ = futils.get_theory_dicts(nells=None, lmax=args.mlmax, grad=True)

Als = np.load(data_path(f'stage_filter/{utils.get_norm_name(args)}'), allow_pickle=True).item()
noise_dict = np.load(data_path(f'stage_filter/{utils.get_noise_dict_name(args)}'), allow_pickle=True).item()
cl_tot = filters[args.est]

local_masks = {}
for i in npatches:
    local_masks[i] = enmap.read_map(mask_path(f'mask_{i}.fits'))

for n in range(npatches):

    mask_project = enmap.project(local_masks[n], args.shape, wcs=args.wcs)
    px_local = qe.pixelization(shape=mask_project.shape,wcs=mask_project.wcs)

    data_map_local = data_map * mask_project

    local_data_alms = cs.map2alm(data_map_local, lmax=args.mlmax)
    local_data_cls = cs.alm2cl(local_data_alms) / w_n(mask_project,2)
    cl_fg = smooth_cls(local_data_cls, points=300)

    cl_2pt_tcls = {args.est: cl_tot**2 / local_data_cls}

    Al_dust_N0_TT = pytempura.get_norms([args.est], ucls, ucls, cl_2pt_tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)
    N0_TT_grad = Als[args.est][0]**2 / Al_dust_N0_TT[args.est][0]

    qfunc = get_qfunc(px_local, ucls, args.mlmax, args.est, Al1=Als[args.est], est2=None, Al2=None, R12=None)

    noise_cls_TT = noise_dict[args.est]

    ell = np.arange(len(noise_cls_TT))
    noise_interp_func = maps.interp(ell, noise_cls_TT)
    filter_func_T = 1. / noise_interp_func(ell)

    f_foreground_alms = qe.filter_alms(local_data_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)

    f_alms = np.array([f_foreground_alms, np.zeros_like(f_foreground_alms), np.zeros_like(f_foreground_alms)])

    # Compute the 4-point function of the filtered alms
    foreground_4pt = qfunc(f_alms , f_alms)

    cls_4pt_nomf = cs.alm2cl(foreground_4pt[0]) / w_n(mask_project, 4)

    rg = []
    ig = []

    for i in range(args.nsims_mf):
        print(i)
        np.random.seed(i)
        # dust_gauss_map = hp.sphtfunc.synfast(dust_cl_generate, nside=args.nside) # randomly generated gaussian dust map
        # gauss_map_car = reproject.healpix2map(dust_gauss_map, masks['GAL070'].shape, masks['GAL070'].wcs, lmax = args.mlmax, rot="gal,cel")
        dust_random_map_alm = cs.rand_alm(cl_fg, lmax=args.mlmax)
        dust_random_map = cs.alm2map(dust_random_map_alm, enmap.empty((1,) + mask_project.shape, mask_project.wcs))[0]
        dust_map_car_mask = dust_random_map * mask_project
        
        # transform to alm
        ng_alms = cs.map2alm(dust_map_car_mask, lmax=args.mlmax)
        # filter
        ng_foreground_alms = qe.filter_alms(ng_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)
        ng_alms = np.array([ng_foreground_alms, np.zeros_like(ng_foreground_alms), np.zeros_like(ng_foreground_alms)])

        # compute 4pt
        ng_4pt = qfunc(ng_alms, ng_alms)
        ng_g = ng_4pt[0]
        # add to stack
        rg.append(ng_g.real)
        ig.append(ng_g.imag)

def compute_mean_field(cl_2pt, random_seed, seed_set):
    np.random.seed(int(random_seed + seed_set))
    # dust_gauss_map = hp.sphtfunc.synfast(dust_cl_generate, nside=args.nside) # randomly generated gaussian dust map
    # gauss_map_car = reproject.healpix2map(dust_gauss_map, masks['GAL070'].shape, masks['GAL070'].wcs, lmax = args.mlmax, rot="gal,cel")
    dust_random_map_alm = cs.rand_alm(cl_2pt, lmax=args.mlmax)
    dust_random_map = cs.alm2map(dust_random_map_alm, enmap.empty((1,) + mask_project.shape, mask_project.wcs))[0]
    dust_map_car_mask = dust_random_map * mask_project
    
    # transform to alm
    ng_alms = cs.map2alm(dust_map_car_mask, lmax=args.mlmax)
    # filter
    ng_foreground_alms = qe.filter_alms(ng_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)
    ng_alms = np.array([ng_foreground_alms, np.zeros_like(ng_foreground_alms), np.zeros_like(ng_foreground_alms)])

    # compute 4pt
    ng_4pt = qfunc(ng_alms, ng_alms)
    ng_g = ng_4pt[0]
    # add to stack
    rg.append(ng_g.real)
    ig.append(ng_g.imag)