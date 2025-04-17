import numpy as np
import os
from pixell import enmap, curvedsky as cs
from solenspipe.utility import w_n, smooth_cls
from solenspipe import get_qfunc
from falafel import utils as futils, qe
import pytempura
import argparse
from orphics import maps, stats, mpi, io
import utils as autils

parser = argparse.ArgumentParser(description="New Reconstruction Code")
parser.add_argument("--output-dir", type=str,  default=None,help="Output directory.")
parser.add_argument("--est", type=str, default='TT', help='Estimator, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
parser.add_argument("--skyfrac", type=str, default='GAL070')
parser.add_argument("--dust-type", type=str, default='gauss')
parser.add_argument("--dust-freq", type=float, default=90.)
parser.add_argument("--nsims-mf", type=int, default=50)
parser.add_argument("--width_ra", type=int,  default=15,help="Width of the RA patch.")
parser.add_argument("--width_dec", type=int,  default=10,help="Width of the DEC patch.")

args = parser.parse_args()
ell_array, lfac = autils.get_ell_arrays(args.lmax)

baseline_path = args.output_dir + '/../'
output_dir = baseline_path + 'stage_reconstruction/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

output_path = lambda x: os.path.join(output_dir, x)

npatches = np.load(baseline_path + f'local_masks/submap_coordinates_{args.skyfrac}_{args.width_ra}_{args.width_dec}.npy').shape[0]

# data
data_map = enmap.read_map(baseline_path + f'stage_generate_dustng_maps/{autils.get_scaled_map_name(args.dust_type, sim_id=1000, fsky=args.skyfrac, freq=args.dust_freq)}')
args.wcs, args.shape = data_map.wcs, data_map.shape

filters = futils.get_theory_dicts_white_noise(args.filter_whitefwhm, args.filter_whiteamplitude, lmax=args.mlmax, grad=False)[1]
ucls, _ = futils.get_theory_dicts(nells=None, lmax=args.mlmax, grad=True)

Als = np.load(baseline_path + f'stage_compute_filters/{autils.get_norm_name(args)}', allow_pickle=True).item()
noise_dict = np.load(baseline_path + f'stage_compute_filters/{autils.get_noise_dict_name(args)}', allow_pickle=True).item()
cl_tot = filters[args.est]

noise_cls_TT = noise_dict[args.est]

ell = np.arange(len(noise_cls_TT))
noise_interp_func = maps.interp(ell, noise_cls_TT)
filter_func_T = 1. / noise_interp_func(ell)

local_masks = {}
for i in range(npatches):
    local_masks[i] = enmap.read_map(baseline_path + f'local_masks/mask_{args.skyfrac}_{i}.fits')

comm,rank,my_tasks = mpi.distribute(npatches)
s = stats.Stats(comm)

for n in my_tasks:

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

    np.savetxt(output_path(autils.get_recons_name(n, args, tag='N0')), N0_TT_grad)

    qfunc = get_qfunc(px_local, ucls, args.mlmax, args.est, Al1=Als[args.est], est2=None, Al2=None, R12=None)

    f_foreground_alms = qe.filter_alms(local_data_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)

    f_alms = np.array([f_foreground_alms, np.zeros_like(f_foreground_alms), np.zeros_like(f_foreground_alms)])

    # Compute the 4-point function of the filtered alms
    foreground_4pt = qfunc(f_alms , f_alms)

    cls_4pt_nomf = cs.alm2cl(foreground_4pt[0]) / w_n(mask_project, 4)

    mf_data = {}

    for set in range(2):

        rg = []
        ig = []

        for sim_id in range(args.nsims_mf): 
            # print(sim_id)
            np.random.seed(int(sim_id + set * 500))

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

            rg.append(ng_4pt[0].real)
            ig.append(ng_4pt[0].imag)

        rg_mean = np.asarray(rg).mean(axis=0)
        ig_mean = np.asarray(ig).mean(axis=0)

        mf_data[f'mf_grad_set{set}'] = rg_mean + 1j * ig_mean

    xy_g0 = foreground_4pt[0] - mf_data['mf_grad_set0']
    xy_g1 = foreground_4pt[0] - mf_data['mf_grad_set1']

    # Compute the power spectrum of the 4-point function
    cls_4pt = cs.alm2cl(xy_g0, xy_g1) / w_n(mask_project, 4)
    #cls_4pt_noise0 = cs.alm2cl(xy_g0) / w_n(mask_project, 4)

    mcg_bh = cs.alm2cl(mf_data['mf_grad_set0'], mf_data['mf_grad_set1']) / w_n(mask_project, 4)
    
    # Save the power spectrum to a file
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='mf_2pt', mf=True)), mcg_bh)
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='auto', mf=True)), cls_4pt)
    # np.savetxt(output_path(get_auto_name(n, mf=True, tag='set00')), cls_4pt_noise0)
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='auto', mf=False)), cls_4pt_nomf)
