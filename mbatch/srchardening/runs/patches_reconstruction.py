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

def get_map_name(fgkey, sim_id=1000, fsky=None, IQU=True, alm=False, wcs=None, ilc_type=None):
    tag = f'{fgkey}_{sim_id}'
    if alm:
        assert wcs is not None
        assert ilc_type is not None
        maptype = 'alm'
        tag = f'{tag}_{wcs}_{ilc_type}'
    else:
        maptype = 'map'
    if fsky is not None:
        tag = f'{tag}_{fsky}'
    if IQU:
        return f'{maptype}_car_{tag}_IQU.fits'
    else: 
        return f'{maptype}_car_{tag}.fits'
    
def get_map_filter(oalms, Noise, args):
    '''
    apply filter (diagonal inverse-variance) to input alms
    assumes covariance used in filter is diagonal in harmonic space
    
    args.lmin, args.lmax: int
        minimum/maximum ell for analysis
    '''
    cltt, clee, clbb = Noise['TT'], Noise['EE'], Noise['BB']

    ls = np.arange(len(cltt))

    nells_t = maps.interp(ls, cltt)
    nells_e = maps.interp(ls, clee)
    nells_b = maps.interp(ls, clbb)

    filt_t = 1./ (nells_t(ls))
    filt_e = 1./ (nells_e(ls))
    filt_b = 1./ (nells_b(ls))
    
    if oalms.shape[0] == 3:
        print('3 component map')
        almt = qe.filter_alms(oalms[0].copy(), filt_t, lmin = args.lmin, lmax = args.lmax)
        alme = qe.filter_alms(oalms[1].copy(), filt_e, lmin = args.lmin, lmax = args.lmax)
        almb = qe.filter_alms(oalms[2].copy(), filt_b, lmin = args.lmin, lmax = args.lmax)
    
    else:
        almt = qe.filter_alms(oalms.copy(), filt_t, lmin = args.lmin, lmax = args.lmax)
        alme = np.zeros_like(almt)
        almb = np.zeros_like(almt)

    return almt, alme, almb

def read_fgmap(args, fgkey, alm=False, fsky=None, wcs=None, ilc_type=None):
    if not alm:
        dust_map = enmap.read_map(maps_path(get_map_name(fgkey, 1000)))
    else:
        assert wcs is not None
        assert ilc_type is not None
        assert fsky is not None
        foreground_alms = hp.read_alm(maps_path(get_map_name(fgkey, 1000, alm=alm, fsky=fsky, wcs=wcs, ilc_type=ilc_type)), hdu=(1,2,3))
        dust_map = cs.alm2map(foreground_alms, enmap.empty((3,) + args.shape, args.wcs))

    return dust_map

parser = argparse.ArgumentParser(description="New Reconstruction Code")
parser.add_argument("--output-dir", type=str,  default=None,help="Output directory.")
parser.add_argument("--est", type=str, default='TT', help='Estimator, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
parser.add_argument("--skyfrac", type=str, default='GAL070')
parser.add_argument("--experiment", dest='exp', type=str, help='experiment name')
parser.add_argument("--fg-type", type=str, default='gauss')
parser.add_argument("--freq", type=int, default=90)
parser.add_argument("--nsims-mf", type=int, default=50)
parser.add_argument("--width_ra", type=int,  default=15,help="Width of the RA patch.")
parser.add_argument("--width_dec", type=int,  default=10,help="Width of the DEC patch.")
parser.add_argument("--profile-path", type=str, default='/home/ia404/gitreps/so-lenspipe/data/tsz_profile5000.txt', help="Path to the profile.")
parser.add_argument("--patch-start", type=int, default=None, help="Patch start.")
parser.add_argument("--patch-end", type=int, default=None, help="Patch end.")
parser.add_argument("--maps-path", type=str, help='Path to the maps.')
parser.add_argument("--ilc", default=False, action='store_true', help='Load ILC maps')
parser.add_argument("--ilc-type", type=str, default=None, help='type of ILC method')

args = parser.parse_args()
ell_array, lfac = autils.get_ell_arrays(args.lmax)

baseline_path = args.output_dir + '/../'
output_dir = baseline_path + 'stage_reconstruction/'

maps_path = lambda x: os.path.join(args.maps_path, x)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
output_path = lambda x: os.path.join(output_dir, x)

npatches = np.load(baseline_path + f'local_masks/submap_coordinates_{args.exp}_{args.skyfrac}_{args.width_ra}_{args.width_dec}.npy').shape[0]

# data
MASK_EXP = {'ACT': 'ACT',
            'SO': 'SOLAT'}
WCS_EXP = {'ACT': 'wcsACT', 'SO': 'wcsSO'}

fullmask = enmap.read_map(f'/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/masks/{MASK_EXP[args.exp]}_{args.skyfrac}_{WCS_EXP[args.exp]}_car_apo3deg.fits')
args.wcs, args.shape = fullmask.wcs, fullmask.shape

if args.ilc:
    fgkey = f'{args.fg_type}'
else:
    fgkey = f'{args.freq}_{args.fg_type}'
            
data_map = read_fgmap(args, fgkey, alm=False, fsky=None, wcs=None, ilc_type=None)[0] # !! T only

Als_noph, Als_ph, _, _ = autils.get_Als_Res(args)
Ucls, Tcls, Filters, Noise = autils.compute_for_filters(args)
cl_tot = Noise[args.est]
profile = np.loadtxt(args.profile_path)

norm_src = pytempura.get_norms(['SRC'], Ucls, Ucls, Tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)
R_prof_tt = pytempura.get_cross('SRC', 'TT', Ucls, Tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)

args.bh = False
args.ph = False
qfunc = autils.get_qfunc_forqe(args)

args.bh = True
args.ph = True
qfunc_ph = autils.get_qfunc_forqe(args)

noise_cls_TT = Noise[args.est]
ell = np.arange(len(noise_cls_TT))
noise_interp_func = maps.interp(ell, noise_cls_TT)
filter_func_T = 1. / noise_interp_func(ell)

local_masks = {}
for i in range(npatches):
    local_masks[i] = enmap.read_map(baseline_path + f'local_masks/mask_{args.exp}_{args.skyfrac}_{i}.fits')

if args.patch_start is not None:
    patches = np.arange(args.patch_start, args.patch_end+1)
    assert len(patches) > 0, "No patches to process."
    npatches = len(patches)
else:
    args.patch_start = 0

comm,rank,my_tasks = mpi.distribute(npatches)
s = stats.Stats(comm)

for jj in my_tasks:

    n = jj + args.patch_start
    print(n)

    mask_project = enmap.project(local_masks[n], args.shape, wcs=args.wcs)
    px_local = qe.pixelization(shape=mask_project.shape,wcs=mask_project.wcs)

    data_map_local = data_map * mask_project

    local_data_alms = cs.map2alm(data_map_local, lmax=args.mlmax)
    local_data_cls = cs.alm2cl(local_data_alms) / w_n(mask_project,2)
    cl_fg = smooth_cls(local_data_cls, points=300)

    cl_2pt_tcls = {args.est: cl_tot**2 / local_data_cls}
    Al_dust_N0_TT = pytempura.get_norms([args.est], Ucls, Ucls, cl_2pt_tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)

    N0_TT_grad = Als_noph[args.est][0]**2 / Al_dust_N0_TT[args.est][0]

    np.savetxt(output_path(autils.get_recons_name(n, args, tag='N0')), N0_TT_grad)

    f_foreground_alms = qe.filter_alms(local_data_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)
    f_alms = np.array([f_foreground_alms, np.zeros_like(f_foreground_alms), np.zeros_like(f_foreground_alms)])

    # Compute the 4-point function of the filtered alms
    foreground_4pt = qfunc(f_alms, f_alms)
    foreground_4pt_ph = qfunc_ph(f_alms , f_alms)
    cls_4pt_nomf = cs.alm2cl(foreground_4pt[0]) / w_n(mask_project, 4)
    cls_4pt_ph_nomf = cs.alm2cl(foreground_4pt_ph[0]) / w_n(mask_project, 4)

    norm_src_fg = pytempura.get_norms(['SRC'], Ucls, Ucls, cl_2pt_tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)
    R_prof_fg = pytempura.get_cross('SRC', 'TT', Ucls, cl_2pt_tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)
    
    i=0
    N0 = N0_TT_grad # norm_lens[i]**2/norm_fg[i]
    N0_s = norm_src['SRC']**2/norm_src_fg['SRC']
    N0_tri = (N0 + R_prof_tt**2 * Als_noph['TT'][i]**2 * N0_s
              - 2 * R_prof_tt * Als_noph['TT'][i]**2 * norm_src['SRC'] * R_prof_fg)
    N0_tri /= (1 - Als_noph['TT'][i]*norm_src['SRC']*R_prof_tt**2)**2

    np.savetxt(output_path(autils.get_recons_name(n, args, tag='N0', ph=True)), N0_tri)

    mf_data = {}
    mf_ph_data = {}

    for sett in range(2):

        rg = []
        ig = []

        rg_ph = []
        ig_ph = []

        for sim_id in range(args.nsims_mf): 
            # print(sim_id)
            np.random.seed(int(sim_id + sett * 500))

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
            ng_4pt_ph = qfunc_ph(ng_alms, ng_alms)

            rg.append(ng_4pt[0].real)
            ig.append(ng_4pt[0].imag)

            rg_ph.append(ng_4pt_ph[0].real)
            ig_ph.append(ng_4pt_ph[0].imag)

        rg_mean = np.asarray(rg).mean(axis=0)
        ig_mean = np.asarray(ig).mean(axis=0)
        rg_ph_mean = np.asarray(rg_ph).mean(axis=0)
        ig_ph_mean = np.asarray(ig_ph).mean(axis=0)

        mf_data[f'mf_grad_set{sett}'] = rg_mean + 1j * ig_mean
        mf_ph_data[f'mf_grad_set{sett}'] = rg_ph_mean + 1j * ig_ph_mean

    xy_g0 = foreground_4pt[0] - mf_data['mf_grad_set0']
    xy_g1 = foreground_4pt[0] - mf_data['mf_grad_set1']

    xy_ph_g0 = foreground_4pt_ph[0] - mf_ph_data['mf_grad_set0']
    xy_ph_g1 = foreground_4pt_ph[0] - mf_ph_data['mf_grad_set1']

    # Compute the power spectrum of the 4-point function
    cls_4pt = cs.alm2cl(xy_g0, xy_g1) / w_n(mask_project, 4)
    cls_4pt_ph = cs.alm2cl(xy_ph_g0, xy_ph_g1) / w_n(mask_project, 4)
    #cls_4pt_noise0 = cs.alm2cl(xy_g0) / w_n(mask_project, 4)

    mcg_bh = cs.alm2cl(mf_data['mf_grad_set0'], mf_data['mf_grad_set1']) / w_n(mask_project, 4)
    mcg_bh_ph = cs.alm2cl(mf_ph_data['mf_grad_set0'], mf_ph_data['mf_grad_set1']) / w_n(mask_project, 4)
    
    # Save the power spectrum to a file
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='mf_2pt', mf=True)), mcg_bh)
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='auto', mf=True)), cls_4pt)
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='auto', mf=False)), cls_4pt_nomf)

    np.savetxt(output_path(autils.get_recons_name(n, args, tag='mf_2pt', ph=True, mf=True)), mcg_bh_ph)
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='auto', ph=True, mf=True)), cls_4pt_ph)
    np.savetxt(output_path(autils.get_recons_name(n, args, tag='auto', ph=True, mf=False)), cls_4pt_ph_nomf)
