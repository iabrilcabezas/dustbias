
'''
stage_measure_gauss_sims_measure_mf.py "stage_mf"

Gaussian dust sims to compute meanfield
'''
import os
import argparse
import numpy as np
import healpy as hp
from solenspipe import get_qfunc
from sofind import DataModel
from pixell import enmap, curvedsky as cs
from orphics import maps, stats, mpi, io
import falafel.utils as futils
from falafel import qe
import utils as autils
# MPI ERROR HANDLER:
import mpi4py.rc
mpi4py.rc.threads = False
from auxiliary import MPI_error_handler

print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--est", type=str, default='TT', help='Estimator, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("--tilt", type=float, default=-0.80)
parser.add_argument("--amplitude", type=float, default=119.47982655)
parser.add_argument("--filter-whiteamplitude", type=float, default=12., help='White noise level [muK-arcmin]')
parser.add_argument("--filter-whitefwhm", type=float, default=1.4, help='White noise beam fwhm [arcmin]')
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--lmax", type=int, default=3000, help="Maximum multipole for lensing.")
parser.add_argument("--lmin", type=int, default=600, help="Minimum multipole for lensing.")
parser.add_argument("--nside", type=int, default=2048, help="nside healpix maps")
parser.add_argument("--sims-start", type=int, default=0, help="start sim generation from this sim id")
parser.add_argument("--sims-end", type=int, default=99, help="until sims-end sim id")
parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='3dg')
parser.add_argument("--skyfrac", type=str, default='GAL070')
parser.add_argument("--dust-type", type=str, default='gauss')
args = parser.parse_args()

sims_ids = np.arange(args.sims_start, args.sims_end + 1)
args.nsims = len(sims_ids)
assert args.nsims != 0, 'No sims to process, check sims-start and sims-end'

output_path = lambda x: os.path.join(args.output_dir, x)

noise_dict = np.load(output_path(f'../stage_filter/{autils.get_noise_dict_name(args)}'), allow_pickle=True).item()
Als = np.load(output_path(f'../stage_filter/{autils.get_norm_name(args)}'), allow_pickle=True).item()

ucls = futils.get_theory_dicts(lmax=args.mlmax, grad=True)[0]

mask_options = {'skyfrac': args.skyfrac, 'apodfact': args.apodfact}
dm = DataModel.from_config(args.config_name)
mask = dm.read_mask(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)
args.wcs, args.shape = mask.wcs, mask.shape
px = autils.get_px_frommask(args)

qfunc = get_qfunc(px, ucls, args.mlmax, args.est, Al1=Als[args.est], est2=None, Al2=None, R12=None)

noise_cls_TT = noise_dict[args.est]

ell = np.arange(len(noise_cls_TT))
noise_interp_func = maps.interp(ell, noise_cls_TT)
filter_func_T = 1. / noise_interp_func(ell)

comm,rank,my_tasks = mpi.distribute(args.nsims)
s = stats.Stats(comm)

for sim_id in my_tasks:
     
    # foreground_map = enmap.read_map(output_path(f'../stage_dust_sims/{get_gauss_dust_map_name(args, sim_id, fsky=args.skyfrac)}'))
    foreground_map = enmap.read_map(output_path(f'../stage_mf_{args.dust_type}_sims/{autils.get_dust_map_name(args, sim_id, fsky=args.skyfrac)}'))
    foreground_alms = cs.map2alm(foreground_map, lmax=args.mlmax)

    f_foreground_alms = qe.filter_alms(foreground_alms, filter_func_T, lmin=args.lmin, lmax=args.lmax)
    f_alms = np.array([f_foreground_alms, np.zeros_like(f_foreground_alms), np.zeros_like(f_foreground_alms)])

    # Compute the 4-point function of the filtered alms
    foreground_4pt = qfunc(f_alms, f_alms)

    foreground_g = foreground_4pt[0]
    foreground_c = foreground_4pt[1]

    s.add_to_stack(f'rf', foreground_g.real)
    s.add_to_stack(f'if', foreground_g.imag)
    s.add_to_stack(f'rfc', foreground_c.real)
    s.add_to_stack(f'ifc', foreground_c.imag)

with io.nostdout():
    s.get_stacks()

if rank==0:

    mf_alm = s.stacks[f'rf'] + 1j * s.stacks[f'if']
    mf_alm_curl = s.stacks[f'rfc'] + 1j * s.stacks[f'ifc']

    mf_name = autils.get_mf_name(args, pl_tag=False)
    hp.write_alm(output_path(mf_name[0]), mf_alm, overwrite = True)
    hp.write_alm(output_path(mf_name[1]), mf_alm_curl, overwrite = True)