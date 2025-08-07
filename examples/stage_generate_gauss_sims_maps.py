'''
stage_generate_gauss_sims.py "stage_dust_maps"

Gaussian dust sims to compute meanfield
'''
import os
import argparse
import numpy as np
import healpy as hp
from sofind import DataModel
from pixell import reproject, enmap, curvedsky as cs
from solenspipe.utility import w_n
from utils import FSKYS, get_dust_2pt_name, get_gauss_dust_map_name

print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--tilt", type=float, default=-0.80)
parser.add_argument("--amplitude", type=float, default=119.47982655)
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--nside", type=int, default=2048, help="nside healpix maps")
parser.add_argument("--sims-start", type=int, default=0, help="start sim generation from this sim id")
parser.add_argument("--sims-end", type=int, default=99, help="until sims-end sim id")
parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='3dg')

args = parser.parse_args()

output_path = lambda x: os.path.join(args.output_dir, x)

sims_ids = np.arange(args.sims_start, args.sims_end + 1)
args.nsims = len(sims_ids)
assert args.nsims != 0, 'No sims to process, check sims-start and sims-end'

mask_options = {'apodfact': args.apodfact}

dm = DataModel.from_config(args.config_name)

masks = {}
w2_masks = {}
for fsky in FSKYS:
    mask_options['skyfrac'] = fsky
    masks[fsky] = dm.read_mask(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)
    w2_masks[fsky] = w_n(masks[fsky], 2)

dust_cl_generate = np.loadtxt(output_path(f'../stage_dust_2pt/{get_dust_2pt_name(args)}'))

for sim in sims_ids:

    np.random.seed(sim)
    dust_gauss_map = hp.sphtfunc.synfast(dust_cl_generate, nside=args.nside) # randomly generated gaussian dust map
    gauss_map_car = reproject.healpix2map(dust_gauss_map, masks['GAL070'].shape, masks['GAL070'].wcs, lmax = args.mlmax, rot="gal,cel")

    enmap.write_map(output_path(get_gauss_dust_map_name(args, sim)), gauss_map_car)

    for fsky in FSKYS:

        gauss_map_car_mask = gauss_map_car * masks[fsky]
        enmap.write_map(output_path(get_gauss_dust_map_name(args, sim, fsky)), gauss_map_car_mask)

        # measure and save 2pt
        # Convert the map to alms and compute the power spectrum
        foreground_alms = cs.map2alm(gauss_map_car_mask, lmax=args.mlmax)
        foreground_cls = cs.alm2cl(foreground_alms) / w2_masks[fsky]

        np.savetxt(output_path(get_dust_2pt_name(args, sim=sim, fsky=fsky)), foreground_cls)
