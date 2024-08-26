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
from solenspipe.utility import w_n, smooth_cls
from utils import FSKYS, DUST_TYPES # get_dust_2pt_name, get_gauss_dust_map_name
import utils as autils

print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--dust-type", type=str, default='gauss', help='type of dust model, one in DUST_TYPES')
# parser.add_argument("--tilt", type=float, default=-0.80)
# parser.add_argument("--amplitude", type=float, default=119.47982655)
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--nside", type=int, default=2048, help="nside healpix maps")
parser.add_argument("--sims-start", type=int, default=0, help="start sim generation from this sim id")
parser.add_argument("--sims-end", type=int, default=99, help="until sims-end sim id")
parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='3dg')
parser.add_argument("--skyfrac", type=str, default='GAL070')
args = parser.parse_args()

assert args.dust_type in DUST_TYPES, f'Invalid dust type {args.dust_type}'

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

# dust_cl_generate = np.loadtxt(output_path(f'../stage_dust_2pt/{get_dust_2pt_name(args)}'))

cl_fg = np.loadtxt(output_path(f'../stage_data/{autils.get_2pt_scaled_map_name(args.dust_type, sim_id=1000, fsky=args.skyfrac)}'))
cl_fg = smooth_cls(cl_fg, points=300)

for sim in sims_ids:

    np.random.seed(sim)
    # dust_gauss_map = hp.sphtfunc.synfast(dust_cl_generate, nside=args.nside) # randomly generated gaussian dust map
    # gauss_map_car = reproject.healpix2map(dust_gauss_map, masks['GAL070'].shape, masks['GAL070'].wcs, lmax = args.mlmax, rot="gal,cel")
    dust_random_map_alm = cs.rand_alm(cl_fg, lmax=args.mlmax)
    dust_random_map = cs.alm2map(dust_random_map_alm, enmap.empty((1,) + masks['GAL070'].shape, masks['GAL070'].wcs))[0]
    enmap.write_map(output_path(autils.get_dust_map_name(args, sim)), dust_random_map)

    for fsky in FSKYS:

        dust_map_car_mask = dust_random_map * masks[fsky]
        enmap.write_map(output_path(autils.get_dust_map_name(args, sim, fsky)), dust_map_car_mask)

        # # measure and save 2pt
        # # Convert the map to alms and compute the power spectrum
        # foreground_alms = cs.map2alm(gauss_map_car_mask, lmax=args.mlmax)
        # foreground_cls = cs.alm2cl(foreground_alms) / w2_masks[fsky]

        # np.savetxt(output_path(get_dust_2pt_name(args, sim=sim, fsky=fsky)), foreground_cls)
