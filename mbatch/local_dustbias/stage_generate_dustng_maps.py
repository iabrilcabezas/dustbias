import argparse
import pickle
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import os
from sofind import DataModel
from pixell import reproject, enmap
from utils import FG_PATH_DICT, FSKYS, DUST_TYPES
import utils as autils


# MPI ERROR HANDLER:
import mpi4py.rc
mpi4py.rc.threads = False
from auxiliary import MPI_error_handler


print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--nside", type=int, default=2048, help="nside healpix maps")

parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='dr6v4_20240919')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='_d2_apo3deg')
parser.add_argument("--daynight", type=str, default='night')

parser.add_argument("--dust-freq", type=float, default=95)
parser.add_argument("--hpmask-fname", type=str, default='/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/Planck/HFI_Mask_GalPlane-apo5_2048_R2.00.fits')

parser.add_argument("--tilt-gauss", type=float, default=-0.80)
parser.add_argument("--amplitudes-path", type=str, default='/home/ia404/gitreps/dustbias/mbatch/local_dustbias/amplitudes.pkl')
args = parser.parse_args()

output_dir = args.output_dir + '/../stage_generate_dustng_maps/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
output_path = lambda x: os.path.join(output_dir, x)

dm = DataModel.from_config(args.config_name)
mask_options = {'apodfact': args.apodfact, 'daynight': args.daynight}

masks_pixell = {}
for fsky in FSKYS:
    mask_options['skyfrac'] = fsky.split('GAL0')[1]
    masks_pixell[fsky] = dm.read_mask(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)


hp_masks = {}
w2_hpmasks = {}
for i, fsky in enumerate(FSKYS):
    hp_masks[fsky] =  hp.read_map(args.hpmask_fname, field=2+i)
    w2_hpmasks[fsky] = np.mean(hp_masks[fsky]**2)

with open(args.amplitudes_path, 'rb') as file:
    amplitudes = pickle.load(file)

# compute 2pt
ell = np.arange(args.mlmax+1)
lfac = ell * (ell + 1) / (2 * np.pi)

sky_hp_maps = {}
for dust_type in ['d9', 'd10', 'd12']:

    print(dust_type)
    sky_dust = pysm3.Sky(nside=args.nside, preset_strings=[dust_type])
    map_freqGHz = sky_dust.get_emission(args.dust_freq * u.GHz)
    sky_hp_maps[dust_type] = map_freqGHz.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(args.dust_freq*u.GHz))[0]

sky_hp_maps['DF'] = hp.read_map(FG_PATH_DICT[f'dust_DF_{args.dust_freq}'] + '.fits', field=0)
sky_hp_maps['van'] = hp.read_map(FG_PATH_DICT[f'dust_van_{args.dust_freq}'] + '.fits', field=0)

dust_hp_cls = {}
rescaled_sky_hp_maps = {}

for fsky in FSKYS:
    amplitude_gauss = amplitudes[f'{args.dust_freq:.1f}_d9_{fsky}']
    dust_cl_generate = autils.dl_to_cl(autils.power_law(ell, amplitude=amplitude_gauss, tilt=args.tilt_gauss), ell)
    dust_cl_generate[0] = 0
    np.random.seed(1000)
    dust_hp_cls[f'gauss_{fsky}'] = dust_cl_generate # randomly generated gaussian dust map
    sky_hp_maps[f'gauss_{fsky}'] = hp.sphtfunc.synfast(dust_cl_generate, nside=args.nside) # randomly generated gaussian dust map
    rescaled_sky_hp_maps[f'gauss_{fsky}'] = sky_hp_maps[f'gauss_{fsky}']
    
factor = {}
for dust_type in ['d9', 'd10', 'd12', 'van', 'DF']:
    for fsky in FSKYS:
        anafast_cl = hp.anafast(sky_hp_maps[dust_type] * hp_masks[fsky], lmax=args.mlmax)
        dust_hp_cls[f'{dust_type}_{fsky}'] = anafast_cl / w2_hpmasks[fsky]
        factor[f'{dust_type}_{fsky}'] = (dust_hp_cls[f'{dust_type}_{fsky}'] / dust_hp_cls[f'gauss_{fsky}'])[80]
        rescaled_sky_hp_maps[f'{dust_type}_{fsky}'] = sky_hp_maps[dust_type] / np.sqrt(factor[f'{dust_type}_{fsky}'])

for dust_type in DUST_TYPES:
    for fsky in FSKYS:
        rescaled_sky_car_map = reproject.healpix2map(rescaled_sky_hp_maps[f'{dust_type}_{fsky}'], masks_pixell[fsky].shape, masks_pixell[fsky].wcs, lmax = args.mlmax, rot="gal,cel")
        rescaled_sky_car_maps_mask = rescaled_sky_car_map * masks_pixell[fsky]
        enmap.write_map(output_path(autils.get_scaled_map_name(dust_type, sim_id=1000, fsky=fsky, freq=args.dust_freq)), rescaled_sky_car_maps_mask)
