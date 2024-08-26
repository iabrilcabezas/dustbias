import argparse
import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
import os
from sofind import DataModel
from pixell import reproject, enmap, curvedsky as cs
from solenspipe.utility import w_n
import matplotlib.pyplot as plt
from utils import FG_PATH_DICT, FSKYS, DUST_TYPES
import utils as autils

# MPI ERROR HANDLER:
import mpi4py.rc
mpi4py.rc.threads = False
from auxiliary import MPI_error_handler

# fit PL
def power_law(x, amplitude, tilt):
    return amplitude * (x+0.001/80)**tilt 
def dl_to_cl(dl, l):
    # convert Dl to Cl
    # ell(ell+1)/2pi Cl = Dl
    return dl * 2 * np.pi / l / (l + 1)

print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")
parser.add_argument("--nside", type=int, default=2048, help="nside healpix maps")
parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='3dg')
parser.add_argument("--dust-freq", type=float, default=95)
parser.add_argument("--hpmask-fname", type=str, default='/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/Planck/HFI_Mask_GalPlane-apo5_2048_R2.00.fits')
parser.add_argument("--hpmask-fname-field", type=int, default=3)
parser.add_argument("--tilt-gauss", type=float, default=-0.80)
parser.add_argument("--amplitude-gauss", type=float, default=119.47982655)
args = parser.parse_args()

rescale = False

output_path = lambda x: os.path.join(args.output_dir, x)

dm = DataModel.from_config(args.config_name)
mask_options = {'apodfact': args.apodfact}

masks = {}
w2_masks = {}
for fsky in FSKYS:
    mask_options['skyfrac'] = fsky
    masks[fsky] = dm.read_mask(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)
    w2_masks[fsky] = w_n(masks[fsky], 2)

hp_mask =  hp.read_map(args.hpmask_fname, field=args.hpmask_fname_field)
w2_hpmask = np.mean(hp_mask**2)

# compute 2pt
ell = np.arange(args.mlmax+1)
lfac = ell * (ell + 1) / (2 * np.pi)

sky_hp_maps = {}
for dust_type in ['d9', 'd10', 'd12']:

    print(dust_type)
    sky_dust = pysm3.Sky(nside=args.nside, preset_strings=[dust_type])
    map_freqGHz = sky_dust.get_emission(args.dust_freq * u.GHz)
    sky_hp_maps[dust_type] = map_freqGHz.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(args.dust_freq*u.GHz))[0]

sky_hp_maps['DF'] = hp.read_map(FG_PATH_DICT['dust_DF'] + '.fits', field=0)
sky_hp_maps['van'] = hp.read_map(FG_PATH_DICT['dust_van'] + '.fits', field=0)

dust_cl_generate = dl_to_cl(power_law(ell, amplitude=args.amplitude_gauss, tilt=args.tilt_gauss), ell)
dust_cl_generate[0] = 0
np.random.seed(1000)
sky_hp_maps['gauss'] = hp.sphtfunc.synfast(dust_cl_generate, nside=args.nside) # randomly generated gaussian dust map
    
assert all(key in DUST_TYPES for key in sky_hp_maps.keys())

dust_hp_cls = {}
for dust_type in DUST_TYPES:
    anafast_cl = hp.anafast(sky_hp_maps[dust_type] * hp_mask, lmax=args.mlmax)
    dust_hp_cls[dust_type] = anafast_cl / w2_hpmask

# plot
plt.figure()
for dust_type in DUST_TYPES:
    plt.plot(ell, lfac * dust_hp_cls[dust_type], label=dust_type)
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi$')
plt.savefig(output_path('dust_raw_cls.png'))
plt.close()

if rescale:
    # re-scale at pivot scale
    factor = {}
    for dust_type in DUST_TYPES:
        factor[dust_type] = (dust_hp_cls[dust_type] / dust_hp_cls['gauss'])[80]

    # plot
    plt.figure()

    for dust_type in DUST_TYPES:
        plt.plot(ell, lfac * dust_hp_cls[dust_type] / factor[dust_type], label = dust_type)
        
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(0.1, 100)
    plt.axvline(80, linestyle='dashed', color='gray', linewidth=3)
    plt.legend()
    plt.savefig(output_path('dust_rescaled_cls.png'))
    plt.close()

    rescaled_sky_hp_maps = {}
    for dust_type in DUST_TYPES:
        rescaled_sky_hp_maps[dust_type] = sky_hp_maps[dust_type] / np.sqrt(factor[dust_type])

# # measure 2pt
# rescaled_dust_hp_cls = {}
# for dust_type in DUST_TYPES:
#     anafast_cl = hp.anafast(rescaled_sky_hp_maps[dust_type] * hp_mask, lmax=args.mlmax)
#     rescaled_dust_hp_cls[dust_type] = anafast_cl / w2_hpmask

# ell_min=100
# ell_max=2000

# # Filtrar los datos para el rango seleccionado
# ell_range = (ell >= ell_min) & (ell <= ell_max)
# ell_fit = ell[ell_range]

# fit_scaled_params = {key: {'amplitude': None, 'tilt': None} for key in DUST_TYPES}
# for dust_type, dust_2pt in rescaled_dust_hp_cls.items():

#     if dust_type == 'gauss':
#         fit_scaled_params[dust_type]['amplitude'], fit_scaled_params[dust_type]['tilt'] = args.amplitude_gauss, args.tilt_gauss
#     else:
#         fit_2pt = dust_2pt * (ell * (ell+1) / (2 * np.pi))
#         fit_scaled_params[dust_type]['amplitude'], fit_scaled_params[dust_type]['tilt'] = curve_fit(power_law, ell_fit, fit_2pt[ell_range])[0]

# np.save(output_path(autils.get_name_fitparams()), fit_scaled_params)

# # save 2pt params
# for dust_type, fit_params in fit_scaled_params.items():
#     dust_cl_generate = dl_to_cl(power_law(ell, amplitude=fit_params['amplitude'], tilt=fit_params['tilt']), ell)
#     dust_cl_generate[0] = 0
#     np.savetxt(output_path(autils.get_dustfgtype_2pt_name(dust_type,fit_params['amplitude'], fit_params['tilt'], sim_id=1000)), dust_cl_generate)

# save maps:
if rescale:

    for dust_type in DUST_TYPES:
        rescaled_sky_car_map = reproject.healpix2map(rescaled_sky_hp_maps[dust_type], masks['GAL070'].shape, masks['GAL070'].wcs, lmax = args.mlmax, rot="gal,cel")
        enmap.write_map(output_path(autils.get_scaled_map_name(dust_type, sim_id=1000)), rescaled_sky_car_map)
        
        for fsky in FSKYS:
            rescaled_sky_car_maps_mask = rescaled_sky_car_map * masks[fsky]
            enmap.write_map(output_path(autils.get_scaled_map_name(dust_type, sim_id=1000, fsky=fsky)), rescaled_sky_car_maps_mask)

            # measure and save 2pt
            # Convert the map to alms and compute the power spectrum
            foreground_alms = cs.map2alm(rescaled_sky_car_maps_mask, lmax=args.mlmax)
            foreground_cls = cs.alm2cl(foreground_alms) / w2_masks[fsky]

            np.savetxt(output_path(autils.get_2pt_scaled_map_name(dust_type, sim_id=1000, fsky=fsky)), foreground_cls)

else: 
    # save maps:
    for dust_type in DUST_TYPES:
        sky_car_map = reproject.healpix2map(sky_hp_maps[dust_type], masks['GAL070'].shape, masks['GAL070'].wcs, lmax = args.mlmax, rot="gal,cel")
        enmap.write_map(output_path(autils.get_map_name(dust_type, sim_id=1000)), sky_car_map)
        
        for fsky in FSKYS:
            sky_car_maps_mask = sky_car_map * masks[fsky]
            enmap.write_map(output_path(autils.get_map_name(dust_type, sim_id=1000, fsky=fsky)), sky_car_maps_mask)

            # measure and save 2pt
            # Convert the map to alms and compute the power spectrum
            foreground_alms = cs.map2alm(sky_car_maps_mask, lmax=args.mlmax)
            foreground_cls = cs.alm2cl(foreground_alms) / w2_masks[fsky]

            np.savetxt(output_path(autils.get_2pt_map_name(dust_type, sim_id=1000, fsky=fsky)), foreground_cls)
