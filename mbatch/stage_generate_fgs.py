# Description: Generate given dust map model for a given frequency and mask
import os
from sofind import DataModel
from pixell import reproject, enmap
from utils import get_pysm_model_muKcmb_GAL
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str)
parser.add_argument("--nside", type=int, default=2048)
parser.add_argument("--dust-type", type=str)
parser.add_argument("--dust-freq", type=float, default=95)
parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='wide_v4_20220316')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--skyfrac", type=str, default='GAL070')
parser.add_argument("--apodfact", type=str, default='3dg')
args = parser.parse_args()

output_dir = f'{args.output_dir}/../stage_generate_fgs/'
os.makedirs(output_dir, exist_ok=True)
output_path = lambda x: os.path.join(output_dir, x)

dust_type = args.dust_type
dust_freq = args.dust_freq
nside = args.nside

mask_options = {'skyfrac': args.skyfrac, 'apodfact': args.apodfact}
dust_model, dust_subtype = dust_type.split('_')

dm = DataModel.from_config(args.config_name)
mask = dm.get_mask_fn(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)

if  dust_model == 'pysm':
    dustmap_muKcmb = get_pysm_model_muKcmb_GAL(dust_subtype, nside=nside, freq_GHz=dust_subtype)

dustmap_muKcmb_EQ = reproject.healpix2map(dustmap_muKcmb,  mask.shape, mask.wcs, rot="gal,cel")

dustmap_muKcmb_EQ *= mask

odust_name = f'dust_{dust_type}_muK_{dust_freq}GHz_mask_{args.skyfrac}_{args.skyfrac}.fits'

enmap.write_map(output_path(odust_name), dustmap_muKcmb_EQ)









