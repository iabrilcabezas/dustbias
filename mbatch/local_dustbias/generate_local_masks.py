import numpy as np
from pixell import enmap
from math import ceil
import argparse
from sofind import DataModel
import os

parser = argparse.ArgumentParser(description="New Reconstruction Code")

parser.add_argument("--output-dir", type=str,  default=None,help="Output directory.")
parser.add_argument("--width_ra", type=int,  default=15,help="Width of the RA patch.")
parser.add_argument("--width_dec", type=int,  default=10,help="Width of the DEC patch.")

parser.add_argument("--config-name", type=str, default='act_dr6v4')
parser.add_argument("--mask-type", type=str, default='dr6v4_20240919')
parser.add_argument("--mask-subproduct", type=str, default='lensing_masks')
parser.add_argument("--apodfact", type=str, default='_d2_apo3deg')
parser.add_argument("--daynight", type=str, default='night')
parser.add_argument("--skyfrac", type=str, default='GAL070')

args = parser.parse_args()

path=args.output_dir + '/../local_masks/'
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
output_path = lambda x: os.path.join(path, x)

# width_ra=35
# width_dec=20
# width_ra=15
# width_dec=10
# apod_pix=80

dm = DataModel.from_config(args.config_name)
mask_options = {'apodfact': args.apodfact, 'daynight': args.daynight}
mask_options['skyfrac'] = (args.skyfrac).split('GAL0')[1]
mask = dm.read_mask(subproduct=args.mask_subproduct, mask_type=args.mask_type,**mask_options)

def get_patch_masks(mask,width_ra=15,width_dec=10,dec_Start=20,dec_End=-60,ra_Start=180,ra_End=-180, apod_pix=80):
    Ny=ceil((dec_Start-dec_End)/width_dec)
    Nx=ceil((ra_Start-ra_End)/width_ra)
    mask[np.where(mask!=1)]=0
    dec_centers=[]
    ra_centers=[]
    for i in range(Ny):
        dec_centers.append(dec_End+(width_dec/2)+i*(width_dec))
    for i in range(Nx):
        ra_centers.append(ra_End+(width_ra/2)+i*(width_ra))
    #save the bounding box
    mask_box=[]
    masks=[]
    for dec in dec_centers:
        for ra in ra_centers:
            dec_rad,ra_rad = np.deg2rad(np.array((dec,ra)))
            width_ra_rad = np.deg2rad(width_ra)
            width_dec_rad=np.deg2rad(width_dec)
            box = np.array([[dec_rad-width_dec_rad/2.,ra_rad-width_ra_rad/2.],[dec_rad+width_dec_rad/2.,ra_rad+width_ra_rad/2.]])
            stamp = mask.submap(box)
            if not(np.all(stamp==0)):
                mask_box.append(box)
                stamp[np.where(stamp!=1)]=0
                if not(np.all(stamp==1)):
                    deg = 1
                    r = np.deg2rad(deg)
                    apodized = 0.5*(1-np.cos(stamp.distance_transform(rmax=r)*(np.pi/r)))
                    taper = enmap.apod(apodized*0+1,apod_pix)
                    masks.append(apodized*taper)
                elif (np.all(stamp==1)):
                    taper = enmap.apod(stamp*0+1,apod_pix)
                    masks.append(stamp*taper)
    return masks,mask_box

masks,mask_box=get_patch_masks(mask,args.width_ra,args.width_dec)

np.save(output_path(f"submap_coordinates_{args.skyfrac}_{args.width_ra}_{args.width_dec}.npy"),np.array(mask_box))

for i in range(len(masks)):
    enmap.write_fits(output_path(f"mask_{args.skyfrac}_{i}.fits"), masks[i], extra={})
