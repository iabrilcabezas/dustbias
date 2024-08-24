'''
stage_generate_gauss_sims.py "stage_dust_2pt"

Gaussian dust sims to compute meanfield
'''

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import get_dust_2pt_name

print("Foregrounds")
parser = argparse.ArgumentParser(description='Foregrounds')

parser.add_argument("--output-dir", type=str)
parser.add_argument("--tilt", type=float, default=-0.80)
parser.add_argument("--amplitude", type=float, default=119.47982655)
parser.add_argument("--mlmax", type=int, default=4000, help="Maximum ell multipole")

args = parser.parse_args()

output_path = lambda x: os.path.join(args.output_dir, x)

# Imprimir los parámetros ajustados
print(f"Amplitud: {args.amplitude}")
print(f"Inclinación: {args.tilt}")

def power_law(x, amplitude=args.amplitude, tilt=args.tilt):
    # power-law fit to dust map 2pt
    return amplitude * (x+0.001/80)**tilt

def dl_to_cl(dl, l):
    # convert Dl to Cl
    # ell(ell+1)/2pi Cl = Dl
    return dl * 2 * np.pi / l / (l + 1)

ell = np.arange(args.mlmax)

dust_cl_generate = dl_to_cl(power_law(ell, amplitude=args.amplitude, tilt=args.tilt), ell)
dust_cl_generate[0] = 0

fig, ax = plt.subplots(dpi=200)
ax.loglog(ell, dust_cl_generate)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$C_\ell$')
ax.set_title('Dust power spectrum')
fig.savefig(output_path('dust_cl_generate.png'), bbox_inches='tight')
plt.close()

np.savetxt(output_path(get_dust_2pt_name(args)), dust_cl_generate)