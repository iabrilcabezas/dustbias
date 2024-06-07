
import healpy as hp
from healpy.rotator import Rotator
import numpy as np
import pysm3
import pysm3.units as u


def hp_rotate(map_hp, coord):
    """Rotate healpix map between coordinate systems

    :param map_hp: A healpix map in RING ordering
    :param coord: A len(2) list of either 'G', 'C', 'E'
    Galactic, equatorial, ecliptic, eg ['G', 'C'] converts
    galactic to equatorial coordinates
    :returns: A rotated healpix map
    """
    if map_hp is None:
        return None
    if coord[0] == coord[1]:
        return map_hp
    rotator_func = Rotator(coord=coord)
    new_map = rotator_func.rotate_map_pixel(map_hp)
    return new_map

def get_pysm_model_muKcmb_GAL(dust_subtype, nside=2048, freq_GHz=95):
    """
    Get a dust map in uK_CMB at a given frequency and nside, in Equatorial coordinates

    :param dust_subtype: The PySM dust model to use, e.g. 'd1', 'd2', 'd3'
    :param nside: The nside of the output map
    :param freq_GHz: The frequency of the output map in GHz
    :returns: A numpy array (T, Q, U) of the dust map in uK_CMB
    """
    # output frequency
    freq_out = freq_GHz * u.GHz 
    # sky map
    sky = pysm3.Sky(nside = nside, preset_strings = [dust_subtype])
    # at given frequency
    dustmap = sky.get_emission(freq_out)
    # convert to muK_CMB units
    dustmap_muKcmb = dustmap.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq_out))
    # return map (I, Q, U) in Galactic coordinates
    return dustmap_muKcmb.value