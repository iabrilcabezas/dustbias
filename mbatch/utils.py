
import healpy as hp
from healpy.rotator import Rotator
import numpy as np
import pysm3
import pysm3.units as u
from solenspipe import get_qfunc
from falafel import utils as futils
from falafel import qe
from orphics import stats

EST_NORM_LIST = ['TT', 'TE', 'TB', 'EB', 'EE', 'MV', 'MVPOL']
FG_PATH_DICT = {'dust_van': '/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/dust_sims/vans_d1_SOS4_090_tophat_map_2048',
                'dust_DF':  '/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/dust_sims/DustFilaments_TQU_NS2048_Nfil180p5M_LR71Normalization_95p0GHz'}

def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return cents,bls

def get_dust_name(args):

    return f'dust_{args.dust_type}_muK_{args.dust_freq:.0f}GHz_mask_{args.skyfrac}_fejer1'

def get_filter_name(args):

    return f'white_{args.filter_whiteamplitude}_{args.filter_whitefwhm}'

def get_auto_name(args):

    return f'phi_{args.est}_{get_dust_name(args)}_{get_filter_name(args)}_{get_name_ellrange(args)}'

def get_name_ellrange(args):
    return f'lmin{args.lmin}_lmax{args.lmax}'

def get_norm_name(args):
    ell_range = get_name_ellrange(args)
    filter_label = get_filter_name(args)
    return f'Als_{filter_label}_{ell_range}.npy'

def get_noise_dict_name(args):

    filter_label = get_filter_name(args)
    return f'noise_dict_{filter_label}.npy'


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

def get_px_frommask(args):
    '''
    initializes pixelization object from mask information
    mask: str
        path to mask
    '''

    px = qe.pixelization(shape=args.shape,wcs=args.wcs)
    return px

def get_qfunc_forqe(args):

    '''
    Obtains q_func for cross split estimator

    Geometry is read from mask
    Responses are computed theoretically (lensed power spectra -- Lewis+ 2011)
    Normalization is read from stage_norm stage

    args.norm_dir: path
        path to normalization filters
    args.mlmax: int
        maximum ell to do calculation (resolution)
    args.lmin, args.lmax: int
        minimum/maximum ell for analysis
    args.est1: estimator
        one of TT,TE,EE,EB,TB,MV,MVPOL
    args.bh: bool
        Option to include bias hardening
    args.ph: bool
        Option to include profile hardening within bias hardening
    args.config_name: str
        sofind datamodel
    args.mask_type: str
        lensing masks type (wide_v4_20220316, wide_v3_20220316, deep_v3)
    args.mask_skyfrac: str
        sky fraction of mask (GAL040, GAL060, GAL070)
    args.mask_apodfact: str
        apodization of mask (3dg, None)

    Returns
    qfunc(X, Y), corresponding to args.est1 
    '''

    px = get_px_frommask(args)

    # CMB cls for response and total cls (includes noise) - select only ucls
    ucls = futils.get_theory_dicts(lmax=args.mlmax, grad=True)[0]

    
    Als = np.load(f'{args.norm_dir}/Als_{filter_label}_lmin{args.lmin}_lmax{args.lmax}.npy',allow_pickle='TRUE').item()

    qfunc = get_qfunc(px, ucls, args.mlmax, args.est, Al1=Als[args.est], est2=None, Al2=None, R12=None)

    return qfunc