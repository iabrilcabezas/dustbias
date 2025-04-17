import os
import healpy as hp
from healpy.rotator import Rotator
import numpy as np
import pysm3
import pysm3.units as u
from solenspipe import get_qfunc
from falafel import utils as futils
from falafel import qe
from orphics import stats
from pixell import curvedsky as cs
import pytempura

EST_NORM_LIST = ['TT', 'TE', 'TB', 'EB', 'EE', 'MV', 'MVPOL', 'SRC']
FG_PATH_DICT = {'dust_van_90.0': '/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/dust_sims/vans_d1_SOS4_090_tophat_map_2048',
                'dust_DF_90.0':  '/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/dust_sims/DustFilaments_TQU_NS2048_Nfil180p5M_LR71Normalization_95p0GHz',
                'dust_van_150.0': '/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/dust_sims/vans_d1_SOS4_150_tophat_map_2048', 
                'dust_DF_150.0': '/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/dust_sims/DustFilaments_TQU_NS2048_Nfil180p5M_LR71Normalization_150p0GHz',}
# FSKYS = ['GAL070', 'GAL060', 'GAL040']
FSKYS = ['GAL060', 'GAL070', 'GAL080']
DUST_TYPES = ['d9', 'd10', 'd12', 'van', 'DF'] # 'gauss', 

# def bandedcls(cl,_bin_edges):
#     ls=np.arange(cl.size)
#     binner = stats.bin1D(_bin_edges)
#     cents,bls = binner.bin(ls,cl)
#     return cents,bls 

def get_ell_arrays(lmax):

    '''
    defines ell array and ell factors
    '''

    ell_array = np.arange(lmax + 1)
    ell_factor2 = (ell_array * (ell_array + 1.))**2 / (2. * np.pi )

    return ell_array, ell_factor2

def get_filter_name(args):

    return f'white_{args.filter_whiteamplitude}_{args.filter_whitefwhm}'

def get_name_ellrange(args):
    return f'lmin{args.lmin}_lmax{args.lmax}'

def get_norm_name(args):
    ell_range = get_name_ellrange(args)
    filter_label = get_filter_name(args)
    return f'Als_{filter_label}_{ell_range}.npy'

def get_noise_dict_name(args):

    filter_label = get_filter_name(args)
    return f'noise_dict_{filter_label}.npy'

def get_map_name(dust_type, freq, sim_id, fsky=None):
    tag = f'{dust_type}_{int(freq)}_{sim_id}'
    if fsky is not None:
        tag = f'{tag}_{fsky}'
    return f'map_car_{tag}.fits'

def get_recons_name(ipatch, args, tag, mf=False, ph=False):

    '''
    args.lmin
    args.lmax
    args.filter_whiteamplitude
    args.filter_whitefwhm
    args.skyfrac

    tag: ['N0', 'auto', 'mf_2pt']
    '''
    assert tag in ['N0', 'auto', 'mf_2pt'], 'tag must be one of N0, auto, mf_2pt'
    
    ell_range = get_name_ellrange(args)
    filter_label = get_filter_name(args)

    if mf:
        tag += f'_mf{args.nsims_mf}'
    else:
        tag += f'_nomf'

    if ph:
        ph_tag = 'ph'
    else:
        ph_tag = 'noph'

    tag += f'_{args.fg_type}_{args.freq:.1f}GHz_{filter_label}_{args.skyfrac}_{ell_range}'

    return f'recons_{tag}_{ph_tag}_{ipatch}.txt'

def get_px_frommask(args):
    '''
    initializes pixelization object from mask information
    mask: str
        path to mask
    '''

    px = qe.pixelization(shape=args.shape,wcs=args.wcs)
    return px

def compute_for_filters(args):
    
    
    ucls, tcls = futils.get_theory_dicts(nells=None, lmax=args.mlmax, grad=True)

    filters = futils.get_theory_dicts_white_noise(args.filter_whitefwhm, args.filter_whiteamplitude, lmax=args.mlmax, grad=False)[1]
    # tcls gets overwritten because filter stage has the correct cell (lensed) + noise
    tcls['TT'] = filters['TT']
    tcls['TE'] = filters['TE']
    tcls['EE'] = filters['EE']
    tcls['BB'] = filters['BB']
    
    noise = {'TT': filters['TT'][:args.mlmax+1], 'EE': filters['EE'][:args.mlmax+1], 'BB': filters['BB'][:args.mlmax+1]} # Ignoring TE already
    
    return ucls, tcls, filters, noise

def get_Als_Res(args):
    
    profile = np.loadtxt(args.profile_path)
    
    ucls, tcls, _, _ = compute_for_filters(args)

    Als = pytempura.get_norms(EST_NORM_LIST, ucls, ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)
    Als_ph = pytempura.get_norms(EST_NORM_LIST, ucls, ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)
    
    R_src_tt = pytempura.get_cross('SRC', 'TT', ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)
    R_prof_tt = pytempura.get_cross('SRC', 'TT', ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)
    
    return Als, Als_ph, R_src_tt, R_prof_tt

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
    args.est: estimator
        one of TT,TE,EE,EB,TB,MV,MVPOL
    args.bh: bool
        Option to include bias hardening
    args.ph: bool
        Option to include profile hardening within bias hardening
    args.config_name: str
        sofind datamodel
    args.shape: tuple
        mask shape
    args.wcs: wcs
        mask wcs

    Returns
    qfunc(X, Y), corresponding to args.est 
    '''

    px = get_px_frommask(args)

    # CMB cls for response and total cls (includes noise) - select only ucls
    ucls = futils.get_theory_dicts(lmax = args.mlmax, grad = True)[0]
    
    norm_output = get_Als_Res(args)

    if args.bh:
        est2 = 'SRC'
        Aest2 = 'SRC'
        if args.ph:
            profile = np.loadtxt(args.profile_path)
            R_tt = norm_output[3]
        else:
            profile = None
            R_tt = norm_output[2]
        Als = norm_output[1]
        qfunc = get_qfunc(px,ucls,args.mlmax,args.est,Al1=Als[args.est],est2=est2,Al2=Als[Aest2],Al3=Als['TT'],R12=R_tt,profile=profile)

    else:
        Als = norm_output[0]
        qfunc = get_qfunc(px,ucls,args.mlmax,args.est,Al1=Als[args.est],est2=None,Al2=None,R12=None)

    return qfunc
