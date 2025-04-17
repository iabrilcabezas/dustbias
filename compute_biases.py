import argparse
import numpy as np
import os
import healpy as hp
import pytempura
from falafel import utils as futils, qe
import pyfisher
from sofind import DataModel
from pixell import enmap, curvedsky as cs, enplot
from orphics import maps
from solenspipe import get_qfunc
from solenspipe.utility import w_n, smooth_cls
import matplotlib.pyplot as plt
from orphics import cosmology
import pickle

args = argparse.ArgumentParser(description = 'compute biases')

args.add_argument("--output-dir", type=str, help='Output directory to save 4pts')
args.add_argument("--maps-path", type=str, help='path to maps')
args.add_argument("--experiment", dest='exp', type=str, help='experiment name')
args.add_argument("--mlmax", type=int, help='maximum ell to do calculation (resolution)')
args.add_argument("--lmin", type=int, help='minimum ell for analysis')
args.add_argument("--lmax", type=int, help='maximum ell for analysis')
args.add_argument("--estimator", dest='est1', type=str, help='estimator')
args.add_argument("--bias-hardening", dest='bh', default=False, action='store_true', help='Option to include bias hardening')
args.add_argument("--profile-hardening", dest='ph', default=False, action='store_true', help='Option to include profile hardening within bias hardening')
args.add_argument("--profile-path", type=str, default=None, help='path to profile')
args.add_argument("--filter-whitefwhm", type=float, help='white noise fwhm')
args.add_argument("--filter-whiteamplitude", type=float, help='white noise amplitude')
args.add_argument("--fgtypes", type=str, help='foreground types')
args.add_argument("--fskys", type=str, help='fskys')
args.add_argument("--freq", type=int, help='frequency', default=None)
args.add_argument("--ilc", default=False, action='store_true', help='Load ILC maps')
args.add_argument("--ilc-type", type=str, default=None, help='type of ILC method')
args = args.parse_args()

EST_NORM_LIST = ['TT', 'TE', 'TB', 'EB', 'EE', 'MV', 'MVPOL', 'SRC']
FSKY_EXP = {'ACT': ['GAL060', 'GAL070', 'GAL080'],
            'SO':['GAL060','GAL070', 'GAL080']}
MASK_EXP = {'ACT': 'ACT',
            'SO': 'SOLAT'}
FREQ_EXP = {'ACT': [90, 150],
            'SO': [27,39,93,145,225,280]}

PYSM_STR = {'d9': ['d9'],
            'd10': ['d10'],
            'd12': ['d12'],
            's4': ['s4'],
            's5': ['s5'],
            's7': ['s7'],
            'low':['d9', 's4'],
            'medium':['d10', 's5'],
            'high': ['d12','s7']
           }

FG_TYPES = list(PYSM_STR.keys())
FG_TYPES += ['DF_EB0', 'DF_EB', 'vand1', 'vand1s1']

WCS_EXP = {'ACT': 'wcsACT', 'SO': 'newwcsSO'} # 'wcsSO'}

def get_px_frommask(args):
    '''
    initializes pixelization object from mask information
    args:
        args.shape: tuple
            mask shape
        args.wcs: wcs
            mask wcs
    '''

    px = qe.pixelization(shape=args.shape,wcs=args.wcs)
    return px


def get_qfunc_forqe(args):

    '''
    Obtains q_func for cross split estimator

    Geometry is read from mask
    Responses are computed theoretically (lensed power spectra -- Lewis+ 2011)
    Normalization is read from stage_norm stage

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
    args.profile_path: str
        path to profile
    args.shape: tuple
        mask shape
    args.wcs: wcs
        mask wcs

    Returns
    qfunc(X, Y), corresponding to args.est1 
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
        qfunc = get_qfunc(px,ucls,args.mlmax,args.est1,Al1=Als[args.est1],est2=est2,Al2=Als[Aest2],Al3=Als['TT'],R12=R_tt,profile=profile)

    else:
        Als = norm_output[0]
        qfunc = get_qfunc(px,ucls,args.mlmax,args.est1,Al1=Als[args.est1],est2=None,Al2=None,R12=None)

    return qfunc

def compute_for_filters(args):
    
    '''
    args:
    
    args.mlmax: int
        maximum ell to do calculation (resolution)
    args.filter_whitefwhm: float
        white noise fwhm
    args.filter_whiteamplitude: float
        white noise amplitude
    '''
    
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
    
    '''
    args:
    args.profile_path: str
        path to profile
    args.mlmax: int
        maximum ell to do calculation (resolution)
    args.lmin, args.lmax: int
        minimum/maximum ell for analysis
    '''
    
    profile = np.loadtxt(args.profile_path)
    
    ucls, tcls, _, _ = compute_for_filters(args)

    Als = pytempura.get_norms(EST_NORM_LIST, ucls, ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)
    Als_ph = pytempura.get_norms(EST_NORM_LIST, ucls, ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)
    
    R_src_tt = pytempura.get_cross('SRC', 'TT', ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=None)
    R_prof_tt = pytempura.get_cross('SRC', 'TT', ucls, tcls, args.lmin, args.lmax, k_ellmax=args.mlmax, profile=profile)
    
    return Als, Als_ph, R_src_tt, R_prof_tt


def get_map_filter(oalms, Noise, args):
    '''
    apply filter (diagonal inverse-variance) to input alms
    assumes covariance used in filter is diagonal in harmonic space
    
    args.lmin, args.lmax: int
        minimum/maximum ell for analysis
    '''
    cltt, clee, clbb = Noise['TT'], Noise['EE'], Noise['BB']

    ls = np.arange(len(cltt))

    nells_t = maps.interp(ls, cltt)
    nells_e = maps.interp(ls, clee)
    nells_b = maps.interp(ls, clbb)

    filt_t = 1./ (nells_t(ls))
    filt_e = 1./ (nells_e(ls))
    filt_b = 1./ (nells_b(ls))
    
    if oalms.shape[0] == 3:
        print('3 component map')
        almt = qe.filter_alms(oalms[0].copy(), filt_t, lmin = args.lmin, lmax = args.lmax)
        alme = qe.filter_alms(oalms[1].copy(), filt_e, lmin = args.lmin, lmax = args.lmax)
        almb = qe.filter_alms(oalms[2].copy(), filt_b, lmin = args.lmin, lmax = args.lmax)
    
    else:
        almt = qe.filter_alms(oalms.copy(), filt_t, lmin = args.lmin, lmax = args.lmax)
        alme = np.zeros_like(almt)
        almb = np.zeros_like(almt)

    return almt, alme, almb

def get_map_name(fgkey, sim_id=1000, fsky=None, IQU=True, alm=False, wcs=None, ilc_type=None):
    tag = f'{fgkey}_{sim_id}'
    if alm:
        assert wcs is not None
        assert ilc_type is not None
        maptype = 'alm'
        tag = f'{tag}_{wcs}_{ilc_type}'
    else:
        maptype = 'map'
    if fsky is not None:
        tag = f'{tag}_{fsky}'
    if IQU:
        return f'{maptype}_car_{tag}_IQU.fits'
    else: 
        return f'{maptype}_car_{tag}.fits'

def compute_biases(args, mask, fgkey, alm=False, fsky=None, wcs=None, ilc_type=None):
    if not alm:
        dust_map = mask * enmap.read_map(maps_path(get_map_name(fgkey, 1000)))
        foreground_alms = cs.map2alm(dust_map, lmax=args.mlmax)
        #foreground_cls = cs.alm2cl(foreground_alms) / w_n(mask,2)
        #print(foreground_cls.shape)

    else:
        assert wcs is not None
        assert ilc_type is not None
        assert fsky is not None
        foreground_alms = hp.read_alm(maps_path(get_map_name(fgkey, 1000, alm=alm, fsky=fsky, wcs=wcs, ilc_type=ilc_type)), hdu=(1,2,3))
    
    #print(Noise['TT'].shape)
    #np.save(f'/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/update_filters_so/Noise_{fgkey}.npy', Noise)
    #Noise['TT'] += smooth_cls(foreground_cls[0], points=300)## !!
    #Noise['EE'] += smooth_cls(foreground_cls[1], points=300)
    #Noise['BB'] += smooth_cls(foreground_cls[2], points=300)
    #np.save(f'/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/update_filters_so/Noise_fgs_{fgkey}.npy', Noise)
    almt, alme, almb = get_map_filter(foreground_alms, Noise, args)
    f_alms = np.array([almt, alme, almb])

    # Compute the 4-point function of the filtered alms
    foreground_4pt = qfunc(f_alms , f_alms)
    # # Compute the power spectrum of the 4-point function, [0] for grad
    cls_4pt = cs.alm2cl(foreground_4pt[0]) / w_n(mask, 4)

    return cls_4pt 

output_path = args.output_dir + '/../stage_reconstruction/'
os.makedirs(output_path, exist_ok=True)

maps_path = lambda x: os.path.join(args.maps_path, x)

args.fgtypes = args.fgtypes.split()
args.fskys = args.fskys.split()
assert all([x in FSKY_EXP[args.exp] for x in args.fskys]), 'fskys not recognized'
assert all([x in FG_TYPES for x in args.fgtypes]), 'Foreground types not recognized'
if not args.ilc:
    assert args.freq in FREQ_EXP[args.exp], 'Frequency not recognized'
    wcs_tag = None

if args.ilc:
    assert args.ilc_type is not None, 'ILC type not specified'
    wcs_tag = WCS_EXP[args.exp]

masks = {}

for fsky in FSKY_EXP[args.exp]:
    masks[fsky] = enmap.read_map(f'/rds/project/dirac_vol5/rds-dirac-dp002/ia404/fgs/masks/{MASK_EXP[args.exp]}_{fsky}_{WCS_EXP[args.exp]}_car_apo3deg.fits')

args.shape, args.wcs = masks[fsky].shape, masks[fsky].wcs

w4_factors = {}
for fsky in masks.keys():
    w4_factors[fsky] = w_n(masks[fsky],4)
    
Noise = compute_for_filters(args)[3]

qfunc = get_qfunc_forqe(args)

raw4pt = {}
for fsky in args.fskys:
    for fg_type in args.fgtypes:
        print(fsky, fg_type)
        if args.ilc:
            key = f'{fg_type}'
        else:
            key = f'{args.freq}_{fg_type}'
        raw4pt[f'{key}_{fsky}'] = compute_biases(args, masks[fsky], key, alm=args.ilc, fsky=fsky, wcs=wcs_tag, ilc_type=args.ilc_type)

tag_fgtypes = '_'.join(args.fgtypes)
tag_run = f'{args.exp}_{f"ilc_{args.ilc_type}" if args.ilc else args.freq}_{args.est1}_bh{args.bh}_ph{args.ph}_{args.mlmax}_{args.lmin}_{args.lmax}'

with open(output_path + f'raw4pts_{tag_run}_{tag_fgtypes}_{"_".join(args.fskys)}.pkl', 'wb') as f:
    pickle.dump(raw4pt, f)