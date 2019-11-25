'''
script for making the mtl file from desitarget output 
'''
import os 
import numpy as np 

import fitsio
import healpy as hp 
from astropy.table import Table
# -- desitarget --
from desitarget.targets import calc_priority, main_cmx_or_sv, set_obsconditions
from desitarget.sv1.sv1_targetmask import bgs_mask


dir_dat = '/global/cscratch1/sd/chahah/feasibgs/survey_validation/'


def make_mtl(targets):
    '''
    '''
    # determine whether the input targets are main survey, cmx or SV.
    colnames, masks, survey = main_cmx_or_sv(targets)
    # ADM set the first column to be the "desitarget" column
    desi_target, desi_mask = colnames[0], masks[0]
    n = len(targets)

    # ADM if the input target columns were incorrectly called NUMOBS or PRIORITY
    # ADM rename them to NUMOBS_INIT or PRIORITY_INIT.
    for name in ['NUMOBS', 'PRIORITY']:
        targets.dtype.names = [name+'_INIT' if col == name else col for col in targets.dtype.names]

    # ADM if a redshift catalog was passed, order it to match the input targets
    # ADM catalog on 'TARGETID'.
    ztargets = Table()
    ztargets['TARGETID'] = targets['TARGETID']
    ztargets['NUMOBS'] = np.zeros(n, dtype=np.int32)
    ztargets['Z'] = -1 * np.ones(n, dtype=np.float32)
    ztargets['ZWARN'] = -1 * np.ones(n, dtype=np.int32)
    # ADM if zcat wasn't passed, there is a one-to-one correspondence
    # ADM between the targets and the zcat.
    zmatcher = np.arange(n)

    # ADM extract just the targets that match the input zcat.
    targets_zmatcher = targets[zmatcher]

    # ADM use passed value of NUMOBS_INIT instead of calling the memory-heavy calc_numobs.
    # ztargets['NUMOBS_MORE'] = np.maximum(0, calc_numobs(ztargets) - ztargets['NUMOBS'])
    ztargets['NUMOBS_MORE'] = np.maximum(0, targets_zmatcher['NUMOBS_INIT'] - ztargets['NUMOBS'])

    # ADM need a minor hack to ensure BGS targets are observed once
    # ADM (and only once) every time during the BRIGHT survey, regardless
    # ADM of how often they've previously been observed. I've turned this
    # ADM off for commissioning. Not sure if we'll keep it in general.

    # ADM only if we're considering bright survey conditions.
    ii = targets_zmatcher[desi_target] & desi_mask.BGS_ANY > 0
    ztargets['NUMOBS_MORE'][ii] = 1

    # ADM assign priorities, note that only things in the zcat can have changed priorities.
    # ADM anything else will be assigned PRIORITY_INIT, below.
    priority = calc_priority(targets_zmatcher, ztargets, 'BRIGHT')

    # set subpriority in order to tune the SV target densities 
    # BGS target classes: BRIGHT, FAINT, EXTFAINT, FIBERMAG, LOWQ
    # initial DR8 target density ---> desired density
    # BRIGHT:   882.056980 ---> 540 = 63%   0.62 - 1  
    # FAINT:    746.769486 ---> 300 = 41%   0.41 - 1
    # EXTFAINT: 623.470673 ---> 150 = 24%   0    - 1
    # FIBERMAG: 207.534409 ---> 150 = 71%   0.66 - 1
    # LOW Q:    55.400240  ---> 60  = 100%  0.76 - 1
    # (depending on imaging LOWQ varies a lot! DES~50/deg2, DECALS~114/deg2, North~185/deg2) 

    # bgs bitmask
    bitmask_bgs = targets['SV1_BGS_TARGET']
    bgs_bright      = (bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool)
    bgs_faint       = (bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool)
    bgs_extfaint    = (bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool) # extended faint
    bgs_fibmag      = (bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool) # fiber magnitude limited
    bgs_lowq        = (bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool) # low quality
   
    subpriority = np.random.uniform(0., 1., n) 
    subpriority[bgs_bright]     = np.random.uniform(0.62, 1., np.sum(bgs_bright))
    subpriority[bgs_faint]      = np.random.uniform(0.41, 1., np.sum(bgs_faint))
    subpriority[bgs_extfaint]   = np.random.uniform(0., 1, np.sum(bgs_extfaint))
    subpriority[bgs_fibmag]     = np.random.uniform(0.66, 1, np.sum(bgs_fibmag))
    subpriority[bgs_lowq]       = np.random.uniform(0.76, 1, np.sum(bgs_lowq))

    # If priority went to 0==DONOTOBSERVE or 1==OBS or 2==DONE, then NUMOBS_MORE should also be 0.
    # ## mtl['NUMOBS_MORE'] = ztargets['NUMOBS_MORE']
    #ii = (priority <= 2)
    #log.info('{:d} of {:d} targets have priority zero, setting N_obs=0.'.format(np.sum(ii), n))
    #ztargets['NUMOBS_MORE'][ii] = 0

    # - Set the OBSCONDITIONS mask for each target bit.
    obsconmask = set_obsconditions(targets)

    # ADM set up the output mtl table.
    mtl = Table(targets)
    mtl.meta['EXTNAME'] = 'MTL'
    # ADM any target that wasn't matched to the ZCAT should retain its
    # ADM original (INIT) value of PRIORITY and NUMOBS.
    mtl['NUMOBS_MORE'] = mtl['NUMOBS_INIT']
    mtl['PRIORITY'] = mtl['PRIORITY_INIT']
    # ADM now populate the new mtl columns with the updated information.
    mtl['OBSCONDITIONS'] = obsconmask
    mtl['PRIORITY'][zmatcher] = priority
    mtl['SUBPRIORITY'][zmatcher] = subpriority
    mtl['NUMOBS_MORE'][zmatcher] = ztargets['NUMOBS_MORE']

    # Filtering can reset the fill_value, which is just wrong wrong wrong
    # See https://github.com/astropy/astropy/issues/4707
    # and https://github.com/astropy/astropy/issues/4708
    mtl['NUMOBS_MORE'].fill_value = -1
    return mtl


def test_mtl(mtl): 
    '''
    '''
    # bgs bitmask
    bitmask_bgs = mtl['SV1_BGS_TARGET']
    bgs_bright      = (bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool)
    bgs_faint       = (bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool)
    bgs_extfaint    = (bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool) # extended faint
    bgs_fibmag      = (bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool) # fiber magnitude limited
    bgs_lowq        = (bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool) # low quality

    
    for name, bgs_class in zip(['bgs_bright', 'bgs_faint', 'bgs_extfaint', 'bgs_fibmag', 'bgs_lowq'],
            [bgs_bright, bgs_faint, bgs_extfaint, bgs_fibmag, bgs_lowq]): 
        print('%s: %.2f - %.2f' % (name, mtl['SUBPRIORITY'][bgs_class].min(), mtl['SUBPRIORITY'][bgs_class].max()))
    return None 


def compile_SV_targets(): 
    '''
    '''
    sv = fitsio.read(os.path.join(dir_dat, 'BGS_SV_30_3x_superset60_Sep2019.fits')) 
    ra, dec = sv['RA'], sv['DEC']

    hp_pixs = np.unique(hp.pixelfunc.ang2pix(2, np.radians(90. - dec), np.radians(360. - ra))) 
    
    dir_targ = '/project/projectdirs/desi/target/catalogs/dr8/0.34.0/targets/sv/resolve/bright/'
   
    ftargs = [os.path.join(dir_targ, 'sv1-targets-dr8-hp-%i.fits' % ipix) for ipix in hp_pixs]
    print(1e-9*np.sum([os.path.getsize(ftarg) for ftarg in ftargs]))
    
    for itarg, ftarg in enumerate(ftargs):
        targ = fitsio.read(ftarg)
        if itarg == 0: 
            targ_comb = targ 
        else: 
            targ_comb = np.concatenate([targ_comb, targ]) 
        print('%i pixel' % itarg) 
        print('targ', targ.shape) 
        print('comb', targ_comb.shape) 
    
    fcomb = os.path.join(dir_dat, 'desitarget.dr8.0.34.0.bgs_sv.hp_comb.fits') 
    fitsio.write(fcomb, targ_comb) 
    return None 


if __name__=="__main__": 
    #targets = fitsio.read('/global/cscratch1/sd/chahah/feasibgs/survey_validation/bright/desitarget-targets-1400deg2.fits') 
    #mtl = make_mtl(targets)
    #mtl.write('/global/cscratch1/sd/chahah/feasibgs/survey_validation/bright/mtl-1400deg2.fits', format='fits') 

    #mtl = fitsio.read('/global/cscratch1/sd/chahah/feasibgs/survey_validation/bright/mtl-1400deg2.fits')
    #test_mtl(mtl)
    
    # full MTL 
    #compile_SV_targets()
    #targets = fitsio.read(os.path.join(dir_dat, 'desitarget.dr8.0.34.0.bgs_sv.hp_comb.fits'))
    targets = fitsio.read(os.path.join(dir_dat, 'sv1-targets-dr8-hp-24.fits'))
    mtl = make_mtl(targets)
    mtl.write(os.path.join(dir_dat, 'mtl.dr8.0.34.0.bgs_sv.hp-24.fits'), format='fits') 

