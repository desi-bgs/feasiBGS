'''
script for making the mtl file from desitarget output 
'''
import os 
import glob
import numpy as np 

import fitsio
import healpy as hp 
from astropy.table import Table
# -- desitarget --
from desitarget.targets import calc_priority, main_cmx_or_sv, set_obsconditions
from desitarget.sv1.sv1_targetmask import bgs_mask


dir_dat = '/global/cscratch1/sd/chahah/feasibgs/survey_validation/'
if not os.path.isdir(dir_dat): 
    dir_dat = '/Users/ChangHoon/data/feasiBGS/survey_validation/'


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
    bgs_all         = (bitmask_bgs).astype(bool)
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
    subpriority[bgs_lowq]       = np.random.uniform(0.99, 1, np.sum(bgs_lowq))

    # set priority of all BGS targets equal 
    priority[bgs_all] = 2000
    
    n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq = \
            bgs_targetclass(targets['SV1_BGS_TARGET'])
    print('---------------------------------')
    print('total n_bgs = %i' % n_bgs)
    print('nobj, frac (ls frac)')  
    print('BGS Bright %i %.3f (0.35)' % (n_bgs_bright, n_bgs_bright/n_bgs))
    print('BGS Faint %i %.3f (0.29)' % (n_bgs_faint, n_bgs_faint/n_bgs))
    print('BGS Ext.Faint %i %.3f (0.25)' % (n_bgs_extfaint, n_bgs_extfaint/n_bgs))
    print('BGS Fib.Mag %i %.3f (0.08)' % (n_bgs_fibmag, n_bgs_fibmag/n_bgs))
    print('BGS Low Q. %i %.3f (0.02)' % (n_bgs_lowq, n_bgs_lowq/n_bgs))

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



def test_mtl(fmtl): 
    '''
    '''
    mtl = fitsio.read(fmtl)
    # bgs bitmask
    bitmask_bgs = mtl['SV1_BGS_TARGET']
    bgs_bright      = (bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool)
    bgs_faint       = (bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool)
    bgs_extfaint    = (bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool) # extended faint
    bgs_fibmag      = (bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool) # fiber magnitude limited
    bgs_lowq        = (bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool) # low quality

    
    for name, bgs_class in zip(['bgs_bright', 'bgs_faint', 'bgs_extfaint', 'bgs_fibmag', 'bgs_lowq'],
            [bgs_bright, bgs_faint, bgs_extfaint, bgs_fibmag, bgs_lowq]): 
        print('--- %s ---' % name) 
        print('PRIORITY: %.2f - %.2f' % (mtl['PRIORITY'][bgs_class].min(), mtl['PRIORITY'][bgs_class].max()))
        print('SUBPRIORITY: %.2f - %.2f' % (mtl['SUBPRIORITY'][bgs_class].min(), mtl['SUBPRIORITY'][bgs_class].max()))
    return None 


def mtl_SV_healpy(): 
    ''' generate MTLs from targets in healpixels with SV tiles 
    '''
    sv = fitsio.read(os.path.join(dir_dat, 'BGS_SV_30_3x_superset60_Sep2019.fits')) 
    phi = np.deg2rad(sv['RA'])
    theta = 0.5 * np.pi - np.deg2rad(sv['DEC'])

    ipixs = np.unique(hp.ang2pix(2, theta, phi, nest=True)) 
    
    dir_sv = '/project/projectdirs/desi/target/catalogs/dr8/0.34.0/targets/sv/resolve/bright/'
    for i in ipixs: 
        print('--- %i pixel ---' % i) 
        targets = fitsio.read(os.path.join(dir_sv, 'sv1-targets-dr8-hp-%i.fits' % i))
        mtl = make_mtl(targets)
        mtl.write(os.path.join(dir_dat, 'mtl.dr8.0.34.0.bgs_sv.hp-%i.fits' % i), format='fits', overwrite=True) 
    return None 


def target_densities(): 
    ''' Examine the target class densities for the different healpixels to
    see the variation in the targeting 
    '''
    dir_sv = '/project/projectdirs/desi/target/catalogs/dr8/0.34.0/targets/sv/resolve/bright/'
    for 

    for i in ipixs: 
        print('--- %i pixel ---' % i) 
        targets = fitsio.read(os.path.join(dir_sv, 'sv1-targets-dr8-hp-%i.fits' % i))
        mtl = make_mtl(targets)
        mtl.write(os.path.join(dir_dat, 'mtl.dr8.0.34.0.bgs_sv.hp-%i.fits' % i), format='fits') 


def master_truth_table(): 
    ''' compile master list of brickid, objid, ra, dec, north or south, name of survey of
    spectroscopic truth tables 
    '''
    import h5py 
    import glob 
    brickid, objid, ra, dec, nors, survey = [], [], [], [], [], []
    for ns in ['north', 'south']: 
        dir_match = '/project/projectdirs/desi/target/analysis/truth/dr8.0/%s/matched/' % ns
        ftabs = glob.glob(dir_match+'ls-dr8.0*') 
        for ftab in ftabs: 
            tab = fitsio.read(ftab)
            print('%i objects in %s' % (tab.shape[0], os.path.basename(ftab)))

            brickid.append(tab['BRICKID'])
            objid.append(tab['OBJID'])
            ra.append(tab['RA'])
            dec.append(tab['DEC'])
            nors.append(np.repeat(ns, tab.shape[0]))
            survey.append(np.repeat(os.path.basename(ftab), tab.shape[0]))
    
    fmaster = h5py.File(os.path.join(dir_dat, 'master_truth_table.hdf5'), 'w') 
    fmaster.create_dataset('BRICKID', data=np.concatenate(brickid))
    fmaster.create_dataset('OBJID', data=np.concatenate(objid))
    fmaster.create_dataset('RA', data=np.concatenate(ra))
    fmaster.create_dataset('DEC', data=np.concatenate(dec))
    fmaster.create_dataset('NORS', data=np.concatenate(nors).astype('S'))
    fmaster.create_dataset('SURVEY', data=np.concatenate(survey).astype('S'))
    fmaster.close() 
    return None  


def bgs_targetclass(bitmask_bgs): 
    n_bgs = np.float(np.sum(bitmask_bgs.astype(bool))) 
    n_bgs_bright    = np.sum((bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool))
    n_bgs_faint     = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool))
    n_bgs_extfaint  = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool)) # extended faint
    n_bgs_fibmag    = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool)) # fiber magnitude limited
    n_bgs_lowq      = np.sum((bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool)) # low quality
    return n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq


'''
    def _make_mtl(targets, test=0):
        '''
        '''
        # determine whether the input targets are main survey, cmx or SV.
        colnames, masks, survey = main_cmx_or_sv(targets)
        print('%s survey' % survey)
        # ADM set the first column to be the "desitarget" column
        desi_target, bgs_target = colnames[0], colnames[1]
        desi_mask, bgs_mask = masks[0], masks[1]
        n = len(targets)

        for name in ['BGS_BRIGHT', 'BGS_FAINT', 'BGS_FAINT_EXT', 'BGS_FIBMAG', 'BGS_LOWQ']: #bgs_mask.names(): 
            print('--- %s ---' % name) 
            print(bgs_mask[name].obsconditions)
            print(np.max([bgs_mask[name].priorities['UNOBS'], bgs_mask[name].priorities['DONE'], bgs_mask[name].priorities['MORE_ZGOOD'], bgs_mask[name].priorities['MORE_ZWARN']])) 


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
        # BRIGHT:   882.056980 (35%) ---> 540 (45%)     = 63%   0.62 - 1  
        # FAINT:    746.769486 (29%) ---> 300 (25%)     = 41%   0.41 - 1
        # EXTFAINT: 623.470673 (25%) ---> 150 (12.5%)   = 24%   0    - 1
        # FIBERMAG: 207.534409 (08%) ---> 150 (12.5%)   = 71%   0.66 - 1
        # LOW Q:    55.400240  (02%) ---> 60  (5%)      = 100%  0.76 - 1
        # (depending on imaging LOWQ varies a lot! DES~50/deg2, DECALS~114/deg2, North~185/deg2) 

        # bgs bitmask
        bitmask_bgs = targets['SV1_BGS_TARGET']
        bgs_all         = (bitmask_bgs).astype(bool)
        bgs_bright      = (bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool)
        bgs_faint       = (bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool)
        bgs_extfaint    = (bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool) # extended faint
        bgs_fibmag      = (bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool) # fiber magnitude limited
        bgs_lowq        = (bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool) # low quality

        # set priority of all BGS targets equal 
        priority[bgs_all] = 2000
        
        n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq = \
                bgs_targetclass(targets['SV1_BGS_TARGET'])
        print('---------------------------------')
        print('total n_bgs = %i' % n_bgs)
        print('BGS Bright %i %.3f (0.35)' % (n_bgs_bright, n_bgs_bright/n_bgs))
        print('BGS Faint %i %.3f (0.29)' % (n_bgs_faint, n_bgs_faint/n_bgs))
        print('BGS Ext.Faint %i %.3f (0.25)' % (n_bgs_extfaint, n_bgs_extfaint/n_bgs))
        print('BGS Fib.Mag %i %.3f (0.08)' % (n_bgs_fibmag, n_bgs_fibmag/n_bgs))
        print('BGS Low Q. %i %.3f (0.02)' % (n_bgs_lowq, n_bgs_lowq/n_bgs))

        print('---------------------------------')
        print('BGS Bright %f' % (1. - (150./n_bgs_extfaint)/(540./n_bgs_bright)))
        print('BGS Faint %f' %  (1. - (150./n_bgs_extfaint)/(300./n_bgs_faint)))
        print('BGS Ext.Faint %f' % (1. - (150./n_bgs_extfaint)/(150./n_bgs_extfaint)))
        print('BGS Fib.Mag %f' % (1. - (150./n_bgs_extfaint)/(150./n_bgs_fibmag)))
        print('BGS Low Q. %f' % (1. - (150./n_bgs_extfaint)/(60./n_bgs_lowq)))

        # original 
        # total n_bgs = 833892
        # BGS Bright    272564 0.327
        # BGS Faint     236859 0.284
        # BGS Ext.Faint 200643 0.241
        # BGS Fib.Mag    87624 0.105
        # BGS Low Q.     36202 0.043
       
        subpriority = np.random.uniform(0., 1., n) 
        subpriority[bgs_bright]     = np.random.uniform(0.35, 1., np.sum(bgs_bright))
        subpriority[bgs_faint]      = np.random.uniform(0., 1., np.sum(bgs_faint))
        subpriority[bgs_extfaint]   = np.random.uniform(0., 1, np.sum(bgs_extfaint))
        subpriority[bgs_fibmag]     = np.random.uniform(0.66, 1, np.sum(bgs_fibmag))
        subpriority[bgs_lowq]       = np.random.uniform(0.76, 1, np.sum(bgs_lowq))
        #if test == 0: 
        #    subpriority[bgs_lowq]       = np.random.uniform(0.99, 1, np.sum(bgs_lowq))
        #elif test == 1: 
        #    subpriority[bgs_lowq]       = np.random.uniform(0.76, 1, np.sum(bgs_lowq))

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


    def _mtl_subpriority_test(): 
        ''' generate MTLs from targets in healpixels with SV tiles 
        '''
        sv = fitsio.read(os.path.join(dir_dat, 'BGS_SV_30_3x_superset60_Sep2019.fits')) 
        phi = np.deg2rad(sv['RA'])
        theta = 0.5 * np.pi - np.deg2rad(sv['DEC'])

        ipixs = np.unique(hp.ang2pix(2, theta, phi, nest=True)) 
        
        dir_sv = '/project/projectdirs/desi/target/catalogs/dr8/0.34.0/targets/sv/resolve/bright/'
        targets = fitsio.read(os.path.join(dir_sv, 'sv1-targets-dr8-hp-0.fits'))
        mtl = _make_mtl(targets, test=0)
        mtl.write(os.path.join(dir_dat, 'subpriority_test', 'mtl.dr8.0.34.0.bgs_sv.hp-0.test0.fits'),
                format='fits', overwrite=True) 
        #mtl = _make_mtl(targets, test=1)
        #mtl.write(os.path.join(dir_dat, 'subpriority_test', 'mtl.dr8.0.34.0.bgs_sv.hp-0.test1.fits'),
        #        format='fits', overwrite=True) 
        return None 
'''

if __name__=="__main__": 
    #targets = fitsio.read('/global/cscratch1/sd/chahah/feasibgs/survey_validation/bright/desitarget-targets-1400deg2.fits') 
    #mtl = make_mtl(targets)
    #mtl.write('/global/cscratch1/sd/chahah/feasibgs/survey_validation/bright/mtl-1400deg2.fits', format='fits') 
    
    #for fmtl in glob.glob(os.path.join(dir_dat, 'mtl.dr8.0.34.0.bgs_sv.hp-*.fits')): 
    #    test_mtl(fmtl)
    # full MTL 
    mtl_SV_healpy()
    #master_truth_table()
