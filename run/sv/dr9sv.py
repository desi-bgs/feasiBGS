'''
script for generating the MTL and fiberassign on DR9SV imaging. 
'''
import os 
import glob
import h5py
import numpy as np 
import numpy.lib.recfunctions as rfn

import fitsio
import healpy as hp 
from astropy.table import Table
from pydl.pydlutils.spheregroup import spherematch
# -- desitarget --
from desitarget.targets import calc_priority, main_cmx_or_sv, set_obsconditions
from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mask, mws_mask
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
if os.environ['NERSC_HOST'] != 'cori': 
    mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


dir_dat = '/global/cscratch1/sd/chahah/feasibgs/survey_validation/'
dir_cfs = '/global/cfs/cdirs/desi/users/chahah/'
if not os.path.isdir(dir_dat): 
    dir_dat = '/Users/ChangHoon/data/feasiBGS/survey_validation/'

f_svfields = 'BGS_SV_30_3x_superset60_Apr2020v2.fits'

######################################################################
# constructing MTLs
######################################################################
def mtl_dr9sv(seed=0, clobber=False): 
    ''' make MTL using DR9SV imaging 
    '''
    np.random.seed(seed)
    #########################################################################
    # compile sv tiles 
    #########################################################################
    # read SV tiles 
    sv = fitsio.read(os.path.join(dir_dat, f_svfields)) # new SV tiles
    print('%i BGS SV tiles' % len(sv['RA']))
    # get SV tiles *outside* of the DR9 SV imaging region 
    in_dr9 = _in_DR9_SVregion(sv['RA'], sv['DEC'])
    print('%i tiles outside of DR9' % np.sum(~in_dr9))
    #########################################################################
    # compile targets and match to truth table and SN host 
    #########################################################################
    # read targets from DR9SV and DR8 cut out 
    ftargets = [
            'sv1-targets-dr9-hp-X.spec_truth.sn_host.fits', 
            'sv1-targets-dr8.sv_cutout.spec_truth.sn_host.fits'
            ]
    ntargets = len(ftargets)

    for i, _ftarget in enumerate(ftargets): 
        ftarget = os.path.join(dir_dat, 'sv.spec_truth', _ftarget)
        if not os.path.isfile(ftarget) or clobber: 
            # read target files with truth tables 
            _f = os.path.join(dir_dat, 'sv.spec_truth', 
                _ftarget.replace('.spec_truth.sn_host.fits', '.spec_truth.fits'))
            __f = os.path.join(dir_dat, 'sv.spec_truth', 
                _ftarget.replace('.spec_truth.sn_host.fits', '.fits'))

            if not os.path.isfile(_f) or clobber:  
                print('... matching %s to truth table' % __f) 
                _target = fitsio.read(os.path.join(dir_dat, __f)) 
                target = match2spectruth(_target)
                fitsio.write(_f, target, clobber=True)
            else: 
                target = fitsio.read(_f)

            print('... matching %s to SN host' % _ftarget) 
            target = match2snhost(target)
            fitsio.write(ftarget, target, clobber=True)
        else: 
            print('... reading %s targets' % ftarget) 
            target = fitsio.read(ftarget)

        # construct MTLs for set of targets 
        mtl = make_mtl(target, seed=seed)
        fmtl = os.path.join(dir_dat, 'mtl',
                'mtl.bgs.dr9sv.%iof%i.seed%i.fits' % (i+1, ntargets, seed))
        mtl.write(fmtl, format='fits', overwrite=True) 
    return None 


def make_mtl(targets, seed=None):
    ''' construct mtl given targets. 

    notes: 
    -----
    * At the moment, highest priority is set for targets with spectroscopic
    redshifts or are SN hosts. 
    '''
    assert 'IN_SPECTRUTH' in targets.dtype.names
    assert 'HAS_SN' in targets.dtype.names

    np.random.seed(seed)
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
    has_spec    = targets['IN_SPECTRUTH'] # objects in spectroscopic truth table  
    has_sn      = targets['HAS_SN']

    # BGS objects with spectra or hosts SN 
    special         = np.zeros(n).astype(bool) #(has_spec | has_sn) 
    bgs_special    =  special & (bitmask_bgs).astype(bool) 

    bgs_all         = ~special & (bitmask_bgs).astype(bool)
    bgs_bright      = ~special & (bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool)
    bgs_faint       = ~special & (bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool)
    bgs_extfaint    = ~special & (bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool) # extended faint
    bgs_fibmag      = ~special & (bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool) # fiber magn limited
    bgs_lowq        = ~special & (bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool) # low quality

    n_bgs           = np.sum(bgs_special) + np.sum(bgs_all) 
    n_bgs_special   = np.sum(bgs_special) 
    n_bgs_bright    = np.sum(bgs_bright)
    n_bgs_faint     = np.sum(bgs_faint)
    n_bgs_extfaint  = np.sum(bgs_extfaint)
    n_bgs_fibmag    = np.sum(bgs_fibmag)
    n_bgs_lowq      = np.sum(bgs_lowq)

    # target classes with spectra
    n_bgs_sp, n_bgs_bright_sp, n_bgs_faint_sp, n_bgs_extfaint_sp, n_bgs_fibmag_sp, n_bgs_lowq_sp = \
            bgs_targetclass(targets['SV1_BGS_TARGET'][special])

    #f_special   = 1. # keep 100%
    #f_bright    = 0.45 / n_bgs_bright
    #f_faint     = 0.25 / n_bgs_faint
    #f_extfaint  = 0.125 / n_bgs_extfaint
    #f_fibmag    = 0.125 / n_bgs_fibmag
    #f_lowq      = 0.05 / n_bgs_lowq
    f_bright    = 540. / (n_bgs_bright + n_bgs_bright_sp)
    f_faint     = 300. / (n_bgs_faint + n_bgs_faint_sp) 
    f_extfaint  = 150. / (n_bgs_extfaint + n_bgs_extfaint_sp)
    f_fibmag    = 150. / (n_bgs_fibmag + n_bgs_fibmag_sp)
    f_lowq      = 60. / (n_bgs_lowq + n_bgs_lowq_sp) 
    f_ref = np.min([f_bright, f_faint, f_extfaint, f_fibmag, f_lowq])

    r_special   = 1.#(1. - f_ref / f_special) 
    r_bright    = (1. - f_ref / f_bright)
    r_faint     = (1. - f_ref / f_faint)   
    r_extfaint  = (1. - f_ref / f_extfaint)
    r_fibmag    = (1. - f_ref / f_fibmag)
    r_lowq      = (1. - f_ref / f_lowq)

    subpriority = np.random.uniform(0., 1., n) 
    subpriority[bgs_special]    = np.random.uniform(r_special, 1., np.sum(bgs_special))
    subpriority[bgs_bright]     = np.random.uniform(r_bright, 1., np.sum(bgs_bright))
    subpriority[bgs_faint]      = np.random.uniform(r_faint, 1., np.sum(bgs_faint))
    subpriority[bgs_extfaint]   = np.random.uniform(f_extfaint, 1, np.sum(bgs_extfaint))
    subpriority[bgs_fibmag]     = np.random.uniform(r_fibmag, 1, np.sum(bgs_fibmag))
    subpriority[bgs_lowq]       = np.random.uniform(r_lowq, 1, np.sum(bgs_lowq))

    _sample = (bitmask_bgs).astype(bool) & (subpriority > 0.943)#np.random.uniform(0., 1., n)) 
    _n_bgs, _n_bgs_bright, _n_bgs_faint, _n_bgs_extfaint, _n_bgs_fibmag, _n_bgs_lowq = \
            bgs_targetclass(targets['SV1_BGS_TARGET'][_sample])

    # set priority of all BGS targets equal 
    priority[bgs_all] = 2000
    print('---------------------------------')
    print('total n_bgs = %i' % n_bgs)
    print('approx. target class fractions') 
    print('                       orig frac exp. frac (target frac)')  
    print('     ------------------------------------')  
    #print('     BGS special         %i %.3f' % (n_bgs_special, n_bgs_special/n_bgs))
    #print('     BGS Bright          %i %.3f (0.45)' % (n_bgs_bright, n_bgs_bright/n_bgs))
    #print('     BGS Faint           %i %.3f (0.25)' % (n_bgs_faint, n_bgs_faint/n_bgs))
    #print('     BGS Ext.Faint       %i %.3f (0.125)' % (n_bgs_extfaint, n_bgs_extfaint/n_bgs))
    #print('     BGS Fib.Mag         %i %.3f (0.125)' % (n_bgs_fibmag, n_bgs_fibmag/n_bgs))
    #print('     BGS Low Q.          %i %.3f (0.05)' % (n_bgs_lowq, n_bgs_lowq/n_bgs))
    print('     BGS Bright        %.3f  %.3f (0.45)' % (n_bgs_bright/n_bgs, _n_bgs_bright/_n_bgs))
    print('     BGS Faint         %.3f  %.3f (0.25)' % (n_bgs_faint/n_bgs, _n_bgs_faint/_n_bgs))
    print('     BGS Ext.Faint     %.3f  %.3f (0.125)' % (n_bgs_extfaint/n_bgs, _n_bgs_extfaint/_n_bgs))
    print('     BGS Fib.Mag       %.3f  %.3f (0.125)' % (n_bgs_fibmag/n_bgs, _n_bgs_fibmag/_n_bgs))
    print('     BGS Low Q.        %.3f  %.3f (0.05)' % (n_bgs_lowq/n_bgs, _n_bgs_lowq/_n_bgs))

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


def match2spectruth(targets): 
    ''' match target table to spectroscopic truth table
    '''
    assert 'BRICKID' in targets.dtype.names
    assert 'BRICK_OBJID' in targets.dtype.names 
    isbgs = (targets['SV1_BGS_TARGET']).astype(bool) 
    targ_brickid    = targets['BRICKID'][isbgs]
    targ_objid      = targets['BRICK_OBJID'][isbgs]

    # read in spectroscopic truth table
    spectruth   = h5py.File(os.path.join(dir_dat, 'bgs_truth_table.hdf5'), 'r') 
    st_brickid  = spectruth['BRICKID'][...]
    st_objid    = spectruth['OBJID'][...]

    in_spectruth = np.zeros(targets.shape[0]).astype(bool)
    gama_cataid  = np.repeat(-999, targets.shape[0]) 
    #in_spectruth.dtype.names = ['ID', 'IN_SPECTRUTH']
    ii = np.arange(targets.shape[0])
    indices, cataid = [], [] 

    uniq_brickid = np.unique(targ_brickid) 
    for brickid in uniq_brickid: 
        in_targbrick = (targ_brickid == brickid)
        in_specbrick = (st_brickid == brickid)
        #in_spec = np.isin(targ_objid[in_targbrick], st_objid[in_specbrick])
        _, in_spec, in_targ = np.intersect1d(targ_objid[in_targbrick], st_objid[in_specbrick], 
                return_indices=True)
        if len(in_spec) > 0: 
            #print(len(in_spec))
            #print(targets['RA'][isbgs][in_targbrick][in_spec] - spectruth['RA'][...][in_specbrick][in_targ]) 
            #print(targets['DEC'][isbgs][in_targbrick][in_spec] - spectruth['DEC'][...][in_specbrick][in_targ])
            #print(spectruth['GAMA_CATAID'][...][in_specbrick][in_targ]) 
            indices.append(ii[isbgs][in_targbrick][in_spec])
            cataid.append(spectruth['GAMA_CATAID'][...][in_specbrick][in_targ]) 

    in_spectruth[np.concatenate(indices)] = True
    gama_cataid[np.concatenate(indices)] = np.concatenate(cataid) 
    print('%i BGS SV targets have spectra' % np.sum(in_spectruth)) 
    targets = rfn.append_fields(targets, ['IN_SPECTRUTH'], [in_spectruth]) 
    targets = rfn.append_fields(targets, ['GAMA_CATAID'], [gama_cataid]) 
    return targets


def match2snhost(targets): 
    ''' match target table to supernovae hosts compiled by Segev
    '''
    assert 'BRICKID' in targets.dtype.names
    assert 'BRICK_OBJID' in targets.dtype.names 
    isbgs = (targets['SV1_BGS_TARGET']).astype(bool) 
    targ_ra     = targets['RA'][isbgs]
    targ_dec    = targets['DEC'][isbgs]

    # read in supernovae hosts
    snhost  = fitsio.read(os.path.join(dir_dat, 'snhost_dr8_target.fits'))
    sn_ra   = snhost['RA']
    sn_dec  = snhost['DEC']

    has_sn = np.zeros(targets.shape[0]).astype(bool)
    # spherematch compiled hosts 
    m_targ, m_sn, d_match = spherematch(targ_ra, targ_dec, sn_ra, sn_dec, 0.000277778, maxmatch=1) 
    has_sn[m_targ] = True
    print('%i BGS SV targets are supernova hosts' % np.sum(has_sn)) 
    targets = rfn.append_fields(targets, ['HAS_SN'], [has_sn]) 
    return targets


def _in_DR9_SVregion(ras, decs): 
    ''' DR9 imaging SV region listed in
    https://desi.lbl.gov/trac/wiki/TargetSelectionWG/SVFields_for_DR9
    '''
    sv_regions = {}
    sv_regions['01_s82']            = [30.,40.,-7.,2.]
    sv_regions['02_egs']            = [210.,220.,50.,55.]
    sv_regions['03_gama09']         = [129.,141.,-2.,3.]
    sv_regions['04_gama12']         = [175.,185.,-3.,2.]
    sv_regions['05_gama15']         = [212.,222.,-2.,3.]
    sv_regions['06_overlap']        = [135.,160.,30.,35.]
    sv_regions['07_refnorth']       = [215.,230.,41.,46.]
    sv_regions['08_ages']           = [215.,220.,30.,40.]
    sv_regions['09_sagittarius']    = [200.,210.,5.,10.]
    sv_regions['10_highebv_n']      = [140.,150.,65.,70.]
    sv_regions['11_highebv_s']      = [240.,245.,20.,25.]
    sv_regions['12_highstardens_n'] = [273.,283.,40.,45.]
    sv_regions['13_highstardens_s'] = [260.,270.,15.,20.]
   
    n_tiles = len(ras)
    in_dr9 = np.zeros(n_tiles).astype(bool) 
    for i, ra, dec in zip(range(n_tiles), ras, decs): 
        for k in sv_regions.keys(): 
            if ((ra >= sv_regions[k][0]) & (ra <= sv_regions[k][1]) & 
                    (dec >= sv_regions[k][2]) & (dec <= sv_regions[k][3])): 
                in_dr9[i] = True
    return in_dr9 


def bgs_targetclass(bitmask_bgs): 
    n_bgs = np.float(np.sum(bitmask_bgs.astype(bool))) 
    n_bgs_bright    = np.sum((bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool))
    n_bgs_faint     = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool))
    n_bgs_extfaint  = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool)) # extended faint
    n_bgs_fibmag    = np.sum((bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool)) # fiber magnitude limited
    n_bgs_lowq      = np.sum((bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool)) # low quality
    return n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq


def check_targets_dr9sv(): 
    '''
    '''
    # read SV tiles 
    sv = fitsio.read(os.path.join(dir_dat, f_svfields)) # new SV tiles
    print('%i BGS SV tiles' % len(sv['RA']))

    ftargets = ['sv1-targets-dr9-hp-X.fits', 'sv1-targets-dr8.sv_cutout.fits']
    ntargets = len(ftargets)

    # plot confirming coverage
    fig = plt.figure(figsize=(10,7))
    sub = fig.add_subplot(111)
    targs = [] 
    for i, _ftarget in enumerate(ftargets): 
        ftarget = os.path.join(dir_dat, 'sv.spec_truth', 
                _ftarget.replace('.fits', '.spec_truth.sn_host.fits'))

        targ = fitsio.read(ftarget)

        sub.scatter(targ['RA'][::100], targ['DEC'][::100], c='k')  
        targs.append(targ)

    for ra, dec in zip(sv['RA'], sv['DEC']): 
        circ = plt.Circle((ra, dec), 1.6275, fill=False, edgecolor='C1',
                linewidth=3)
        sub.add_artist(circ)
    sub.set_xlabel('RA', fontsize=25) 
    sub.set_xlim(0., 360.) 
    sub.set_ylabel('DEC', fontsize=25) 
    sub.set_ylim(-40., 85)
    fig.savefig(os.path.join(dir_dat, 'sv.spec_truth', 
            'check_dr9sv_targets.png'), bbox_inches='tight') 

    # plot confirming coverage tile by tile 
    fig = plt.figure(figsize=(20,12))
    bkgd = fig.add_subplot(111, frameon=False)
    for i, ra, dec in zip(range(len(sv['RA'])), sv['RA'], sv['DEC']): 
        sub = fig.add_subplot(6,10,i+1)
        for targ in targs: 
            sub.scatter(targ['RA'][::100], targ['DEC'][::100], c='k', s=1) 
        circ = plt.Circle((ra, dec), 1.6275, fill=False, edgecolor='C1',
                linewidth=3)
        sub.add_artist(circ)
        sub.set_xlim(ra - 2.5, ra + 2.5) 
        sub.set_ylim(dec - 2.5, dec + 2.5) 

    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'RA', labelpad=7, fontsize=25)
    bkgd.set_ylabel(r'DEC', labelpad=7, fontsize=25)
    fig.savefig(os.path.join(dir_dat, 'sv.spec_truth', 
            'check_dr9sv_targets.tile_by_tile.png'), bbox_inches='tight') 
    return None 


def check_mtl_dr9sv(): 
    ''' check the target fraction of the MTLs  
    '''
    mtls = [] 
    for fmtl in glob.glob(os.path.join(dir_dat, 'mtl', 'mtl*.fits')): 
        print('--- %s ---' % fmtl) 
        # read MTL 
        mtl = fitsio.read(fmtl)

        assigned  = mtl['SUBPRIORITY'] > 0.943

        n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq = \
                bgs_targetclass(mtl['SV1_BGS_TARGET'][assigned])

        print('total n_bgs = %i' % n_bgs)
        print('                       nobj frac (expected frac)')  
        print('     ------------------------------------')  
        print('     BGS Bright          %i %.3f (0.45)' % (n_bgs_bright, n_bgs_bright/n_bgs))
        print('     BGS Faint           %i %.3f (0.25)' % (n_bgs_faint, n_bgs_faint/n_bgs))
        print('     BGS Ext.Faint       %i %.3f (0.125)' % (n_bgs_extfaint, n_bgs_extfaint/n_bgs))
        print('     BGS Fib.Mag         %i %.3f (0.125)' % (n_bgs_fibmag, n_bgs_fibmag/n_bgs))
        print('     BGS Low Q.          %i %.3f (0.05)' % (n_bgs_lowq, n_bgs_lowq/n_bgs))
        mtls.append(mtl) 

    # read SV tiles 
    sv = fitsio.read(os.path.join(dir_dat, f_svfields))

    # plot confirming coverage
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    for mtl in mtls: 
        sub.scatter(mtl['RA'][::100], mtl['DEC'][::100], c='k', s=1) 

    for ra, dec in zip(sv['RA'], sv['DEC']): 
        circ = plt.Circle((ra, dec), 1.6275, fill=False, edgecolor='C1',
                linewidth=3)
        sub.add_artist(circ)
    sub.set_xlabel('RA', fontsize=25) 
    sub.set_xlim(0., 360.) 
    sub.set_ylabel('DEC', fontsize=25) 
    sub.set_ylim(-40., 85)
    fig.savefig(os.path.join(dir_dat, 'mtl', 'mtl_dr9sv_check.png'), 
            bbox_inches='tight') 

    # plot confirming coverage tile by tile 
    fig = plt.figure(figsize=(20,12))
    bkgd = fig.add_subplot(111, frameon=False)
    for i, ra, dec in zip(range(len(sv['RA'])), sv['RA'], sv['DEC']): 
        sub = fig.add_subplot(6,10,i+1)
        for mtl in mtls: 
            sub.scatter(mtl['RA'][::100], mtl['DEC'][::100], c='k', s=1) 
        circ = plt.Circle((ra, dec), 1.6275, fill=False, edgecolor='C1',
                linewidth=3)
        sub.add_artist(circ)
        sub.set_xlim(ra - 2.5, ra + 2.5) 
        sub.set_ylim(dec - 2.5, dec + 2.5) 

    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'RA', labelpad=7, fontsize=25)
    bkgd.set_ylabel(r'DEC', labelpad=7, fontsize=25)
    fig.savefig(os.path.join(dir_dat, 'mtl', 'mtl_dr9sv_check.tile_by_tile.png'), 
            bbox_inches='tight') 
    return None 


def _dr8_target_cutouts(): 
    ''' combine dr8 target files for SV tiles that are outside of the dr9sv
    region. 
    
    * April 9, 2020: Turns out some of the BGS SV fields are chopped up! 
    '''
    # read SV tiles 
    sv = fitsio.read(os.path.join(dir_dat, f_svfields)) # new SV tiles
    print('%i BGS SV tiles' % len(sv['RA']))
    # get SV tiles *outside* of the DR9 SV imaging region 
    in_dr9 = _in_DR9_SVregion(sv['RA'], sv['DEC'])
    print('%i tiles outside of DR9' % np.sum(~in_dr9))

    # for tiles outside of DR9SV read all dr8 healpix that sufficiently covers
    # the tiles 
    ras, decs = [], []
    for ra, dec in zip(sv['RA'][~in_dr9], sv['DEC'][~in_dr9]): 
        corners_ra = [ra - 2., ra + 2., ra + 2., ra - 2.]
        corners_dec = [dec + 2., dec + 2., dec - 2., dec - 2.]
        ras += corners_ra 
        decs += corners_dec

    phi = np.deg2rad(ras) 
    theta = 0.5 * np.pi - np.deg2rad(decs)

    ipixs = np.unique(hp.ang2pix(2, theta, phi, nest=True)) 
    print('     reading in healpixels', ipixs)
    
    targs = [] 
    for i, ipix in enumerate(ipixs): 
        fpix = os.path.join(dir_dat, 'sv.spec_truth', 
                'sv1-targets-dr8-hp-%i.spec_truth.sn_host.fits' % ipix)
        if not os.path.isfile(fpix): 
            ftarg = os.path.join(dir_dat, 'sv1-targets-dr8-hp-%i.fits' % ipix)
            ftrue = fpix.replace('.spec_truth.sn_host.fits', '.spec_truth.fits')

            if not os.path.isfile(ftrue): 
                # read target files with truth tables 
                print('... matching %s to truth table' % ftarg) 
                _targ = fitsio.read(ftarg) 
                targ = match2spectruth(_targ)

                fitsio.write(ftrue, targ) 
            else: 
                targ = fitsio.read(ftrue)

            print('... matching %s to SN host' % os.path.basename(ftrue)) 
            targ = match2snhost(targ)
            fitsio.write(fpix, targ)
        else: 
            print('reading ... %s' % fpix)
            targ = fitsio.read(fpix)

        near_tile = np.zeros(len(targ)).astype(bool) 
        for ra, dec in zip(sv['RA'][~in_dr9], sv['DEC'][~in_dr9]): 
            near_tile |= ((targ['RA'] > ra - 2.) & (targ['RA'] < ra + 2.) & 
                    (targ['DEC'] > dec - 2.) & (targ['DEC'] < dec + 2.))

        assert np.sum(near_tile) > 0
        if i == 0: 
            targs = targ[near_tile]
        else: 
            targs = np.concatenate([targs, targ[near_tile]]) 
    
    fitsio.write(os.path.join(dir_dat, 'sv.spec_truth', 
        'sv1-targets-dr8.sv_cutout.spec_truth.sn_host.fits'), targs,
        clobber=True)
    
    # plot confirming coverage
    fig = plt.figure(figsize=(10,7))
    sub = fig.add_subplot(111)
    sub.scatter(targs['RA'], targs['DEC'], c='k')  
    for ra, dec in zip(sv['RA'][~in_dr9], sv['DEC'][~in_dr9]): 
        circ = plt.Circle((ra, dec), 1.6275, fill=False, edgecolor='C1',
                linewidth=3)
        sub.add_artist(circ)
    sub.set_xlabel('RA', fontsize=25) 
    sub.set_xlim(0., 360.) 
    sub.set_ylabel('DEC', fontsize=25) 
    sub.set_ylim(-40., 85)
    fig.savefig(os.path.join(dir_dat, 'sv.spec_truth', 
            'sv1-targets-dr8.sv_cutout.png'), bbox_inches='tight') 
    return None 


def _dr8_skies_cutouts(): 
    ''' compiled skies file for BGS SV tiles outside of the dr9sv regions 
    '''
    sv = fitsio.read(os.path.join(dir_dat, f_svfields)) # new SV tiles
    in_dr9 = _in_DR9_SVregion(sv['RA'], sv['DEC'])
    print("%i tiles outside of DR9SV" % (np.sum(~in_dr9)))

    # DR8 sky files 
    fskies = glob.glob('/global/cfs/cdirs/desi/target/catalogs/dr8/0.37.0/skies/*fits')

    dr8_skies = [] 
    for fsky in fskies: 
        print('... reading %s' % fsky)
        sky = fitsio.read(fsky)
        
        keep = np.zeros(len(sky['RA'])).astype(bool)
        for tile_ra, tile_dec in zip(sv['RA'][~in_dr9], sv['DEC'][~in_dr9]): 
            keep = keep | (np.sqrt((sky['RA'] - tile_ra)**2 + (sky['DEC'] - tile_dec)**2) < 2.)

            dr8_skies.append(sky[keep])

    dr8_skies = np.concatenate(dr8_skies) 

    # only kep unique TARGTEID
    _, uniq = np.unique(dr8_skies['TARGETID'], return_index=True)

    fitsio.write(os.path.join(dir_dat, 'mtl', 'dr8_skies_cutout.fits'),
            dr8_skies[uniq], clobber=True)
    return None 

######################################################################
# fiberassign 
######################################################################
def run_fiberassign(sky_supp=False): 
    ''' generate script for running fiberassign (for posterity) and run it
    '''
    assert os.environ['NERSC_HOST'] == 'cori'
    
    if not sky_supp: 
        dir_out = '/global/cfs/cdirs/desi/users/chahah/fba_dr9sv.spec_truth.Apr2020'
    else: 
        dir_out = '/global/cfs/cdirs/desi/users/chahah/fba_dr9sv.spec_truth.Apr2020.sky_supp'

    dir_mtl = '/global/cfs/cdirs/desi/users/chahah/mtl_apr2020'

    fmtls = glob.glob(os.path.join(dir_mtl, 'mtl*fits'))
    fskies = glob.glob(os.path.join('/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/', 
        'skies*.fits'))
    fskies += [os.path.join(dir_mtl, 'dr8_skies_cutout.fits')]

    if sky_supp: 
        fsupps = glob.glob(os.path.join(
            '/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-supp', 
            'skies*.fits'))
        fskies += fsupps

    scrpt = '\n'.join([
        '#!/bin/bash', 
        'export DESIMODEL="/global/cscratch1/sd/chahah/feasibgs/desimodel_0.12.0"', 
        '', 
        'odir="%s"' % dir_out, 
        'tfile="%s"' % os.path.join(dir_dat, f_svfields), 
        '', 
        'export OMP_NUM_THREADS=32', 
        '', 
        'mkdir ${odir}', 
        'rm ${odir}/*.fits', 
        '', #'export DESI_LOGLEVEL=DEBUG', 
        'fba_run --targets %s --sky %s --footprint ${tfile} --standards_per_petal 20 --sky_per_petal 80 --write_all_targets --dir ${odir} --overwrite | tee log.o' % (' '.join(fmtls), ' '.join(fskies)),
        '', 
        'fba_merge_results --targets %s --dir ${odir}' % (' '.join(fmtls + fskies))
        ])

    f = open('cori_dr9sv_fba.sh','w')
    f.write(scrpt)
    f.close()
    os.system('sh cori_dr9sv_fba.sh')
    os.system('cp cori_dr9sv_fba.sh %s/cori_dr9sv_fba.sh' % dir_out) 
    return None 


def check_fba(sky_supp=False): 
    ''' test target densities in the fiberassign output of DR9SV 
    '''
    # all the fiberassign output files 
    if not sky_supp: 
        dir_fba = os.path.join(dir_cfs, 'fba_dr9sv.spec_truth.Apr2020')
    else: 
        dir_fba = os.path.join(dir_cfs, 'fba_dr9sv.spec_truth.sky_supp')

    f_fbas = glob.glob(os.path.join(dir_fba, 'fiberassign*.fits'))
    # sarah's fiberassign files 
    #f_fbas = glob.glob(os.path.join(dir_fba, 'fba_dr9sv.sarah', 
    #    'fiberassign*.fits'))

    n_zero = 0
    n_nosky = 0 
    tbl = ['\t'.join(['TILEID', 'SV1_BGS_TARGET', 'BGS_BRIGHT (0.45)', 'BGS_FAINT (0.25)'
        'BGS_FAINT_EXT (0.125)', 'BGS_FIBMAG (0.125)', 'BGS_LOWQ (0.05)',
        'MWS', 'STD', 'SKY', 'BAD', 'BLANK'])]
    __n_bgs_bright, __n_bgs_faint, __n_bgs_extfaint, __n_bgs_fibmag, __n_bgs_lowq, __n_sky = [], [], [], [], [], [] 
    for i, f in enumerate(f_fbas): 
        # read in tile
        tile_i = fitsio.read(f)
        if i == 0: 
            tile = tile_i
        else: 
            tile = np.concatenate([tile, tile_i]) 

        _n_bgs, _n_bgs_bright, _n_bgs_faint, _n_bgs_extfaint, _n_bgs_fibmag, _n_bgs_lowq = \
                bgs_targetclass(tile_i['SV1_BGS_TARGET'])
        _n_mws = np.sum(tile_i['SV1_MWS_TARGET'].astype(bool))
        _n_std = np.sum(~tile_i['SV1_MWS_TARGET'].astype(bool) & (tile_i['SV1_DESI_TARGET'] &
                desi_mask.mask('STD_FAINT|STD_WD|STD_BRIGHT')).astype(bool))
        _n_sky = np.sum(tile_i['OBJTYPE'] == 'SKY')
        _n_bad = np.sum(tile_i['OBJTYPE'] == 'BAD')
        _n_blank = np.sum(tile_i['OBJTYPE'] == '') 

        if _n_bgs + _n_mws + _n_std + _n_sky + _n_bad + _n_blank < 5000: 
            print('--- %s ---' % f.split('-')[-1].split('.')[0]) 
            notdesi = ~(tile_i['SV1_DESI_TARGET'].astype(bool))
            notbgs = ~(tile_i['SV1_BGS_TARGET'].astype(bool))
            notmws = ~(tile_i['SV1_MWS_TARGET'].astype(bool))
            notsky = (tile_i['OBJTYPE'] != 'SKY') 
            notbad = (tile_i['OBJTYPE'] != 'BAD') 
            print(5000 - (_n_bgs + _n_mws + _n_std + _n_sky + _n_bad)) 
            print(np.sum(notdesi & notbgs & notmws))
            print(np.sum(notdesi & notbgs & notmws & notsky & notbad))
            print(tile_i['OBJTYPE'][notdesi & notbgs & notmws & notsky &
                notbad] )

            raise ValueError

        #not_any = ((tile_i['OBJTYPE'] != 'SKY') & 
        #        (tile_i['OBJTYPE'] != 'BAD') & 
        #        (tile_i['SV1_BGS_TARGET'] == 0) & 
        #        (tile_i['SV1_MWS_TARGET'] == 0))

        ##print(tile_i['SV1_DESI_TARGET'][not_any])
        #for i in range(58): 
        #    if np.sum((tile_i['SV1_DESI_TARGET'][not_any] &
        #        desi_mask.mask(i)).astype(bool)) > 0: 
        #        print('%i %s' % 
        #                (np.sum((tile_i['SV1_DESI_TARGET'][not_any] & desi_mask.mask(i)).astype(bool)), 
        #                    desi_mask.bitname(i)))

        __n_bgs_bright.append(_n_bgs_bright/_n_bgs)
        __n_bgs_faint.append(_n_bgs_faint/_n_bgs)
        __n_bgs_extfaint.append(_n_bgs_extfaint/_n_bgs)
        __n_bgs_fibmag.append(_n_bgs_fibmag/_n_bgs)
        __n_bgs_lowq.append(_n_bgs_lowq/_n_bgs)

        print('---------------------------------')
        print('tiles: %s' % os.path.basename(f))
        print('total n_bgs = %i' % _n_bgs)
        print('                       nobj frac (expected frac)')  
        print('     ------------------------------------')  
        print('     BGS Bright          %i %.3f (0.45)' % (_n_bgs_bright, _n_bgs_bright/_n_bgs))
        print('     BGS Faint           %i %.3f (0.25)' % (_n_bgs_faint, _n_bgs_faint/_n_bgs))
        print('     BGS Ext.Faint       %i %.3f (0.125)' % (_n_bgs_extfaint, _n_bgs_extfaint/_n_bgs))
        print('     BGS Fib.Mag         %i %.3f (0.125)' % (_n_bgs_fibmag, _n_bgs_fibmag/_n_bgs))
        print('     BGS Low Q.          %i %.3f (0.05)' % (_n_bgs_lowq, _n_bgs_lowq/_n_bgs))
        print('     SKY                 %i' % _n_sky) 
        print('     BAD                 %i' % _n_bad)
        __n_sky.append(_n_sky) 

        tbl.append('\t'.join([
            '%s' % f.split('-')[-1].split('.')[0], 
            '%i' % _n_bgs, 
            '%i (%.3f)' % (_n_bgs_bright, _n_bgs_bright/_n_bgs),
            '%i (%.3f)' % (_n_bgs_faint, _n_bgs_faint/_n_bgs),
            '%i (%.3f)' % (_n_bgs_extfaint, _n_bgs_extfaint/_n_bgs),
            '%i (%.3f)' % (_n_bgs_fibmag, _n_bgs_fibmag/_n_bgs),
            '%i (%.3f)' % (_n_bgs_lowq, _n_bgs_lowq/_n_bgs),
            '%i' % _n_mws,
            '%i' % _n_std,
            '%i' % _n_sky, 
            '%i' % _n_bad, 
            '%i' % _n_blank]))
        
        # tiles with no sky targets 
        if _n_sky == 0: n_nosky += 1
        # tiles with no BGS targets 
        if _n_bgs == 0: n_zero += 1
    print('---------------------------------')
    print('%i tiles with zero BGS targets' % n_zero)
    print('%i tiles with zero SKY targets' % n_nosky)

    n_bgs, n_bgs_bright, n_bgs_faint, n_bgs_extfaint, n_bgs_fibmag, n_bgs_lowq = \
            bgs_targetclass(tile['SV1_BGS_TARGET'])
    n_mws = np.sum(tile['SV1_MWS_TARGET'].astype(bool))
    n_std = np.sum((tile['SV1_DESI_TARGET'] &
            desi_mask.mask('STD_FAINT|STD_WD|STD_BRIGHT')).astype(bool))
    n_sky = np.sum(tile['OBJTYPE'] == 'SKY')
    n_bad = np.sum(tile['OBJTYPE'] == 'BAD')
    n_blank = np.sum(tile['OBJTYPE'] == '')
    print('---------------------------------')
    print('total n_bgs = %i' % n_bgs)
    print('total n_bgs = %i' % np.sum(tile['SV1_BGS_TARGET'] != 0))
    print('                       nobj frac (expected frac)')  
    print('     ------------------------------------')  
    print('     BGS Bright          %i %.3f, %.3f-%.3f (0.45)' % (n_bgs_bright, n_bgs_bright/n_bgs, np.min(__n_bgs_bright), np.max(__n_bgs_bright)))
    print('     BGS Faint           %i %.3f, %.3f-%.3f (0.25)' % (n_bgs_faint, n_bgs_faint/n_bgs, np.min(__n_bgs_faint), np.max(__n_bgs_faint)))
    print('     BGS Ext.Faint       %i %.3f, %.3f-%.3f (0.125)' % (n_bgs_extfaint, n_bgs_extfaint/n_bgs, np.min(__n_bgs_extfaint), np.max(__n_bgs_extfaint)))
    print('     BGS Fib.Mag         %i %.3f, %.3f-%.3f (0.125)' % (n_bgs_fibmag, n_bgs_fibmag/n_bgs, np.min(__n_bgs_fibmag), np.max(__n_bgs_fibmag)))
    print('     BGS Low Q.          %i %.3f, %.3f-%.3f (0.05)' % (n_bgs_lowq, n_bgs_lowq/n_bgs, np.min(__n_bgs_lowq), np.max(__n_bgs_lowq)))
    print('     SKY                 %i' % (np.mean(__n_sky)))
    tbl.append('---------------------------------') 
    print('     BGS Bright          %i' % np.sum((tile['SV1_BGS_TARGET'] & 2**0) != 0))
    print('     BGS Faint           %i' % np.sum((tile['SV1_BGS_TARGET'] & 2**1) != 0))
    print('     BGS Ext.Faint       %i' % np.sum((tile['SV1_BGS_TARGET'] & 2**2) != 0))
    print('     BGS Fib.Mag         %i' % np.sum((tile['SV1_BGS_TARGET'] & 2**4) != 0))
    print('     BGS Low Q.          %i' % np.sum((tile['SV1_BGS_TARGET'] & 2**3) != 0))


    tbl.append('\t'.join([
        'TOTAL', 
        '%i' % n_bgs, 
        '%i (%.3f)' % (n_bgs_bright, n_bgs_bright/n_bgs),
        '%i (%.3f)' % (n_bgs_faint, n_bgs_faint/n_bgs),
        '%i (%.3f)' % (n_bgs_extfaint, n_bgs_extfaint/n_bgs),
        '%i (%.3f)' % (n_bgs_fibmag, n_bgs_fibmag/n_bgs),
        '%i (%.3f)' % (n_bgs_lowq, n_bgs_lowq/n_bgs),
        '%i' % n_mws,
        '%i' % n_std,
        '%i' % n_sky,
        '%i' % n_bad, 
        '%i' % n_blank]))

    ftbl = os.path.join(dir_fba, 'fba_table.dat')
    f = open(ftbl,'w')
    f.write('\n'.join(tbl))
    f.close() 

    #fig = plt.figure(figsize=(10,5))
    #sub = fig.add_subplot(111)
    #sub.scatter(tile['TARGET_RA'], tile['TARGET_DEC'], c='k', s=2)
    #sub.scatter(tile['TARGET_RA'][_flags], tile['TARGET_DEC'][_flags], c='C1', s=2)

    ##sub.legend(loc='upper right', handletextpad=0.2, markerscale=5, fontsize=15) 
    #sub.set_xlabel('RA', fontsize=20)
    #sub.set_xlim(360., 0.)#tile['TARGET_RA'].min(), tile['TARGET_RA'].max())
    #sub.set_ylabel('Dec', fontsize=20)
    ##sub.set_ylim(22., 26) 
    #sub.set_ylim(-30., 80.)#tile['TARGET_DEC'].min(), tile['TARGET_DEC'].max())
    #fig.savefig(os.path.join(dir_dat, 'fba_dr9sv.spec_truth.Apr2020', 'fiberassign_outliers.png'), bbox_inches='tight') 

    # plot targets tile by tile by tile 
    fig = plt.figure(figsize=(18,14))
    bkgd = fig.add_subplot(111, frameon=False)
    
    for i, f in enumerate(f_fbas): 
        # read in tile
        tile_i = fitsio.read(f)
        print('--------') 
        print(f)

        bitmask_bgs = tile_i['SV1_BGS_TARGET']

        bgs             = bitmask_bgs.astype(bool)
        bgs_bright      = (bitmask_bgs & bgs_mask.mask('BGS_BRIGHT')).astype(bool)
        bgs_faint       = (bitmask_bgs & bgs_mask.mask('BGS_FAINT')).astype(bool)
        bgs_extfaint    = (bitmask_bgs & bgs_mask.mask('BGS_FAINT_EXT')).astype(bool) # extended faint
        bgs_fibmag      = (bitmask_bgs & bgs_mask.mask('BGS_FIBMAG')).astype(bool) # fiber magnitude limited
        bgs_lowq        = (bitmask_bgs & bgs_mask.mask('BGS_LOWQ')).astype(bool) # low quality
        sky             = (tile_i['OBJTYPE'] == 'SKY')
        unassigned      = (tile_i['FA_TARGET'] == 0)

        print('%i sky fibers' % np.sum(sky))
        print('%i unassigned fibers' % np.sum(unassigned)) 
        
        ra, dec = tile_i['TARGET_RA'], tile_i['TARGET_DEC']
        # plot tile 
        sub = fig.add_subplot(7,9,i+1)
        sub.scatter(ra, dec, c='k', s=0.1)
        sub.scatter(ra[bgs_bright], dec[bgs_bright], c='C1', s=0.2, 
                label='BGS BRIGHT') 
        sub.scatter(ra[bgs_faint], dec[bgs_faint], c='C2', s=0.2, 
                label='BGS FAINT') 
        sub.scatter(ra[bgs_extfaint], dec[bgs_extfaint], c='C4', s=0.2, 
                label='BGS EXTFAINT') 
        sub.scatter(ra[bgs_fibmag], dec[bgs_fibmag], c='C5', s=0.2, 
                label='BGS FIBMAG') 
        sub.scatter(ra[bgs_lowq], dec[bgs_lowq], c='C6', s=0.2, 
                label='BGS LOWQ') 
        sub.scatter(ra[sky], dec[sky], c='C0', s=0.2, label='Sky')
        sub.scatter(ra[unassigned], dec[unassigned], c='r', s=0.1,
                label='unassigned') 

        sub.set_xlim(ra.min()-0.2, ra.max()+0.2)
        sub.set_ylim(dec.min()-0.2, dec.max()+0.2)
    sub = fig.add_subplot(7,9,i+2)
    sub.scatter([0], [0], c='k', s=1)
    sub.scatter([0], [0], c='C1', s=2, label='BGS BRIGHT') 
    sub.scatter([0], [0], c='C2', s=2, label='BGS FAINT') 
    sub.scatter([0], [0], c='C3', s=2, label='BGS EXTFAINT') 
    sub.scatter([0], [0], c='C4', s=2, label='BGS FIBMAG') 
    sub.scatter([0], [0], c='C5', s=2, label='BGS LOWQ') 
    sub.scatter([0], [0], c='C0', s=2, label='Sky')
    sub.scatter([0], [0], c='r', s=1, label='unassigned') 
    sub.set_xticklabels([])
    sub.set_yticklabels([])
    sub.legend(loc='upper left', markerscale=5, handletextpad=0., fontsize=7)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'RA', labelpad=7, fontsize=25)
    bkgd.set_ylabel(r'DEC', labelpad=7, fontsize=25)
    fig.savefig(os.path.join(dir_fba, 
        'check_fba.tile_by_tile.png'), bbox_inches='tight') 
    return None 


def check_fba_skies(): 
    ''' check that the fiberassign outputs 
    '''
    # all the fiberassign output files 
    dir_fba = '/global/cfs/cdirs/desi/users/chahah/'
    f_fbas = glob.glob(os.path.join(dir_fba, 'fba_dr9sv.spec_truth.Apr2020',
        'fiberassign*.fits'))
    # all the sky targets
    f_skies = [ 
            "/global/cfs/cdirs/desi/target/catalogs/dr8/0.37.0/skies/skies-dr8-hp-0.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr8/0.37.0/skies/skies-dr8-hp-15.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr8/0.37.0/skies/skies-dr8-hp-19.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr8/0.37.0/skies/skies-dr8-hp-47.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-4.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-5.fits", 
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-7.fits", 
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-8.fits", 
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-9.fits", 
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-10.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-11.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-14.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-17.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-21.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-24.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-25.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-26.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-27.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-31.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-35.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-39.fits",
            "/global/cfs/cdirs/desi/target/catalogs/dr9sv/0.37.0/skies/skies-dr9-hp-43.fits"]

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    for f in f_skies: 
        sky = fitsio.read(f) 
        _plt_sky = sub.scatter(sky['RA'][::100], sky['DEC'][::100], c='k', s=1)

    dr8sky = fitsio.read(os.path.join(dir_dat, 'mtl', 'dr8_skies_cutout.fits'))
    _plt_sky2 = sub.scatter(dr8sky['RA'][::100], dr8sky['DEC'][::100], c='C1', s=1)

    for f in f_fbas: 
        # read in tile
        tile_i = fitsio.read(f)
        _n_sky = np.sum(tile_i['OBJTYPE'] == 'SKY')

        if _n_sky == 0: 
            _plt_nosky = sub.scatter(tile_i['TARGET_RA'][::10], tile_i['TARGET_DEC'][::10],
                    c='r', s=1, label='No Sky Fibers')
        else: 
            _plt_bgs = sub.scatter(tile_i['TARGET_RA'][::10], tile_i['TARGET_DEC'][::10],
                    c='C0', s=1, label='BGS SV tile')

    sub.legend([_plt_sky, _plt_sky2, _plt_nosky, _plt_bgs], 
            ['Sky Targets', 'DR8 cut out', 'No Sky Fibers', 'BGS SV tiles'], 
            loc='upper right', handletextpad=0.2, markerscale=5, fontsize=15) 
    sub.set_xlabel('RA', fontsize=20)
    sub.set_xlim(360., 0.)#tile['TARGET_RA'].min(), tile['TARGET_RA'].max())
    sub.set_ylabel('Dec', fontsize=20)
    sub.set_ylim(-40., 90.)#tile['TARGET_DEC'].min(), tile['TARGET_DEC'].max())
    fig.savefig('fba_dr9sv_nosky.png', bbox_inches='tight') 
    return None 

######################################################################
# examine tiles  
######################################################################
def low_BGS_target(sky_supp=False): 
    ''' examinie tiles with unusually low BGS targets 
    '''
    # compile BGS tagets from MTL 
    for i, fmtl in enumerate(glob.glob(os.path.join(dir_cfs, 'mtl_apr2020',
        'mtl.bgs.dr9sv*.fits'))): 
        mtl = fitsio.read(fmtl)

        isbgs = mtl['SV1_BGS_TARGET'].astype(bool) 

        if i == 0: 
            mtl_ra = mtl['RA'][isbgs]
            mtl_dec = mtl['DEC'][isbgs]
        else: 
            mtl_ra = np.concatenate([mtl_ra, mtl['RA'][isbgs]]) 
            mtl_dec = np.concatenate([mtl_dec, mtl['DEC'][isbgs]]) 

    # all the fiberassign output files 
    if not sky_supp: 
        dir_fba = os.path.join(dir_cfs, 'fba_dr9sv.spec_truth.Apr2020')
    else: 
        dir_fba = os.path.join(dir_cfs, 'fba_dr9sv.spec_truth.sky_supp')
    f_fbas = glob.glob(os.path.join(dir_fba, 'fiberassign*.fits'))

    for i, f in enumerate(f_fbas): 
        # read in tile
        tile_i = fitsio.read(f)
    
        isbgs = tile_i['SV1_BGS_TARGET'].astype(bool) 
        if np.sum(isbgs) > 2000: continue 
        
        ra_min, ra_max = tile_i['TARGET_RA'].min(), tile_i['TARGET_RA'].max()  
        dec_min, dec_max = tile_i['TARGET_DEC'].min(), tile_i['TARGET_DEC'].max()  

        near_tile = ((mtl_ra > ra_min) & (mtl_ra < ra_max) & 
                (mtl_dec > dec_min) & (mtl_dec < dec_max))

        fig = plt.figure(figsize=(6,6)) 
        sub = fig.add_subplot(111)

        # plot MTL 
        sub.scatter(mtl_ra[near_tile], mtl_dec[near_tile], c='k', s=1) 
        # plot assigned targets
        sub.scatter(tile_i['TARGET_RA'][isbgs], tile_i['TARGET_DEC'][isbgs],
                c='C1', s=4) 

        sub.set_xlabel("RA", fontsize=20) 
        sub.set_xlim(ra_min-0.5, ra_max+0.5)
        sub.set_ylabel("Dec", fontsize=20) 
        sub.set_ylim(dec_min-0.5, dec_max+0.5)
        ffig = os.path.join(dir_cfs, 'fba_dr9sv.spec_truth.Apr2020',
                'tile%s.mtl_compare.png' % f.split('-')[-1].split('.')[0])
        fig.savefig(ffig, bbox_inches='tight')
    return None 



if __name__=="__main__": 
    # construct MTL
    #_dr8_target_cutouts()
    #check_targets_dr9sv() 
    #mtl_dr9sv(seed=0)#, clobber=True)
    #check_mtl_dr9sv()

    #_dr8_skies_cutouts()
    #run_fiberassign()
    #run_fiberassign(sky_supp=True)
    check_fba(sky_supp=False)
    #check_fba(sky_supp=True)
    #low_BGS_target(sky_supp=False)
