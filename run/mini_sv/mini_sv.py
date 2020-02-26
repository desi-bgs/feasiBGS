#!/bin/python 
''' 
hack on mini SV data 
'''
import os 
import sys 
import glob
import h5py 
import numpy as np 
from astropy.io import fits 
# -- feasibgs -- 
from feasibgs import util as UT
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
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

try: 
    print('running on NERSC %s' % os.environ['NERSC_HOST'])
    local = False
except KeyError: 
    print('running on local machine') 
    local = True 
    mpl.rcParams['text.usetex'] = True

dir_tiles   = '/project/projectdirs/desi/spectro/redux/daily/tiles/'


def match70502(): 
    ''' match tile 70502 to GAMA G12 galaxies  
    '''
    from desitarget.cmx import cmx_targetmask
    from pydl.pydlutils.spheregroup import spherematch
    date        = 20200225
    i_tile      = 70502
    i_coadds    = [0, 3, 6, 7, 9] 
    expids      = [52112, 52113, 52114, 52115, 52116]

    # read GAMA objects 
    gleg = read_gleg() 
    # ra/dec cut to trim to the sample to G12
    cut_radec  = (
            (gleg['legacy.ra'] > 174.0) &
            (gleg['legacy.ra'] < 186.0) &
            (gleg['legacy.dec'] > -3.0) &
            (gleg['legacy.dec'] < 2.0))
    z_g12   = gleg['gama.z'][cut_radec]
    ra_g12  = gleg['gama.ra'][cut_radec]
    dec_g12 = gleg['gama.dec'][cut_radec]
    rmag_g12= UT.flux2mag(gleg['legacy.flux_r'][cut_radec], method='log')
    print('%i GAMA G12 galaxies' % np.sum(cut_radec))

    # get target ids with good fibers
    good_tid = good_fiber(i_tile, date, expids)
    print('%i good fibers in tile' % len(good_tid))
    
    n_good, n_match = 0, 0 
    coadds, zbests, matches, ztrues, r_mags = [], [], [], [], [] 
    for i_coadd in i_coadds: 
        # read coadd 
        coadd = read_coadd(i_tile, date, i_coadd) 
        # add CMX_TARGET flag 
        coadd['IS_MSV_BRIGHT']  = (coadd['cmx_target'] & cmx_targetmask.cmx_mask.mask('MINI_SV_BGS_BRIGHT')) != 0
        coadd['IS_SV0']         = (coadd['cmx_target'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
        # read zbest 
        zbest = read_zbest(i_tile, date, i_coadd) 
        # good fibers  
        isbgs = coadd['IS_MSV_BRIGHT'] | coadd['IS_SV0']
        isgood = np.in1d(coadd['targetid'], good_tid) 
        for k in coadd.keys(): coadd[k] = coadd[k][isgood & isbgs] 
        for k in zbest.keys(): zbest[k] = zbest[k][isgood & isbgs] 
        coadds.append(coadd) 
        zbests.append(zbest) 

        # match RA/Dec 
        match = spherematch(
                ra_g12, dec_g12,
                coadd['target_ra'], coadd['target_dec'],
                0.000277778)
        matches.append(match[1]) # coadd indices with matches 
        ztrues.append(z_g12[match[0]]) 
        r_mags.append(rmag_g12[match[0]])
        n_good += np.sum(isgood) 
        n_match += len(match[0])
    print('%i good fibers on bgs targets in the %i petals' % (n_good, len(i_coadds)))
    print('%i matched to GAMA G12' % n_match) 

    # --- plot RA and Dec --- 
    fig = plt.figure(figsize=(10,5)) 
    sub = fig.add_subplot(111)
    plt0 = sub.scatter(ra_g12, dec_g12, c='C0', s=0.2, label='G12') 
    for coadd in coadds: 
        plt1 = sub.scatter(coadd['target_ra'], coadd['target_dec'], c='k', s=0.5,
                label='Good Fibers') 
    
    n_zsuccess = 0
    zsuccesses = [] 
    for coadd, match, zbest, ztrue in zip(coadds, matches, zbests, ztrues): 
        # get zsuccess rate 
        zsuccess = UT.zsuccess(zbest['z'][match], ztrue, zbest['zwarn'][match],
                min_deltachi2=zbest['deltachi2'][match]) 
        plt2 = sub.scatter(coadd['target_ra'][match][zsuccess],
                coadd['target_dec'][match][zsuccess], 
                c='C1', s=2, label='$z$ success') 
        plt3 = sub.scatter(coadd['target_ra'][match][~zsuccess],
                coadd['target_dec'][match][~zsuccess],
                c='r', s=2, label='$z$ fail') 
        n_zsuccess += np.sum(zsuccess)
        zsuccesses.append(zsuccess)
    print('%i success redshifts' % n_zsuccess)

    sub.legend([plt0, plt1, plt2, plt3], 
            ['G12', 'Good Fibers', '$z$ success', '$z$ fail'], 
            loc='upper right', fontsize=20, markerscale=10, handletextpad=0.2, frameon=True) 
    sub.set_xlabel('RA', fontsize=25) 
    sub.set_xlim(173.0, 187.0) 
    sub.set_ylabel('Dec', fontsize=25) 
    sub.set_ylim(-3.5, 2.5) 
    fig.savefig('tile70502.g12.radec.png', bbox_inches='tight') 
    plt.close() 
    # --- plot z-success(r_mag) --- 
    # calculate z success 
    wmean, rate, err_rate = UT.zsuccess_rate(
            np.concatenate(r_mags),
            np.concatenate(zsuccesses), 
            range=[15, 22], nbins=28, bin_min=10) 
        
    fig = plt.figure(figsize=(10,5)) 
    sub = fig.add_subplot(111)
    sub.plot([15., 22.], [1., 1.], c='k', ls='--', lw=2)
    sub.errorbar(wmean, rate, err_rate, fmt='.C0', elinewidth=2, markersize=10)
    sub.plot(wmean, rate, color='C0')
    sub.set_xlabel(r'$r_{\rm Legacy}$ magnitude', fontsize=25)
    sub.set_ylabel(r'$z$ success rate', fontsize=25)
    sub.text(0.95, 0.95, 'mini SV: tile %i' % i_tile, ha='right', va='top',
            transform=sub.transAxes, fontsize=20)
    sub.set_xlim([17., 20.]) 
    sub.set_ylim([0.6, 1.1])
    sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
    fig.savefig('tile70502.g12.zsuccess.png', bbox_inches='tight') 
    plt.close() 
    return None  


def good_fiber(tile, date, expids): 
    ''' given tile number and list of exposure ids return target ra and dec
    that had good fibers. 
    '''
    from astropy.table import Table, join 
    dir = '/global/cfs/cdirs/desi/spectro/data/%i/' % date
    
    isgood = []
    for i, expid in enumerate(expids):
        # fiberassign 
        fibas = fits.open(os.path.join(dir, str(expid).zfill(8),
            'fiberassign-%s.fits' % (str(tile).zfill(6))))[1].data
        # read in coordinate files with hardware 
        coord = fits.open(os.path.join(dir, str(expid).zfill(8), 
            'coordinates-%s.fits' % (str(expid).zfill(8))))[1].data 
        # combine coordinate files 
        combined = join(Table(coord), Table(fibas), keys=['TARGET_RA', 'TARGET_DEC']) 
        combined.sort('TARGETID') 
        
        if i == 0: target_id = combined['TARGETID']
        else: assert np.array_equal(target_id, combined['TARGETID']) 

        isgood.append(combined['FLAGS_EXP_2'] == 4)  

    return target_id[np.all(np.array(isgood), axis=0)]


def read_gleg(): 
    ''' read GAMA-Legacy matched catalog from Ronpu 

    notes
    -----
    *   http://www.gama-survey.org/dr3/schema/table.php?id=31
    '''
    dir_matched = '/project/projectdirs/desi/target/analysis/truth/dr8.0/south/matched/'
    fgama   = os.path.join(dir_matched, 'GAMA-DR3-SpecObj-match.fits') 
    flegacy = os.path.join(dir_matched, 'ls-dr8.0-GAMA-DR3-SpecObj-match.fits') 
    _gama   = fits.open(fgama)[1].data
    _legacy = fits.open(flegacy)[1].data 
    
    gleg = {} 
    for col in _gama.dtype.names: 
        gleg['gama.%s' % col.lower()] = _gama[col]
    for col in _legacy.dtype.names: 
        gleg['legacy.%s' % col.lower()] = _legacy[col]
    return gleg


def read_zbest(tile, date, i_co): 
    ''' read redrock output for a given coadd  
    '''
    # redrock zbest fits outputs  
    f_zbest = os.path.join(dir_tiles, str(tile), str(date), 
            'zbest-%i-%i-%i.fits' % (i_co, tile, date))
    _zbest = fits.open(f_zbest)[1].data

    zbest = {} 
    for col in _zbest.dtype.names: 
        zbest[col.lower()] = _zbest[col] 
    return zbest


def read_coadd(tile, date, i_co):
    ''' read coadd file given tile #, date string, and coadd #

    :param tile:
        tile number 
    :param date:
        int specifying the date YYYYMMDD (e.g. 20200225 for Feb 25, 2020) 
    :param i_co: 
        co-add number 
    '''
    # coadd file 
    f_coadd = os.path.join(dir_tiles, str(tile), str(date), 
            'coadd-%i-%i-%i.fits' % (i_co, tile, date))
    _coadd= fits.open(f_coadd)[1].data
    
    coadd = {}
    for col in _coadd.dtype.names: 
        coadd[col.lower()] = _coadd[col] 
    return coadd 


if __name__=="__main__": 
    match70502()
