#!/bin/python 
'''
scripts to validate surveysim outputs
'''
import os 
import h5py 
import numpy as np 
import scipy as sp 
import corner as DFM 
import desisurvey.etc as detc
# --- astropy --- 
import astropy.units as u
from astropy.io import fits
from astropy.table import Table as aTable
# -- feasibgs -- 
from feasibgs import util as UT
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
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


dir_dat = os.path.join(UT.dat_dir(), 'survey_validation')


def surveysim_BGS(expfile): 
    ''' read in surveysim output exposures and plot the exposure time distributions 
    for BGS exposures. Also check the BGS exposure time vs exposure properties.
    '''
    # master-branch surveysim output (for comparison) 
    fmaster = os.path.join(dir_dat, 'exposures_surveysim_master.fits')
    exps_master = extractBGS(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractBGS(fexp) # get BGS exposures only 
        
    twilight = (exps['sun_alt'] >= -20.) 

    print('master: %.f < texp < %.f' % (exps_master['texp'].min(), exps_master['texp'].max()))
    print('forked: %.f < texp < %.f' % (exps['texp'].min(), exps['texp'].max()))
    
    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(exps_master['texp'], bins=50, density=True, range=(0, 2500), color='C0', label='master branch')
    sub.hist(exps['texp'], bins=50, density=True, range=(0, 2500), alpha=0.75, color='C1', label=r'surveysim fork')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'$t_{\rm exp}$ (sec) ', fontsize=20) 
    sub.set_xlim(0.,2500) 
    fig.savefig(os.path.join(dir_dat, 'texp.BGS.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    
    # exposure time vs various properties 
    fig = plt.figure(figsize=(15,15)) 
    props   = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep', 'seeing', 'transp']
    lbls    = ['airmass', 'moon ill.', 'moon alt.', 'moon sep.', 'sun alt.', 'sun sep.', 'seeing', 'transp.']
    for i, k in enumerate(props):
        sub = fig.add_subplot(3,3,i+1) 
        sub.scatter(exps['texp'], exps[k], s=1, c='k') 
        # highlight twilight exposures 
        sub.scatter(exps['texp'][twilight], exps[k][twilight], s=2, c='C1', zorder=10) 
        sub.set_xlim(-100., 2500) 
        sub.set_ylabel(lbls[i], fontsize=20) 

    # exposure time vs exposure time correction factor
    from desisurvey import etc as ETC
    exp_factor = np.zeros(len(exps['texp'])) 
    for iexp in range(len(exps['texp'])): 
        exp_factor[iexp] = ETC.bright_exposure_factor(
                exps['moon_ill'][iexp], exps['moon_alt'][iexp], np.array(exps['moon_sep'][iexp]),
                exps['sun_alt'][iexp], np.array(exps['sun_sep'][iexp]), np.array(exps['airmass'][iexp]))
    sub = fig.add_subplot(3,3,i+2) 
    sub.scatter(exps['texp'], exp_factor, s=1, c='k') 
    sub.scatter(exps['texp'][twilight], exp_factor[twilight], s=2, c='C1', zorder=10, label='twilight') 
    sub.set_xlim(-100., 2500) 
    sub.set_ylabel('Bright Exposure Factor', fontsize=20) 
    sub.legend(loc='upper right', fontsize=15, handletextpad=0.2, markerscale=5) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm exp}$ (sec)', fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.3) 
    ffig = os.path.join(dir_dat, 'texp_condition.BGS.%s.png' % expfile.replace('.fits', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def surveysim_All(expfile): 
    ''' read in surveysim output exposures and plot the exposure time distributions 
    '''
    # master-branch surveysim output (for comparison) 
    fmaster = os.path.join(dir_dat, 'exposures_surveysim_master.fits')
    exps_master = extractAll(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractAll(fexp) # get BGS exposures only 
    
    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(exps_master['texp'], bins=50, density=True, range=(0, 2500), color='C0', label='master branch')
    sub.hist(exps['texp'], bins=50, density=True, range=(0, 2500), alpha=0.75, color='C1', label=r'surveysim fork')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'$t_{\rm exp}$ (sec) ', fontsize=20) 
    sub.set_xlim(0.,2500) 
    fig.savefig(os.path.join(dir_dat, 'texp.All.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    return None 


def surveysim_Weird(expfile):
    ''' examine the odd surveysim output exposures 
    '''
    # master-branch surveysim output (for comparison) 
    fmaster = os.path.join(dir_dat, 'exposures_surveysim_master.fits')
    exps_master = extractAll(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(dir_dat, expfile)
    exps = extractAll(fexp) # get BGS exposures only 

    isweird = (exps['texp'] < 300.)
    
    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(exps_master['texp'], bins=50, density=True, range=(-200, 2500), color='C0', label='master branch')
    sub.hist(exps['texp'], bins=50, density=True, range=(-200, 2500), alpha=0.75, color='C1', label=r'surveysim fork')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'$t_{\rm exp}$ (sec) ', fontsize=20) 
    sub.set_xlim(-200.,500) 
    fig.savefig(os.path.join(dir_dat, 'texp.Weird.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    
    # exposure time vs various properties 
    fig = plt.figure(figsize=(15,15)) 
    props   = ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep', 'seeing', 'transp']
    lbls    = ['airmass', 'moon ill.', 'moon alt.', 'moon sep.', 'sun alt.', 'sun sep.', 'seeing', 'transp.']
    for i, k in enumerate(props):
        sub = fig.add_subplot(3,3,i+1) 
        sub.scatter(exps['texp'][isweird], exps[k][isweird], s=1, c='k') 
        sub.set_xlim(-200., 300) 
        sub.set_ylabel(lbls[i], fontsize=20) 
        if k == 'sun_alt': 
            sub.plot([-100., 2500.], [-20., -20.], c='r', ls='--') 
            sub.set_ylim(-25., None) 

    # exposure time vs exposure time correction factor
    from desisurvey import etc as ETC
    exp_factor = np.zeros(len(exps['texp'])) 
    for iexp in np.arange(len(exps['texp']))[isweird]: 
        exp_factor[iexp] = ETC.bright_exposure_factor(
                exps['moon_ill'][iexp], exps['moon_alt'][iexp], np.array(exps['moon_sep'][iexp]),
                exps['sun_alt'][iexp], np.array(exps['sun_sep'][iexp]), np.array(exps['airmass'][iexp]))
    sub = fig.add_subplot(3,3,i+2) 
    sub.scatter(exps['texp'][isweird], exp_factor[isweird], s=1, c='k') 
    sub.set_xlim(-200., 300) 
    sub.set_ylabel('Bright Exposure Factor', fontsize=20) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$t_{\rm exp}$ (sec)', fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.3) 
    ffig = os.path.join(dir_dat, 'texp_condition.Weird.%s.png' % expfile.replace('.fits', ''))
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def extractBGS(fname, notwilight=True): 
    """ extra data on bgs exposures from surveysim output 

    no cosmics split adds 20% to margin
    total BGS time: 2839 hours
    total BGS minus twilight: 2372
    assuming 7.5 deg^2 per field
    assumiong 68% open-dome fraction
    """
    total_hours = 2839
    if notwilight: total_hours = 2372

    open_hours = total_hours*0.68
    
    ssout = fits.open(fname)[1].data # survey sim output 
    tiles = fits.open(os.path.join(UT.dat_dir(), 'bright_exposure', 'desi-tiles.fits'))[1].data # desi-tiles
    
    isbgs = (tiles['PROGRAM'] == 'BRIGHT') # only bgs 
    
    uniq_tiles, iuniq = np.unique(ssout['TILEID'], return_index=True)  
    print('%i unique tiles out of %i total exposures' % (len(uniq_tiles), len(ssout['TILEID'])))

    _, ssbgs, bgsss = np.intersect1d(ssout['TILEID'][iuniq], tiles['TILEID'][isbgs], return_indices=True)  
    print('%i total BGS fields: ' % len(ssbgs))
    print('approx. BGS coverage [#passes]: %f' % (float(len(ssbgs)) * 7.5 / 14000.)) 
    
    RAs, DECs = [], [] 
    airmasses, moon_ills, moon_alts, moon_seps, sun_alts, sun_seps, texps = [], [], [], [], [], [], []
    seeings, transps = [], [] 
    nexps = 0 
    for i in range(len(ssbgs)): 
        isexps  = (ssout['TILEID'] == ssout['TILEID'][iuniq][ssbgs[i]]) 
        nexp    = np.sum(isexps)
        ra      = tiles['RA'][isbgs][bgsss[i]]
        dec     = tiles['DEC'][isbgs][bgsss[i]]
        mjd     = ssout['MJD'][isexps]
        # get sky parameters for given ra, dec, and mjd
        moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = UT.get_thetaSky(np.repeat(ra, nexp), np.repeat(dec, nexp), mjd)

        texps.append(ssout['EXPTIME'][isexps]) 
        RAs.append(np.repeat(ra, nexp)) 
        DECs.append(np.repeat(dec, nexp)) 
        airmasses.append(ssout['AIRMASS'][isexps]) 
        moon_ills.append(moon_ill)
        moon_alts.append(moon_alt) 
        moon_seps.append(moon_sep)
        sun_alts.append(sun_alt) 
        sun_seps.append(sun_sep) 
        
        # atmospheric seeing and transp
        seeings.append(ssout['SEEING'][isexps]) 
        transps.append(ssout['TRANSP'][isexps]) 

        nexps += nexp 

    exps = {
        'texp':     np.concatenate(texps), 
        'ra':       np.concatenate(RAs),
        'dec':      np.concatenate(DECs),
        'airmass':  np.concatenate(airmasses),
        'moon_ill': np.concatenate(moon_ills), 
        'moon_alt': np.concatenate(moon_alts), 
        'moon_sep': np.concatenate(moon_seps), 
        'sun_alt':  np.concatenate(sun_alts), 
        'sun_sep':  np.concatenate(sun_seps),
        'seeing':   np.concatenate(seeings),
        'transp':   np.concatenate(transps)
    }
    return exps 


def extractAll(fname, notwilight=True): 
    """ extra data on all exposures from surveysim output 

    no cosmics split adds 20% to margin
    total BGS time: 2839 hours
    total BGS minus twilight: 2372
    assuming 7.5 deg^2 per field
    assumiong 68% open-dome fraction
    """
    total_hours = 2839
    if notwilight: total_hours = 2372

    open_hours = total_hours*0.68
    
    ssout = fits.open(fname)[1].data # survey sim output 
    tiles = fits.open(os.path.join(UT.dat_dir(), 'bright_exposure', 'desi-tiles.fits'))[1].data # desi-tiles
    
    uniq_tiles, iuniq = np.unique(ssout['TILEID'], return_index=True)  

    _, ssbgs, bgsss = np.intersect1d(ssout['TILEID'][iuniq], tiles['TILEID'], return_indices=True)  
    
    RAs, DECs = [], [] 
    airmasses, moon_ills, moon_alts, moon_seps, sun_alts, sun_seps, texps = [], [], [], [], [], [], []
    seeings, transps = [], [] 
    nexps = 0 
    for i in range(len(ssbgs)): 
        isexps  = (ssout['TILEID'] == ssout['TILEID'][iuniq][ssbgs[i]]) 
        nexp    = np.sum(isexps)
        ra      = tiles['RA'][bgsss[i]]
        dec     = tiles['DEC'][bgsss[i]]
        mjd     = ssout['MJD'][isexps]
        # get sky parameters for given ra, dec, and mjd
        moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = UT.get_thetaSky(np.repeat(ra, nexp), np.repeat(dec, nexp), mjd)

        texps.append(ssout['EXPTIME'][isexps]) 
        RAs.append(np.repeat(ra, nexp)) 
        DECs.append(np.repeat(dec, nexp)) 
        airmasses.append(ssout['AIRMASS'][isexps]) 
        moon_ills.append(moon_ill)
        moon_alts.append(moon_alt) 
        moon_seps.append(moon_sep)
        sun_alts.append(sun_alt) 
        sun_seps.append(sun_sep) 
        
        # atmospheric seeing and transp
        seeings.append(ssout['SEEING'][isexps]) 
        transps.append(ssout['TRANSP'][isexps]) 

        nexps += nexp 

    exps = {
        'texp':     np.concatenate(texps), 
        'ra':       np.concatenate(RAs),
        'dec':      np.concatenate(DECs),
        'airmass':  np.concatenate(airmasses),
        'moon_ill': np.concatenate(moon_ills), 
        'moon_alt': np.concatenate(moon_alts), 
        'moon_sep': np.concatenate(moon_seps), 
        'sun_alt':  np.concatenate(sun_alts), 
        'sun_sep':  np.concatenate(sun_seps),
        'seeing':   np.concatenate(seeings),
        'transp':   np.concatenate(transps)
    }
    return exps 


if __name__=="__main__": 
    surveysim_BGS('exposures_surveysim_fork_150sv0p4.fits') 
    #surveysim_All('exposures_surveysim_fork_150sv0p4.fits') 
    #surveysim_Weird('exposures_surveysim_fork_150sv0p4.fits') 
