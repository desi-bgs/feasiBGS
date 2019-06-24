#!/bin/python 
'''
scripts to check outputs from hacked surveysim  
'''
import os 
import h5py 
import numpy as np 
import scipy as sp 
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
    
    exps = {
        'nexp':     np.zeros(len(ssbgs)).astype(int),
        'texptot':  np.zeros(len(ssbgs)),
        'texps':    np.zeros((len(ssbgs), 25)), 
        'snr2max':  np.zeros(len(ssbgs)),
        'snr2arr':  np.zeros((len(ssbgs), 25)), 
        'ra':       np.zeros(len(ssbgs)), 
        'dec':      np.zeros(len(ssbgs)),
        'mjd':      np.zeros((len(ssbgs), 25)),
        'airmass':  np.zeros((len(ssbgs), 25)),
        'seeing':   np.zeros((len(ssbgs), 25)),
        'transp':   np.zeros((len(ssbgs), 25))
    }

    for i in range(len(ssbgs)): 
        isexps = (ssout['TILEID'] == ssout['TILEID'][iuniq][ssbgs[i]]) 
        nexp = np.sum(isexps)

        exps['nexp'][i]         = nexp
        exps['texps'][i,:nexp]  = ssout['EXPTIME'][isexps] 
        exps['texptot'][i]      = np.sum(ssout['EXPTIME'][isexps]) 
        exps['snr2max'][i]      = np.max(ssout['SNR2FRAC'][isexps]) 
        exps['snr2arr'][i,:nexp]= ssout['SNR2FRAC'][isexps]
        exps['ra'][i]           = tiles['RA'][isbgs][bgsss[i]]
        exps['dec'][i]          = tiles['DEC'][isbgs][bgsss[i]]
        exps['mjd'][i,:nexp]    = ssout['MJD'][isexps]
        exps['airmass'][i,:nexp]= ssout['AIRMASS'][isexps]
        exps['seeing'][i,:nexp] = ssout['SEEING'][isexps]
        exps['transp'][i,:nexp] = ssout['TRANSP'][isexps]
    #print(exps['texptot'].max()) 
    #pickle.dump(exps, open('bgs.%s.p' % os.path.basename(fname).strip('.fits'), 'wb'))
    return exps 


def surveysim_output(expfile): 
    ''' read in surveysim output that Jeremy provided and check the exposures
    with super high exptime. 
    '''
    fmaster = os.path.join(UT.dat_dir(), 'bright_exposure', 'exposures_surveysim_master.fits')
    exps_master = extractBGS(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(UT.dat_dir(), 'bright_exposure', expfile)
    # get BGS exposures only 
    exps = extractBGS(fexp) 
    
    nexp    = exps['nexp']
    ra      = exps['ra']
    dec     = exps['dec']
    mjds    = exps['mjd'] 
    texptot = exps['texptot']  
    texps   = exps['texps']  
    snr2max = exps['snr2max'] 
    snr2arr = exps['snr2arr']
    airmass = exps['airmass'] 
    seeing  = exps['seeing']
    print('')  
    print('tile exposures for the longest time has...')
    print('texp=%f' % (texptot.max()/60.)) 
    print('nexp=%i' % nexp[np.argmax(texptot)]) 
    print('texps=', texps[np.argmax(texptot),:nexp[np.argmax(texptot)]]) 
    print('airmass=', airmass[np.argmax(texptot),:nexp[np.argmax(texptot)]])  
    print('seeing=', seeing[np.argmax(texptot),:nexp[np.argmax(texptot)]])  
    print('')  

    # histogram of total exposures: 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.hist(exps_master['texptot'], bins=100, density=True, range=(0, 10000), color='C0', label='master branch')
    sub.hist(texptot, bins=100, density=True, range=(0, 10000), alpha=0.75, color='C1', label=r'$t_{\rm exp}$ corr. factor')
    sub.legend(loc='upper right', fontsize=20) 
    sub.set_xlabel(r'$t_{\rm exp}$', fontsize=20) 
    sub.set_xlim(0., 10000) 
    fig.savefig(os.path.join(UT.dat_dir(), 'bright_exposure', 'texp.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')

    superhigh = (texptot > 60*60.) # tiles exposed for longer than 60 mins 
    print('%i exposures with very high exposure times' % np.sum(superhigh)) 
    '''
    for i in np.arange(len(texptot))[superhigh]: 
        # get sky parameters for given ra, dec, and mjd
        moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = get_thetaSky(
            np.repeat(ra[i], nexp[i]), 
            np.repeat(dec[i], nexp[i]), 
            mjds[i,:nexp[i]])
        exp_factor = np.array([detc.bright_exposure_factor(moon_ill[_i], moon_alt[_i], np.array([moon_sep[_i]]), 
                                                  np.array([sun_alt[_i]]), np.array([sun_sep[_i]]), np.array([airmass[i,_i]])) 
                      for _i in range(nexp[i])]).flatten() 
        print('---') 
        print(exp_factor)
        print(texps[i,:nexp[i]])
        print(snr2arr[i,:nexp[i]])
    '''
    return None 


def surveysim_convexhull(expfile): 
    ''' read in surveysim output and examine the observing parameters and compare
    it to the convex hull of the GP training set. 
    '''
    fmaster = os.path.join(UT.dat_dir(), 'bright_exposure', 'exposures_surveysim_master.fits')
    exps_master = extractBGS(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(UT.dat_dir(), 'bright_exposure', expfile)
    # get BGS exposures only 
    exps = extractBGS(fexp) 
    
    nexp    = exps['nexp']
    ra      = exps['ra']
    dec     = exps['dec']
    mjds    = exps['mjd'] 
    texptot = exps['texptot']  
    texps   = exps['texps']  
    snr2max = exps['snr2max'] 
    snr2arr = exps['snr2arr']
    airmass = exps['airmass'] 
    seeing  = exps['seeing']

    moon_ill, moon_alt, moon_sep, sun_alt, sun_sep, airmasses = [], [], [], [], [], [] 
    for  _ra, _dec, _texps, _mjds, _airmass in zip(ra, dec, texps, mjds, airmass): 
        hasexp = (_texps > 0.)
        _moon_ill, _moon_alt, _moon_sep, _sun_alt, _sun_sep = get_thetaSky(np.repeat(_ra, np.sum(hasexp)), np.repeat(_dec, np.sum(hasexp)), _mjds[hasexp])
        airmasses.append(_airmass[hasexp]) 
        moon_ill.append(_moon_ill) 
        moon_alt.append(_moon_alt) 
        moon_sep.append(_moon_sep) 
        sun_alt.append(_sun_alt) 
        sun_sep.append(_sun_sep) 

    airmasses  = np.concatenate(airmasses)
    moon_ill    = np.concatenate(moon_ill)
    moon_alt    = np.concatenate(moon_alt)
    moon_sep    = np.concatenate(moon_sep)
    sun_alt     = np.concatenate(sun_alt)
    sun_sep     = np.concatenate(sun_sep)

    print('%f < airmass < %f' % (airmasses.min(), airmasses.max())) 
    print('%f < moonill < %f' % (moon_ill.min(), moon_ill.max())) 
    print('%f < moonalt < %f' % (moon_alt.min(), moon_alt.max())) 
    print('%f < moonsep < %f' % (moon_sep.min(), moon_sep.max())) 
    print('%f < sun_alt < %f' % (sun_alt.min(), sun_alt.max())) 
    print('%f < sun_sep < %f' % (sun_sep.min(), sun_sep.max())) 

    thetas = np.zeros((len(moon_ill), 5))
    thetas[:,0] = moon_ill
    thetas[:,1] = moon_alt
    thetas[:,2] = moon_sep
    thetas[:,3] = sun_alt
    thetas[:,4] = sun_sep

    # read BGS exposures used to train the GP 
    _fexps = h5py.File(''.join([UT.dat_dir(), 'bgs_survey_exposures.withsun.hdf5']), 'r')
    theta_train = np.zeros((len(_fexps['MOONALT'][...]), 5))
    theta_train[:,0] = _fexps['MOONFRAC'][...]
    theta_train[:,1] = _fexps['MOONALT'][...]
    theta_train[:,2] = _fexps['MOONSEP'][...]
    theta_train[:,3] = _fexps['SUNALT'][...]
    theta_train[:,4] = _fexps['SUNSEP'][...]

    theta_hull = sp.spatial.Delaunay(theta_train)
    inhull = (theta_hull.find_simplex(thetas) >= 0) 

    fboss = os.path.join(UT.dat_dir(), 'sky', 'Bright_BOSS_Sky_blue.fits')
    boss = aTable.read(fboss)
    theta_boss = np.zeros((len(boss['MOON_ALT']), 5))
    theta_boss[:,0] = boss['MOON_ILL']
    theta_boss[:,1] = boss['MOON_ALT']
    theta_boss[:,2] = boss['MOON_SEP']
    theta_boss[:,3] = boss['SUN_ALT']
    theta_boss[:,4] = boss['SUN_SEP']
    
    theta_hull_boss = sp.spatial.Delaunay(theta_boss)
    inbosshull = (theta_hull_boss.find_simplex(thetas) >= 0) 

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(131)
    sub.scatter(moon_alt, moon_ill, c='k', s=1, label='SurveySim exp.')
    sub.scatter(moon_alt[inhull], moon_ill[inhull], c='C1', s=1, label='w/in training')
    sub.scatter(moon_alt[inbosshull], moon_ill[inbosshull], c='C0', s=2, label='w/in BOSS skies')
    sub.set_xlabel('Moon Altitude', fontsize=20)
    sub.set_xlim([-90., 90.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0.0, 1.])
    sub.legend(loc='upper left', handletextpad=0, markerscale=10, frameon=True, fontsize=12) 

    sub = fig.add_subplot(132)
    sub.scatter(moon_sep, moon_ill, c='k', s=1)
    sub.scatter(moon_sep[inhull], moon_ill[inhull], c='C1', s=1)
    sub.scatter(moon_sep[inbosshull], moon_ill[inbosshull], c='C0', s=2)
    sub.set_xlabel('Moon Separation', fontsize=20)
    sub.set_xlim([0., 180.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0., 1.])

    sub = fig.add_subplot(133)
    sub.scatter(sun_alt, sun_sep, c='k', s=1)
    sub.scatter(sun_alt[inhull], sun_sep[inhull], c='C1', s=1)
    sub.scatter(sun_alt[inbosshull], sun_sep[inbosshull], c='C0', s=2)
    sub.set_xlabel('Sun Altitude', fontsize=20)
    sub.set_xlim([-90., 0.])
    sub.set_ylabel('Sun Separation', fontsize=20)
    sub.set_ylim([40., 180.])
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(os.path.join(UT.dat_dir(), 'bright_exposure', 'params.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(131)
    sub.scatter(moon_alt, moon_ill, c='k', s=1, zorder=1)
    sub.scatter(theta_train[:,1], theta_train[:,0], c='C1', s=1, zorder=5)
    sub.scatter(theta_boss[:,1], theta_boss[:,0], c='C0', s=2, zorder=10)
    sub.set_xlabel('Moon Altitude', fontsize=20)
    sub.set_xlim([-90., 90.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0.0, 1.])

    sub = fig.add_subplot(132)
    sub.scatter(moon_sep, moon_ill, c='k', s=1, zorder=1)
    sub.scatter(theta_train[:,2], theta_train[:,0], c='C1', s=1, zorder=5)
    sub.scatter(theta_boss[:,2], theta_boss[:,0], c='C0', s=2, zorder=10)
    sub.set_xlabel('Moon Separation', fontsize=20)
    sub.set_xlim([0., 180.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0., 1.])

    sub = fig.add_subplot(133)
    sub.scatter(sun_alt, sun_sep, c='k', s=1, zorder=1)
    sub.scatter(theta_train[:,3], theta_train[:,4], c='C1', s=1, zorder=5)
    sub.scatter(theta_boss[:,3], theta_boss[:,4], c='C0', s=2, zorder=10)
    sub.set_xlabel('Sun Altitude', fontsize=20)
    sub.set_xlim([-90., 0.])
    sub.set_ylabel('Sun Separation', fontsize=20)
    sub.set_ylim([40., 180.])
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(os.path.join(UT.dat_dir(), 'bright_exposure', 'params_overlap.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    return None 


def surveysim_convexhull_exposure_samples(expfile): 
    ''' read in surveysim output and examine the observing parameters and construct
    a sample of exposures that includes the convexhull and a random set of exposures. 
    '''
    from scipy.spatial import ConvexHull
    fmaster = os.path.join(UT.dat_dir(), 'bright_exposure', 'exposures_surveysim_master.fits')
    exps_master = extractBGS(fmaster) 
    # read in exposures output from surveysim 
    print('--- %s ---' % expfile) 
    fexp = os.path.join(UT.dat_dir(), 'bright_exposure', expfile)
    # get BGS exposures only 
    exps = extractBGS(fexp) 
    
    nexp    = exps['nexp']
    ra      = exps['ra']
    dec     = exps['dec']
    mjds    = exps['mjd'] 
    texptot = exps['texptot']  
    texps   = exps['texps']  
    snr2max = exps['snr2max'] 
    snr2arr = exps['snr2arr']
    airmass = exps['airmass'] 
    seeing  = exps['seeing']
    
    # first lets compile the sky parameters of all exposures 
    moon_ill, moon_alt, moon_sep, sun_alt, sun_sep, airmasses = [], [], [], [], [], [] 
    for  _ra, _dec, _texps, _mjds, _airmass in zip(ra, dec, texps, mjds, airmass): 
        hasexp = (_texps > 0.)
        _moon_ill, _moon_alt, _moon_sep, _sun_alt, _sun_sep = get_thetaSky(np.repeat(_ra, np.sum(hasexp)), np.repeat(_dec, np.sum(hasexp)), _mjds[hasexp])
        airmasses.append(_airmass[hasexp]) 
        moon_ill.append(_moon_ill) 
        moon_alt.append(_moon_alt) 
        moon_sep.append(_moon_sep) 
        sun_alt.append(_sun_alt) 
        sun_sep.append(_sun_sep) 

    airmasses   = np.concatenate(airmasses)
    moon_ill    = np.concatenate(moon_ill)
    moon_alt    = np.concatenate(moon_alt)
    moon_sep    = np.concatenate(moon_sep)
    sun_alt     = np.concatenate(sun_alt)
    sun_sep     = np.concatenate(sun_sep)

    params = np.zeros((len(airmasses), 6))
    params[:,0] = airmasses  
    params[:,1] = moon_ill  
    params[:,2] = moon_alt 
    params[:,3] = moon_sep 
    params[:,4] = sun_alt
    params[:,5] = sun_sep 
    hull = ConvexHull(params)
    samples = np.zeros(params.shape[0]).astype(bool) # veritices of the hull 
    samples[hull.vertices] = True
    samples[np.random.choice(np.arange(params.shape[0])[~samples], size=5000-np.sum(samples), replace=False)] = True
    fsamp = os.path.join(UT.dat_dir(), 'bright_exposure', 
            'params.exp_samples.%s.npy' % expfile.replace('.fits', ''))
    np.save(fsamp, params[samples,:]) 

    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(131)
    sub.scatter(moon_alt, moon_ill, c='k', s=1, label='SurveySim exp.')
    sub.scatter(moon_alt[samples], moon_ill[samples], c='C1', s=0.5, label='samples')
    sub.set_xlabel('Moon Altitude', fontsize=20)
    sub.set_xlim([-90., 90.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0.0, 1.])
    sub.legend(loc='upper left', handletextpad=0, markerscale=10, frameon=True, fontsize=12) 

    sub = fig.add_subplot(132)
    sub.scatter(moon_sep, moon_ill, c='k', s=1)
    sub.scatter(moon_sep[samples], moon_ill[samples], c='C1', s=0.5)
    sub.set_xlabel('Moon Separation', fontsize=20)
    sub.set_xlim([0., 180.])
    sub.set_ylabel('Moon Illumination', fontsize=20)
    sub.set_ylim([0., 1.])

    sub = fig.add_subplot(133)
    sub.scatter(sun_alt, sun_sep, c='k', s=1)
    sub.scatter(sun_alt[samples], sun_sep[samples], c='C1', s=0.5)
    sub.set_xlabel('Sun Altitude', fontsize=20)
    sub.set_xlim([-90., 0.])
    sub.set_ylabel('Sun Separation', fontsize=20)
    sub.set_ylim([40., 180.])
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(os.path.join(UT.dat_dir(), 'bright_exposure', 
        'params.exp_samples.%s.png' % expfile.replace('.fits', '')), bbox_inches='tight')
    return None 


def get_thetaSky(ra, dec, mjd): 
    ''' given RA, Dec, and mjd time return sky parameters at kitt peak 
    '''
    import ephem 
    from astropy.time import Time
    import desisurvey.config
    import desisurvey.utils as dutils
    config = desisurvey.config.Configuration()

    mayall = ephem.Observer()
    mayall.lat = config.location.latitude().to(u.rad).value
    mayall.lon = config.location.longitude().to(u.rad).value
    mayall.elevation = config.location.elevation().to(u.m).value
    # Configure atmospheric refraction model for rise/set calculations.
    mayall.pressure = 1e3 * config.location.pressure().to(u.bar).value
    mayall.temp = config.location.temperature().to(u.C).value

    # observed time (MJD) 
    mjd_time = Time(mjd, format='mjd')

    moon_alt    = np.zeros(len(mjd))
    moon_ra     = np.zeros(len(mjd))
    moon_dec    = np.zeros(len(mjd)) 
    moon_ill    = np.zeros(len(mjd))
    sun_alt     = np.zeros(len(mjd))
    sun_ra      = np.zeros(len(mjd))
    sun_dec     = np.zeros(len(mjd)) 
    for i in range(len(mjd)):
        mayall.date = mjd_time.datetime[i] 
        _moon = ephem.Moon()
        _moon.compute(mayall) 
        _sun = ephem.Sun()
        _sun.compute(mayall) 
        
        moon_alt[i] = 180./np.pi*_moon.alt
        moon_ra[i]  = 180./np.pi*_moon.ra
        moon_dec[i] = 180./np.pi*_moon.dec
        moon_ill[i] = _moon.moon_phase
        sun_alt[i] = 180./np.pi*_sun.alt
        sun_ra[i]  = 180./np.pi*_sun.ra
        sun_dec[i] = 180./np.pi*_sun.dec

    moon_sep    = np.diag(dutils.separation_matrix(moon_ra, moon_dec, np.atleast_1d(ra), np.atleast_1d(dec)))
    sun_sep     = np.diag(dutils.separation_matrix(sun_ra, sun_dec, np.atleast_1d(ra), np.atleast_1d(dec)))
    return moon_ill, moon_alt, moon_sep, sun_alt, sun_sep


if __name__=="__main__": 
    #surveysim_convexhull('exposures_surveysim_fork_150s.fits')
    #surveysim_convexhull_exposure_samples('exposures_surveysim_fork_150s.fits') 
    #surveysim_output('exposures_surveysim_fork_corr.fits') 
    #surveysim_output('exposures_surveysim_fork_300s.fits')
    #surveysim_output('exposures_surveysim_fork_200s.fits')
    #surveysim_output('exposures_surveysim_fork_100s.fits')
    #surveysim_output('exposures_surveysim_fork_150s.fits')
    surveysim_output('exposures_surveysim_fork_150sv0p1.fits') 
