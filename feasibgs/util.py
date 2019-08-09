'''

utility functions 

'''
import os
import sys
import subprocess
import numpy as np


def check_env(): 
    if os.environ.get('FEASIBGS_DIR') is None: 
        raise ValueError("set $FEASIBGS_DIR in bashrc file!") 
    return None


def flux2mag(flux, band='g', method='asinh'): 
    ''' given flux calculate SDSS asinh magnitude  
    '''
    if method == 'asinh': 
        if band == 'u': b = 1.4e-10
        elif band == 'g': b = 0.9e-10
        elif band == 'r': b = 1.2e-10
        elif band == 'i': b = 1.8e-10
        elif band == 'z': b = 7.4e-10
        else: raise ValueError
        
        return -2.5/np.log(10) * (np.arcsinh(1.e-9*flux/(2.*b)) + np.log(b))
    elif method == 'log': 
        return 22.5 - 2.5 * np.log10(flux) 


def mag2flux(mag, band='g', method='asinh'): 
    ''' given flux calculate SDSS asinh magnitude  
    '''
    if method == 'asinh': 
        #mag = -2.5/np.log(10) * (np.arcsinh(1.e-9*flux/(2.*b)) + np.log(b))
        if band == 'u': b = 1.4e-10
        elif band == 'g': b = 0.9e-10
        elif band == 'r': b = 1.2e-10
        elif band == 'i': b = 1.8e-10
        elif band == 'z': b = 7.4e-10
        else: raise ValueError
        return np.sinh(mag/-2.5*np.log(10) - np.log(b)) * 2.* b * 1e9 

    elif method == 'log': 
        return 10**((22.5 - mag)/2.5) 


def dat_dir(): 
    ''' 
    '''
    return os.environ.get('FEASIBGS_DIR') 


def code_dir(): 
    return os.environ.get('FEASIBGS_CODEDIR') 


def fig_dir(): 
    ''' directory for figures 
    '''
    if os.environ.get('FEASIBGS_FIGDIR') is None: 
        return dat_dir()+'/figs/'
    else: 
        return os.environ.get('FEASIBGS_FIGDIR')


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    return fig_dir().split('fig')[0]+'doc/'


def paper_dir(): 
    ''' directory for paper related stuff 
    '''
    return fig_dir().split('fig')[0]+'paper/'


def nersc_submit_job(fjob): 
    ''' submit slurm job to nersc 
    '''
    if not os.path.isfile(fjob): raise ValueError("%s not found" % fjob) 
    subprocess.check_output(['sbatch', fjob])
    return None 


def get_thetaSky(ra, dec, mjd): 
    ''' given RA, Dec, and mjd time return sky parameters at kitt peak 
    '''
    import ephem 
    import astropy.units as u
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


def zeff_hist(prop, ztrue, zest, range=None, threshold=0.003, nbins=20, bin_min=2): 
    ''' histogram looking at dependence of redshift efficiency on `prop`. redshift 
    is considered "successful" if dz/(1+ztrue) < threshold. This is taken from 
    one of John's ipython notebooks. 
    
    parameters 
    ----------
    prop : (numpy.ndarray)
        Some property (e.g. r-band aperture flux magnitude) 

    ztrue : (numpy.ndarray)
        True (input) redshift  
    
    zest : (numpy.ndarray)
        redshift estimate (e.g. redrock output) 
    
    range : (optional, tuple) 
        range of `prop`
    
    threshold : (float) 
        threshold for determining redshift success

    nbins : (int) 
        number of bins to evaluat ethe histogram

    bin_min : (int) 
        minimum number of galaxies that need to be in a `prop` bin 
        to be included 

    '''
    if not len(prop) == len(ztrue): raise ValueError("prop, ztrue, and zbest must have the same dimensions")
    if not len(ztrue) == len(zest): raise ValueError("prop, ztrue, and zbest must have the same dimensions")
    dz = zest - ztrue # delta z 
    dz_1pz = np.abs(dz)/(1.+ztrue)
    s1 = (dz_1pz < threshold)

    h0, bins = np.histogram(prop, bins=nbins, range=range)
    hv, _ = np.histogram(prop, bins=bins, weights=prop)
    h1, _ = np.histogram(prop[s1], bins=bins)

    good = h0 > bin_min 
    hv = hv[good]
    h0 = h0[good]
    h1 = h1[good]

    vv = hv / h0 # weighted mean of prop 

    def _eff(k, n):
        eff = k.astype("float") / (n.astype('float') + (n==0))
        efferr = np.sqrt(eff * (1 - eff)) / np.sqrt(n.astype('float') + (n == 0))
        return eff, efferr

    e1, ee1 = _eff(h1, h0)
    return vv, e1, ee1
