'''


https://github.com/desihub/desicmx/blob/master/analysis/specsky/sky-with-moon.ipynb
https://github.com/desihub/desicmx/blob/master/analysis/gfa/GFA-ETC-Pipeline.ipynb
https://github.com/belaa/cmx_sky/blob/master/sky_level_cmx.ipynb


'''
import os
import glob
import fitsio
import numpy as np
from pathlib import Path
# -- desi --
import speclite
import specsim.atmosphere
import specsim.simulator
from desietcimg.db import DB, Exposures
from desietcimg.util import load_raw
# -- astro -- 
import astropy.units as u
from astroplan import download_IERS_A
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon
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


assert os.environ['NERSC_HOST'] == 'cori'
dir_specdata = '/project/projectdirs/desi/spectro/data/'

if not os.path.exists('db.yaml'):
    import getpass
    pw = getpass.getpass(prompt='Enter database password: ')
    with open('db.yaml', 'w') as f:
        print('host: db.replicator.dev-cattle.stable.spin.nersc.org', file=f)
        print('dbname: desi_dev', file=f)
        print('port: 60042', file=f)
        print('user: desi_reader', file=f)
        print(f'password: {pw}', file=f)
    print('Created db.yaml')


def get_skyspectra(night, exp):  
    ''' compile spectra and observing conditions given night and exposure number
    '''
    spectra = get_spectra(night, exp) 

    db = DB() 
    ExpInfo = Exposures(db)
    texp = ExpInfo(exp, 'mjd_obs')
    assert texp is not None, "not much we can do if we don't know hwen it's observed..."  

    # find GFA exposure observed closest to the spectra exposure 
    allexps = db.query("select id,sequence,mjd_obs from exposure").dropna()
    isGFA = (allexps['sequence'].to_numpy() == 'GFA')
    GFAmjd = allexps['mjd_obs'].to_numpy()[isGFA]
    gfa_exp = allexps['id'].to_numpy()[isGFA][np.argmin(np.abs(GFAmjd - ExpInfo(exp, 'mjd_obs')))]

    assert np.abs(texp - ExpInfo(gfa_exp, 'mjd_obs')) < 0.001

    meta = {} 
    for col in ['airmass', 'skyra', 'skydec', 'moonra', 'moondec']: 
        if ExpInfo(exp, col) is not None: 
            meta[col] = ExpInfo(exp, col)
        else: 
            meta[col] = ExpInfo(gfa_exp, col)
    print(meta)      
    return None


def get_spectra(night, exp):
    ''' given the night and exposure number get wavelength, flux, and spectrograph 
    '''
    dir_spec = '/project/projectdirs/desi/spectro/nightwatch/kpno/' 
    
    spectra = {} 
    for band in ['b', 'r', 'z']: 
        for ispec in range(10): # spectrograph number 
            fspec = os.path.join(dir_spec, '%i' % night, '000%i' % exp, 'qframe-%s%i-000%i.fits' % (band, ispec, exp))
            if not os.path.exists(fspec): continue 
            
            flux = fitsio.read(fspec, 'FLUX') 
            wave = fitsio.read(fspec, 'WAVELENGTH') 

            if ispec == 0: 
                fluxes  = [flux] 
                waves   = [wave] 
                ispecs  = [ispec]
            else: 
                fluxes.append(flux) 
                waves.append(wave) 
                ispecs.append(ispec) 

        spectra['flux_%s' % band] = np.array(fluxes) 
        spectra['wave_%s' % band] = np.array(waves) 
        spectra['spectrograph'] = np.array(ispecs) 

    return spectra 


def get_obscond(ra, dec, mjd):  
    ''' given RA, Dec, and time of observation compile all observing conditions 
    '''
    download_IERS_A()
    # target coordinates 
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg) 
    # observed time (UTC)          
    utc_time = Time(mjd)
    kpno = EarthLocation.of_site('kitt peak')

    kpno_altaz = AltAz(obstime=utc_time, location=kpno) 
    coord_altaz = coord.transform_to(kpno_altaz)
    
    obscond = {} 
    obscond['airmass'] = coord_altaz.secz
    obscond['elc_lat'] = coord.barycentrictrueecliptic.lat.deg
    obscond['gal_lat'] = coord.galactic.l.deg   # galactic latitude ( used for ISL contribution ) 
    obscond['gal_lon'] = coord.galactic.b.deg   # galactic longitude ( used for ISL contribution ) 

    tai = utc_time.tai   

    sun = get_sun(utc_time) 
    sun_altaz   = sun.transform_to(kpno_altaz) 
    obscond['sunalt'] = sun_altaz.alt.deg # sun altitude (degrees)
    obscond['sunsep'] = sun.separation(coord).deg # sun separation

    # used for scattered moonlight
    moon = get_moon(utc_time)
    moon_altaz = moon.transform_to(kpno_altaz) 
    obscond['moon_alt'] = moon_altaz.alt.deg 
    obscond['moon_sep'] = moon.separation(coord).deg #coord.separation(self.moon).deg
            
    elongation  = sun.separation(moon)
    phase       = np.arctan2(sun.distance * np.sin(elongation), moon.distance - sun.distance*np.cos(elongation))
    obscond['moon_phase']   = phase.value
    obscond['moon_ill']     = (1. + np.cos(phase))/2.
    return obscond


if __name__=='__main__': 
    gfa_exp = [27337, 27338, 27340, 27341, 27342, 27343, 27344, 27346, 27347, 27348, 27349, 27350,
           27352, 27353, 273524, 27355, 27356, 27358, 27359, 27360, 27361, 27362, 27364, 27365,
           27366, 27367, 27368, 27369, 27371, 27372, 27373, 27374, 27376, 27377, 27378, 27379,
           27380, 27382, 27383, 27384, 27385, 27386, 27388, 27389, 27390, 27391, 27392, 27394,
           27395, 27396]

    spec_exp = [27339, 27345, 27351, 27357, 27363, 27370, 27375, 27381, 27387, 27393]

    night = 20191112
    #get_spectra(night, 27339)
    #gfa_calib = process_darks(night, 27337, 27338)
    get_skyspectra(night, 27339)
