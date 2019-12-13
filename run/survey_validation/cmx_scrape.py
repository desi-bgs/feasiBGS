'''


https://github.com/desihub/desicmx/blob/master/analysis/specsky/sky-with-moon.ipynb
https://github.com/desihub/desicmx/blob/master/analysis/gfa/GFA-ETC-Pipeline.ipynb
https://github.com/belaa/cmx_sky/blob/master/sky_level_cmx.ipynb


'''
import os
import glob
import pickle
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
    db = DB() 
    ExpInfo = Exposures(db)

    # read in metadata of exposure
    metatable = pickle.load(open('desicmx.exp.metadata.p', 'wb'))
    assert exp in metatable.keys() 
    meta = metatable[exp]

    # read in all the spectra
    spectra = get_spectra(night, exp) 
    # we assume that most are sky spectra but we remove any outliers 
    for ispec in range(spectra['flux_b'].shape[0]): 
        keep = np.ones(spectra['flux_b'].shape[1]).astype(bool)  
        for band in ['b', 'r', 'z']: 
            fluxes = spectra['flux_%s' % band][ispec,:,:] 

            med_flux = np.median(fluxes, axis=1)
            std_flux = np.std(med_flux)

            clipped = (med_flux < np.median(med_flux) - 3 * std_flux) | (med_flux > np.median(med_flux) + 3 * std_flux) 
            keep = keep & ~clipped
    
        if ispec == 0: 
            keeps = [keep] 
        elif ispec == 3: 
            # remove spectrograph 3, which has the highest chance of observing stars 
            keeps.append(np.zeros(spectra['flux_b'].shape[1]).astype(bool))
        else: 
            keeps.append(keep) 
    spectra['keep'] = np.array(keeps)
    return spectra, meta 


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
    #from astroplan import download_IERS_A
    #download_IERS_A()

    # target coordinates 
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg) 
    # observed time (UTC)          
    utc_time = Time(mjd, format='mjd')
    kpno = EarthLocation.of_site('kitt peak')

    kpno_altaz = AltAz(obstime=utc_time, location=kpno) 
    coord_altaz = coord.transform_to(kpno_altaz)
    
    obscond = {} 
    obscond['airmass'] = coord_altaz.secz.value
    obscond['elc_lat'] = coord.barycentrictrueecliptic.lat.deg
    obscond['gal_lat'] = coord.galactic.l.deg   # galactic latitude ( used for ISL contribution ) 
    obscond['gal_lon'] = coord.galactic.b.deg   # galactic longitude ( used for ISL contribution ) 

    tai = utc_time.tai   

    # twilight contribution
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
    obscond['moon_ill']     = (1. + np.cos(phase.value))/2.

    return obscond


def make_obscond_table():  
    ''' scape additional observing conditions for all the exposures so far
    '''
    db = DB()
    ExpInfo = Exposures(db)
    # compile all the exposures with GFA or specotrograph exposures 
    allexps = db.query("select id,night,sequence,mjd_obs from exposure").dropna()
    is_gfa          = (allexps['sequence'].to_numpy() == 'GFA')
    is_spec         = (allexps['sequence'].to_numpy() == 'Spectrographs')
    is_gfa_or_spec  = is_gfa | is_spec
    print('%i nights with GFA or spectra' % len(np.unique(allexps['night'].to_numpy().astype(int)[is_gfa_or_spec])))
    print('%i GFA or spectra exposures' % np.sum(is_gfa_or_spec))
    print('%i GFA exposures' % np.sum(is_gfa))
    print('%i spectra exposures' % np.sum(is_spec))

    mjd_gfa = allexps['mjd_obs'].to_numpy()[is_gfa]

    exps = allexps['id'].to_numpy().astype(int)[is_gfa_or_spec]
    
    metadata = {}
    for exp in exps: 
        gfa_exp = None 
        if ExpInfo(exp, 'sequence') == 'spectrographs': 
            # find nearest GFA exposure 
            _gfa_exp = allexps['id'].to_numpy().astype(int)[is_gfa][np.argmin(np.abs(mjd_gfa - ExpInfo(exp, 'mjd_obs')))]
            if np.abs(texp - ExpInfo(_gfa_exp, 'mjd_obs')) < 0.001:  
                gfa_exp = _gfa_exp

        texp = ExpInfo(exp, 'mjd_obs')
        meta = {} 
        meta['mjd'] = texp 
        for col in ['airmass', 'skyra', 'skydec', 'moonra', 'moondec']: 
            if ExpInfo(exp, col) is not None: 
                meta[col] = ExpInfo(exp, col)
            elif (ExpInfo(exp, 'sequence') == 'Spectrographs') and gfa_exp is not None: 
                meta[col] = ExpInfo(gfa_exp, col)
            else: 
                meta[col] = None 

        if meta['skyra'] is not None and meta['skydec'] is not None and meta['mjd'] is not None: 
            # append additional observing conditions  
            #obscond = get_obscond(meta['skyra'], meta['skydec'], meta['mjd']) 
            #for k in obscond.keys(): 
            #    meta[k] = obscond[k]
            meta['night']       = ExpInfo(exp, 'night')
            meta['mjd']         = ExpInfo(exp, 'mjd_obs')
            meta['sequence']    = ExpInfo(exp, 'sequence')
            metadata[exp] = meta 
    
    pickle.dump(metadata, open('desicmx.exp.metadata.p', 'wb'))
    return None


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
    #get_skyspectra(night, 27339)
    make_obscond_table()
