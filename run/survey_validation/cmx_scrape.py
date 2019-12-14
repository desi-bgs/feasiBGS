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

kpno = EarthLocation.of_site('kitt peak')

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


def darksky():
    ''' compile dark sky spectra and compare with dark sky model to approximate
    flux calibrations for the spectrographs 
    '''
    db = DB() 
    ExpInfo = Exposures(db)

    # read in metadata of exposure
    meta = pickle.load(open('desicmx.exp.metadata.p', 'rb'))

    for ispec in range(10): 
        _i = 0 
        waves, fluxes = {}, {} 
    
        # dark sky  
        isdark = (
                (meta['sequence'] == 'Spectrographs') &  
                (meta['moon_alt'] < 0.) & 
                (meta['airmass'] < 1.1)
                )
        nights  = meta['night'][isdark]
        exps    = meta['exp'][isdark]

        for night, exp in zip(nights, exps): 
            try: 
                spectra = get_spectra(meta['night'], exp, type='sky')
            except AssertionError: 
                continue 

            issky = spectra['keep'][ispec]
            if np.sum(issky) == 0: continue  # no sky spectra in this petal 
            print('%i spectra from %i (exp %i)' % (np.sum(issky), meta['night'], exp)) 

            for band in ['b', 'r', 'z']: 
                if _i == 0: 
                    waves[band] = spectra['wave_%s' % band][ispec,issky,:] 
                    fluxes[band] = spectra['flux_%s' % band][ispec,issky,:]
                else: 
                    waves[band] = np.concatenate([waves[band], spectra['wave_%s' % band][ispec,issky,:]]) 
                    fluxes[band] = np.concatenate([fluxes[band], spectra['flux_%s' % band][ispec,issky,:]]) 
            _i += 1 
        if _i == 0: continue 
        
        fig = plt.figure(figsize=(10,5))
        sub = fig.add_subplot(111)
        for ib, band in enumerate(['b', 'r', 'z']): 
            for i in np.arange(waves[band].shape[0])[::10]: 
                sub.plot(waves[band][i,:], fluxes[band][i,:], c='C%i' % ib, alpha=0.1)
        sub.set_xlim(3500, 9800) 
        sub.set_ylim(0, 500) 
        fig.savefig('desicmx.darksky.spectrograph%i.png' % ispec, bbox_inches='tight') 
    return None 


def maybe():
    ''' compile dark sky spectra and compare with dark sky model to approximate
    flux calibrations for the spectrographs 
    '''
    db = DB() 
    ExpInfo = Exposures(db)

    # read in metadata of exposure
    meta = pickle.load(open('desicmx.exp.metadata.p', 'rb'))

    for ispec in range(10): 
        _i = 0 
        waves, fluxes = {}, {} 
    
        # dark sky  
        isdark = (
                (meta['sequence'] == 'Spectrographs') &  
                (meta['moon_alt'] < 0.) & 
                (meta['airmass'] < 1.1)
                )
        nights  = meta['night'][isdark]
        exps    = meta['exp'][isdark]

        for night, exp in zip(nights, exps): 
            try: 
                spectra = get_spectra(meta['night'], exp, type='maybe')
            except AssertionError: 
                continue 

            issky = spectra['keep'][ispec]
            if np.sum(issky) == 0: continue  # no sky spectra in this petal 
            print('%i spectra from %i (exp %i)' % (np.sum(issky), meta['night'], exp)) 

            for band in ['b', 'r', 'z']: 
                if _i == 0: 
                    waves[band] = spectra['wave_%s' % band][ispec,issky,:] 
                    fluxes[band] = spectra['flux_%s' % band][ispec,issky,:]
                else: 
                    waves[band] = np.concatenate([waves[band], spectra['wave_%s' % band][ispec,issky,:]]) 
                    fluxes[band] = np.concatenate([fluxes[band], spectra['flux_%s' % band][ispec,issky,:]]) 
            _i += 1 
        if _i == 0: continue 
        
        fig = plt.figure(figsize=(10,5))
        sub = fig.add_subplot(111)
        for ib, band in enumerate(['b', 'r', 'z']): 
            for i in np.arange(waves[band].shape[0])[::10]: 
                sub.plot(waves[band][i,:], fluxes[band][i,:], c='C%i' % ib, alpha=0.1)
        sub.set_xlim(3500, 9800) 
        sub.set_ylim(0, 500) 
        fig.savefig('desicmx.maybe.spectrograph%i.png' % ispec, bbox_inches='tight') 
    return None 


def get_spectra(night, exp, type='sky'):  
    ''' compile spectra and observing conditions given night and exposure number
    '''
    # read in all the spectra
    spectra = _get_spectra(night, exp) 

    # we assume that most are sky spectra but we remove any outliers 
    keeps = np.zeros((spectra['flux_b'].shape[0], spectra['flux_b'].shape[1])).astype(bool) 
    for ispec in range(10): 
        if (ispec == 3): continue  # remove spectrograph 3, which has the highest chance of observing stars 
        if (spectra['flux_b'][ispec,0,0] == -999.): continue 

        keep = np.ones(spectra['flux_b'].shape[1]).astype(bool)  
        for band in ['b', 'r', 'z']: 
            fluxes = spectra['flux_%s' % band][ispec,:,:] 

            med_flux = np.median(fluxes, axis=1)
            std_flux = np.std(med_flux)
            
            if type == 'sky': 
                clipped = (med_flux < np.median(med_flux) - std_flux) | (med_flux > np.median(med_flux) + std_flux) 
            elif type == 'maybe': 
                clipped = (med_flux < np.median(med_flux) + 2 * std_flux) 
            keep = keep & ~clipped
    
        keeps[ispec,:] = keep

    spectra['keep'] = keeps
    return spectra


def _get_spectra(night, exp):
    ''' given the night and exposure number get wavelength, flux, and spectrograph 
    '''
    dir_spec = '/project/projectdirs/desi/spectro/nightwatch/kpno/' 
    
    spectra = {} 
    for band in ['b', 'r', 'z']: 
        _i = 0 
        for ispec in range(10): # spectrograph number 
            fspec = os.path.join(dir_spec, '%i' % night, '000%i' % exp, 'qframe-%s%i-000%i.fits' % (band, ispec, exp))
            if not os.path.exists(fspec): continue 
            
            flux = fitsio.read(fspec, 'FLUX') 
            wave = fitsio.read(fspec, 'WAVELENGTH') 

            if _i == 0: 
                fluxes  = np.tile(-999., (10, flux.shape[0], flux.shape[1]))
                waves   = np.tile(-999., (10, wave.shape[0], wave.shape[1]))
            fluxes[ispec,:,:] = flux
            waves[ispec,:,:] = wave
            _i += 1 

        assert _i > 0, 'no spectra files'
        spectra['flux_%s' % band] = fluxes
        spectra['wave_%s' % band] = waves 
    return spectra 


def get_obscond(ra, dec, mjd):  
    ''' given RA, Dec, and time of observation compile all observing conditions 
    '''
    #from astropy.utils.iers import IERS_A_URL_MIRROR
    #IERS_A_URL_MIRROR() 

    #from astroplan import download_IERS_A
    #download_IERS_A()

    # target coordinates 
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg) 
    # observed time (UTC)          
    utc_time = Time(mjd, format='mjd')

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

    _i = 0     
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
            print(exp) 
            # append additional observing conditions  
            obscond = get_obscond(meta['skyra'], meta['skydec'], meta['mjd']) 
            for k in obscond.keys(): 
                meta[k] = obscond[k]
            meta['exp']         = exp
            meta['night']       = ExpInfo(exp, 'night')
            meta['mjd']         = ExpInfo(exp, 'mjd_obs')
            meta['sequence']    = ExpInfo(exp, 'sequence')
            if _i == 0: 
                for k in meta.keys(): 
                    metadata[k] = [meta[k]] 
            for k in meta.keys(): 
                metadata[k].append(meta[k]) 
            _i += 1 
    
    for k in metadata.keys(): 
        metadata[k] = np.array(metadata[k])
    pickle.dump(metadata, open('desicmx.exp.metadata.p', 'wb'))
    return None


if __name__=='__main__': 
    make_obscond_table()
