'''


https://github.com/desihub/desicmx/blob/master/analysis/specsky/sky-with-moon.ipynb
https://github.com/desihub/desicmx/blob/master/analysis/gfa/GFA-ETC-Pipeline.ipynb
https://github.com/belaa/cmx_sky/blob/master/sky_level_cmx.ipynb

see also 
https://docs.google.com/document/d/1DwwubRieoaRA8YBhVxE20QwE0EN5xkU-VZQ5tXZEFAE/edit?usp=sharing


'''
import os
import glob
import pickle
import fitsio
import numpy as np
# -- desi --
import desisim.simexp 
import specsim.atmosphere
import specsim.simulator
from desietcimg.db import DB, Exposures
from desietcimg.util import load_raw
# -- feasibgs --
from feasibgs import util as UT
from feasibgs import skymodel as Sky
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


def brightsky(dark_night=20191206, dark_exp=30948): 
    '''
    '''
    # read in metadata of exposures 
    meta = pickle.load(open('desicmx.exp.metadata.p', 'rb'))
    # select exposures during bright time  
    isbright = (
            (meta['sequence'] == 'Spectrographs') &  
            (meta['moon_alt'] > 0.) & 
            (meta['exptime'] > 10.) 
            )
    print('%i exposures during bright time' % np.sum(isbright))

    fexp_cmx, fexp_model, sig_fexp_cmx = [], [], [] 
    for iexp in np.arange(len(meta['id']))[isbright]: 
        if 'CALIB' in meta['program'][iexp]: continue 
        try: 
            fexptime, sig_fexp = fexptime_CMXspec(meta['night'][iexp], meta['id'][iexp], dark_night=dark_night, dark_exp=dark_exp)
        except AssertionError:
            continue 
        fexptime_m = fexptime_model(meta['_airmass'][iexp], 
                meta['moon_ill'][iexp], meta['moon_alt'][iexp], meta['moon_sep'][iexp],
                meta['sunalt'][iexp], meta['sunsep'][iexp])
        print('--- %s ---' % meta['program'][iexp]) 
        print('%i exposure %i texp=%.2f' % (meta['night'][iexp], meta['id'][iexp], meta['exptime'][iexp]))
        print('airmass=%.2f' % meta['_airmass'][iexp])
        print('moon: ill=%.2f, alt=%.1f, sep=%.1f' % (meta['moon_ill'][iexp], meta['moon_alt'][iexp], meta['moon_sep'][iexp])) 
        print('sun: alt=%.1f, sep=%.1f' % (meta['sunalt'][iexp], meta['sunsep'][iexp]))
        print('cmx exptime factor=%.2f pm %.2f' % (fexptime, sig_fexp))
        print('model exptime factor=%.2f' % fexptime_m) 
        fexp_cmx.append(fexptime)
        fexp_model.append(fexptime_m)
        sig_fexp_cmx.append(sig_fexp) 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.errorbar(fexp_cmx, fexp_model, xerr=sig_fexp_cmx, fmt='.C0')
    sub.plot([1., 10.], [1., 10.], c='k', ls='--') 
    sub.set_xlabel(r'$f_{\rm exp}$ from CMX sky flux', fontsize=20) 
    sub.set_xlim(1., 10.) 
    sub.set_ylabel(r'$f_{\rm exp}$ from updated model', fontsize=20) 
    sub.set_ylim(1., 10.) 
    fig.savefig('desicmx.brightsky_fexp.png', bbox_inches='tight') 
    return None 


def fexptime_model(airmass, moonill, moonalt, moonsep, sun_alt, sun_sep): 
    ''' 
    '''
    # get sky brightness from updated model 
    wave, sky_bright = Sky.Isky_newKS_twi(airmass, moonill, moonalt, moonsep, sun_alt, sun_sep)
    _, sky_bright_cont = getContinuum(wave.value, sky_bright)
    
    # nominal dark sky brightness 
    config = desisim.simexp._specsim_config_for_wave(wave.value, dwave_out=None, specsim_config_file='desi')
    atm_config = config.atmosphere
    surface_brightness_dict = config.load_table(atm_config.sky, 'surface_brightness', as_dict=True)
    sky_dark= surface_brightness_dict['dark'] 
    
    # get the continuums for dark sky 
    w_cont, sky_dark_cont = getContinuum(wave.value, sky_dark.value)

    # calculate (new sky brightness)/(nominal dark sky brightness), which is the correction
    # factor for the exposure time. 
    wlim = (w_cont > 4500) & (w_cont < 5500) 
    f_exp = np.median((sky_bright_cont / sky_dark_cont)[wlim]) 
    return f_exp  


def fexptime_CMXspec(night, exp, dark_night=20191206, dark_exp=30948): 
    ''' calculate the exposure time correction factor by taking the
    ratio of the CMX sky fluxes:

    (b band sky flux at [night, exp]) / (b band sky flux at [dark_night, dark_exp]) 

    This assumes that the sky observation at [dark_night, dark_exp] is
    close to the fiducial dark condition  
    '''
    # read in exposure metadata
    meta = pickle.load(open('desicmx.exp.metadata.p', 'rb'))

    # get dark sky spectra
    dark_spectra = get_spectra(dark_night, dark_exp, type='sky')
    texp_dark = meta['exptime'][meta['id'] == dark_exp][0]

    # get spectra
    spectra = get_spectra(night, exp, type='sky')
    texp    = meta['exptime'][meta['id'] == exp][0]
    
    wave_grid = np.linspace(3550, 5900, 4000)
    wlim = (wave_grid > 4900) & (wave_grid < 5100) 
    
    fexptime = [] 
    for ispec in range(10): 
        issky       = spectra['keep'][ispec]
        isdarksky   = dark_spectra['keep'][ispec]
        if (np.sum(issky) == 0) or (np.sum(isdarksky) == 0): 
            continue
        
        flux_texp_grid = np.empty((np.sum(issky), len(wave_grid)))
        dark_flux_texp_grid = np.empty((np.sum(isdarksky), len(wave_grid)) )

        for i in range(np.sum(issky)): 
            flux_texp_grid[i,:] = np.interp(wave_grid, 
                    spectra['wave_b'][ispec,issky,:][i,:], 
                    spectra['flux_b'][ispec,issky,:][i,:])/texp
        for i in range(np.sum(isdarksky)): 
            dark_flux_texp_grid[i,:] = np.interp(wave_grid, 
                    dark_spectra['wave_b'][ispec,isdarksky,:][i,:], 
                    dark_spectra['flux_b'][ispec,isdarksky,:][i,:])/texp_dark

        fexptime.append(np.median(np.median(flux_texp_grid[:,wlim], axis=1)) / 
                np.median(np.median(dark_flux_texp_grid[:,wlim], axis=1))) 
    assert len(fexptime) > 0
    return np.median(fexptime), np.std(fexptime) 


def darksky():
    ''' compile dark sky spectra and compare with dark sky model to approximate
    flux calibrations for the spectrographs 

    **EXPOSURE 30948 on NIGHT 20191206 as "fiducial dark sky"**
    '''
    # read in metadata of exposure
    meta = pickle.load(open('desicmx.exp.metadata.p', 'rb'))

    for ispec in range(10): 
        _i = 0 
        waves, fluxes, fluxes_texp, expids = {}, {}, {}, [] 
    
        # dark sky  
        isdark = (
                (meta['sequence'] == 'Spectrographs') &  
                (meta['moon_alt'] < 0.) & 
                (meta['airmass'] < 1.1)
                )
        nights  = meta['night'][isdark]
        exps    = meta['exp'][isdark]

        for night, exp, texp in zip(nights, exps, meta['exptime'][isdark]): 
            if exp in [30784]: continue 
            #if exp in [30947, 30951]: continue # positioners placed on targets
            try: 
                spectra = get_spectra(night, exp, type='sky')
            except AssertionError: 
                continue 

            issky = spectra['keep'][ispec]
            if np.sum(issky) == 0: continue  # no sky spectra in this petal 
            print('%i spectra from %i (exp %i) texp=%f' % (np.sum(issky), night, exp, texp)) 

            for band in ['b', 'r', 'z']: 
                if _i == 0: 
                    waves[band] = spectra['wave_%s' % band][ispec,issky,:] 
                    fluxes[band] = spectra['flux_%s' % band][ispec,issky,:]
                    fluxes_texp[band] = spectra['flux_%s' % band][ispec,issky,:]/texp
                else: 
                    waves[band] = np.concatenate([waves[band], spectra['wave_%s' % band][ispec,issky,:]]) 
                    fluxes[band] = np.concatenate([fluxes[band], spectra['flux_%s' % band][ispec,issky,:]]) 
                    fluxes_texp[band] = np.concatenate([fluxes_texp[band], spectra['flux_%s' % band][ispec,issky,:]/texp]) 
            expids.append(np.repeat(exp, np.sum(issky)))
            _i += 1 
        if _i == 0: continue 
        expids = np.concatenate(expids) 

        uniq_exp = np.unique(expids)
    
        fig = plt.figure(figsize=(20,5))
        for ib, band in enumerate(['b', 'r', 'z']): 
            sub = fig.add_subplot(1,3,ib+1)

            if band == 'b': 
                wave_grid = np.linspace(3550, 5900, 4000)
            elif band == 'r': 
                wave_grid = np.linspace(5600, 7700, 4000)
            elif band == 'z':
                wave_grid = np.linspace(7350, 9800, 4000)

            for _iexp, _exp in enumerate(uniq_exp):  
                isexp = (expids == _exp)

                flux_grid = []
                for _i in range(np.sum(isexp)): 
                    flux_grid.append(
                            np.interp(wave_grid, waves[band][isexp,:][_i], fluxes_texp[band][isexp,:][_i]))
                flux_grid = np.array(flux_grid)
                sub.plot(wave_grid, np.median(flux_grid, axis=0), 
                        c='C%i' % _iexp, lw=1, label='exp %i' % _exp)
                flux_m1sig, flux_p1sig = np.quantile(flux_grid, [0.16, 0.84], axis=0) 
                sub.fill_between(wave_grid, flux_m1sig, flux_p1sig, 
                        facecolor='C%i' % _iexp, edgecolor='none', alpha=0.25)
                if ib == 0:             
                    print('-------------------------') 
                    isexp = (meta['exp'] == _exp) 
                    for k in meta.keys(): 
                        print(k, meta[k][isexp][0])

            if ib == 0: 
                sub.set_xlim(3500, 6100) 
                sub.legend(loc='upper left', fontsize=15) 
            elif ib == 1: 
                sub.set_xlim(5500, 8000) 
            elif ib == 2: 
                sub.set_xlim(7200, 10000) 
            sub.set_ylim(0, 0.75) 
            if ib != 0: sub.set_yticklabels([]) 
            else: sub.set_ylabel('flux (counts/A/sec)', fontsize=15) 
        fig.savefig('desicmx.darksky.spectrograph%i.png' % ispec, bbox_inches='tight') 
    return None 


def darksky_spectrographs(dark_night=20191206, dark_exp=30948):
    ''' compare dark sky spectra for the different spectrographs 

    **EXPOSURE 30948 on NIGHT 20191206 as "fiducial dark sky"**
    '''
    # read in metadata of exposure
    meta = pickle.load(open('desicmx.exp.metadata.p', 'rb'))
    isexp = (meta['exp'] == dark_exp) 
    texp  = meta['exptime'][isexp][0]

    spectra = get_spectra(dark_night, dark_exp, type='sky')

    fig = plt.figure(figsize=(20,5))
    for ispec in range(10): 
        issky = spectra['keep'][ispec]
        if np.sum(issky) == 0: continue

        waves, fluxes, fluxes_texp = {}, {}, {} 
        for ib, band in enumerate(['b', 'r', 'z']): 
            sub = fig.add_subplot(1,3,ib+1)

            waves[band] = spectra['wave_%s' % band][ispec,issky,:] 
            fluxes[band] = spectra['flux_%s' % band][ispec,issky,:]
            fluxes_texp[band] = spectra['flux_%s' % band][ispec,issky,:]/texp

            if band == 'b': 
                wave_grid = np.linspace(3550, 5900, 4000)
            elif band == 'r': 
                wave_grid = np.linspace(5600, 7700, 4000)
            elif band == 'z':
                wave_grid = np.linspace(7350, 9800, 4000)

            flux_grid = []
            for _i in range(np.sum(issky)): 
                flux_grid.append(
                        np.interp(wave_grid, waves[band][_i,:], fluxes_texp[band][_i,:]))
            flux_grid = np.array(flux_grid)
            sub.plot(wave_grid, np.median(flux_grid, axis=0), 
                    c='C%i' % ispec, lw=0.5, label='spectrograph %i' % ispec)
            if ib == 0: 
                sub.set_xlim(3500, 6100) 
                sub.legend(loc='upper left', fontsize=15) 
            elif ib == 1: 
                sub.set_xlim(5500, 8000) 
            elif ib == 2: 
                sub.set_xlim(7200, 10000) 
            sub.set_ylim(0, 0.75) 
            if ib != 0: sub.set_yticklabels([]) 
            else: sub.set_ylabel('flux (counts/A/sec)', fontsize=15) 
        fig.savefig('desicmx.darksky.%i_%i.spectrographs.png' % (dark_night, dark_exp), bbox_inches='tight') 
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
                spectra = get_spectra(night, exp, type='maybe')
            except AssertionError: 
                continue 

            issky = spectra['keep'][ispec]
            if np.sum(issky) == 0: continue  # no sky spectra in this petal 
            print('%i spectra from %i (exp %i)' % (np.sum(issky), night, exp)) 

            for band in ['b', 'r', 'z']: 
                if _i == 0: 
                    waves[band] = spectra['wave_%s' % band][ispec,issky,:] 
                    fluxes[band] = spectra['flux_%s' % band][ispec,issky,:]
                else: 
                    waves[band] = np.concatenate([waves[band], spectra['wave_%s' % band][ispec,issky,:]]) 
                    fluxes[band] = np.concatenate([fluxes[band], spectra['flux_%s' % band][ispec,issky,:]]) 
            _i += 1 
        if _i == 0: continue 
        
        fig = plt.figure(figsize=(15,5))
        for ib, band in enumerate(['b', 'r', 'z']): 
            sub = fig.add_subplot(1,3,ib+1)
            for i in np.arange(waves[band].shape[0]): 
                sub.plot(waves[band][i,:], fluxes[band][i,:], lw=0.1)#, alpha=0.1)
            if ib == 0: sub.set_xlim(3500, 6100) 
            elif ib == 1: sub.set_xlim(5500, 8000) 
            elif ib == 2: sub.set_xlim(7200, 10000) 
            sub.set_ylim(0, 500) 
            if ib != 0: sub.set_yticklabels([]) 
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
            
            try: 
                flux = fitsio.read(fspec, 'FLUX') 
                wave = fitsio.read(fspec, 'WAVELENGTH') 
            except OSError:
                print('--------------------------------') 
                print('Could not open %s' % fspec) 
                print('--------------------------------') 
                continue 

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

        texp = ExpInfo(exp, 'mjd_obs')
        if ExpInfo(exp, 'sequence') == 'Spectrographs': 
            # find nearest GFA exposure 
            _gfa_exp = allexps['id'].to_numpy().astype(int)[is_gfa][np.argmin(np.abs(mjd_gfa - ExpInfo(exp, 'mjd_obs')))]
            if np.abs(texp - ExpInfo(_gfa_exp, 'mjd_obs')) < 0.001:  
                gfa_exp = _gfa_exp

        meta = {} 
        meta['mjd'] = texp 
        for col in ExpInfo.columns: #['airmass', 'skyra', 'skydec', 'moonra', 'moondec']: 
            if col in ['skyra', 'skydec']: 
                if (ExpInfo(exp, col) is None) and (ExpInfo(exp, 'sequence') == 'Spectrographs') and (gfa_exp is not None): 
                    print(exp, gfa_exp)
                    meta[col] = ExpInfo(gfa_exp, col)
                else: 
                    meta[col] = ExpInfo(exp, col)
            elif ExpInfo(exp, col) is not None: 
                meta[col] = ExpInfo(exp, col)
            else: 
                meta[col] = None 

        if meta['skyra'] is not None and meta['skydec'] is not None and meta['mjd'] is not None: 
            print(exp) 
            # append additional observing conditions  
            obscond = get_obscond(meta['skyra'], meta['skydec'], meta['mjd']) 
            for k in obscond.keys(): 
                if k == 'airmass': 
                    meta['_airmass'] = obscond[k]
                else: 
                    meta[k] = obscond[k]

            meta['mjd'] = ExpInfo(exp, 'mjd_obs')
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


def getContinuum(ww, sb): 
    ''' smooth out the sufrace brightness somehow...
    '''
    wavebin = np.linspace(3.6e3, 1e4, 10)
    sb_med = np.zeros(len(wavebin)-1)
    for i in range(len(wavebin)-1): 
        inwbin = ((wavebin[i] < ww) & (ww < wavebin[i+1]) & np.isfinite(sb))
        if np.sum(inwbin) > 0.: 
            sb_med[i] = np.median(sb[inwbin])
    return 0.5*(wavebin[1:]+wavebin[:-1]), sb_med


if __name__=='__main__': 
    #make_obscond_table()
    #darksky()
    #darksky_spectrographs(dark_night=20191206, dark_exp=30948)
    #fexptime_spec(20191022, 20137, dark_night=20191206, dark_exp=30948)
    brightsky()
    #maybe()
