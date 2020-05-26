#!/bin/python
'''


compile CMX sky data  by correcting for throughput and 
converting to surface brightness 


'''
import os 
import h5py 
import glob
import fitsio
import numpy as np 

import desisim.simexp
import specsim.instrument
from specsim.simulator import Simulator

from desispec.io import read_sky
from desimodel.io import load_throughput
from desitarget.cmx import cmx_targetmask
# -- astorpy -- 
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon

import warnings, astropy._erfa.core
warnings.filterwarnings('ignore', category=astropy._erfa.core.ErfaWarning)


kpno = EarthLocation.of_site('kitt peak')

dir_redux = "/global/cfs/cdirs/desi/spectro/redux/daily" 
dir_coadd = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'

wave = np.arange(3523., 9923., 0.8) 
config = desisim.simexp._specsim_config_for_wave(wave, dwave_out=0.8, specsim_config_file='desi')

instrument = specsim.instrument.initialize(config, True)

plate_scale = config.load_table(config.instrument.plate_scale, ['radius', 'radial_scale', 'azimuthal_scale'], interpolate=False)
radial_fiber_size = (0.5 * instrument.fiber_diameter /plate_scale['radial_scale'])
azimuthal_fiber_size = (0.5 * instrument.fiber_diameter / plate_scale['azimuthal_scale'])

mean_fiber_area = np.mean(np.pi*radial_fiber_size * azimuthal_fiber_size)
print('mean fiber_area = %f' % mean_fiber_area.value) 

desi = Simulator(config, num_fibers=1)
sbunit = desi.atmosphere.surface_brightness.unit


def compile_skies(): 
    ''' compile all the sky fibers 
    '''
    # read observing conditions compiled from GFAs
    gfa = fitsio.read(os.path.join(
        '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions', 
        'offline_all_guide_ccds_thru_20200315.fits'))

    bgs_minisv_tiles    = [70500, 70502, 70510]
    bgs_sv0_tiles       = [66000, 66014, 66003]
    bgs_tiles = bgs_minisv_tiles + bgs_sv0_tiles 
    
    cmx = [] 
    for tile in bgs_tiles: 
        dates = get_dates(tile)
        for date in dates: 
            exps = get_exposures(tile, date)
            for exp in exps: 
                spectographs = get_spectograph(tile, date, exp)
                for spec in spectographs: 
                    cmx.append([tile, date, exp, spec]) 

    sky_data = {} 
    for k in ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt',
            'sun_sep', 'sky_sb_b', 'sky_sb_r', 'sky_sb_z',
            'tileid', 'date', 'expid', 'spectrograph', 'mjd', 'transparency',
            'transp_min', 'transp_max', 'fwhm', 'exptime']:  
        sky_data[k] = [] 

    for i, _cmx in enumerate(cmx): 
        _tileid, _date, _exp, _spec = _cmx 
        print('--- %i, %i, %i, %i ---' % (_tileid, _date, _exp, _spec))

        f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' %
                (_tileid,_date, _spec, str(_exp).zfill(8)))

        dir_exp = os.path.join(dir_redux, 'exposures', str(_date), str(_exp).zfill(8)) 
        f_sky = lambda band: os.path.join(dir_exp, 
                'sky-%s%i-%s.fits' % (band, _spec, str(_exp).zfill(8)))

        if not os.path.isfile(f_coadd): 
            print('... no coadd: %s' % os.path.basename(f_coadd))
            continue 
    
        # read coadd and sky data. coadd data is used to determine good sky
        # fibers
        coadd       = fitsio.read(f_coadd)

        sky_b       = read_sky(f_sky('b'))
        sky_r       = read_sky(f_sky('r'))
        sky_z       = read_sky(f_sky('z'))

        exptime     = sky_b.header['EXPTIME']

        if i == 0: 
            wave_b = sky_b.wave 
            wave_r = sky_r.wave 
            wave_z = sky_z.wave 

        is_good = (coadd['FIBERSTATUS'] == 0)
        is_sky  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SKY')) != 0 
        good_sky = is_good & is_sky

        # match to GFA obs condition using NIGHT and EXPID
        m_gfa = ((gfa['NIGHT'] == int(_date)) & (gfa['EXPID'] == int(_exp)))
        if np.sum(m_gfa) == 0: 
            print('... no match to GFA conditions')
            continue 
        assert (gfa['MJD'][m_gfa].max() - gfa['MJD'][m_gfa].min()) < 0.08333333333
        
        # median MJD of GFA data 
        mjd_mid = np.median(gfa['MJD'][m_gfa])
        # median transparency from GFA data
        transp = gfa['TRANSPARENCY'][m_gfa]
        not_nan = np.isfinite(transp)
        if np.sum(not_nan) > 0: 
            transp_min = np.min(transp[not_nan])
            transp_max = np.max(transp[not_nan])
            transp_mid = np.median(transp[not_nan])
        else: 
            transp_min = 0.
            transp_max = 1.
            transp_mid = 1.
        # FWHM  
        _fwhm = gfa['FWHM_ASEC'][m_gfa]
        fwhm = np.median(_fwhm[np.isfinite(_fwhm)]) 
        
        # these values more or less agree with the GFA values 
        _airmass, _moon_ill, _moon_alt, _moon_sep, _sun_alt, _sun_sep = \
                _get_obs_param(coadd['TARGET_RA'][good_sky],
                        coadd['TARGET_DEC'][good_sky], mjd_mid)

        print('%.f exptime' % exptime)
        print('%i sky fibers' % np.sum(is_sky)) 
        print('%i good sky fibers' % np.sum(good_sky))
        print('%.2f < transp < %.2f' % (transp_min, transp_max))

        sky_data['tileid'].append(np.repeat(_tileid, len(_airmass)))
        sky_data['date'].append(np.repeat(_date, len(_airmass)))
        sky_data['expid'].append(np.repeat(_exp, len(_airmass))) 
        sky_data['spectrograph'].append(np.repeat(_spec, len(_airmass)))
        sky_data['mjd'].append(np.repeat(mjd_mid, len(_airmass)))
        sky_data['transparency'].append(np.repeat(transp_mid, len(_airmass)))
        sky_data['transp_min'].append(np.repeat(transp_min, len(_airmass)))
        sky_data['transp_max'].append(np.repeat(transp_max, len(_airmass)))
        sky_data['fwhm'].append(np.repeat(fwhm, len(_airmass)))
        sky_data['exptime'].append(np.repeat(exptime, len(_airmass)))
    
        # sky surface brightness
        sky_data['sky_sb_b'].append(convert_sky_rate(sky_b, good_sky, exptime, 'b'))
        sky_data['sky_sb_r'].append(convert_sky_rate(sky_r, good_sky, exptime, 'r')) 
        sky_data['sky_sb_z'].append(convert_sky_rate(sky_z, good_sky, exptime, 'z'))
        
        # observing conditions 
        sky_data['airmass'].append(_airmass)
        sky_data['moon_ill'].append(np.repeat(_moon_ill, len(_airmass))) 
        sky_data['moon_alt'].append(np.repeat(_moon_alt, len(_airmass))) 
        sky_data['moon_sep'].append(_moon_sep) 
        sky_data['sun_alt'].append(np.repeat(_sun_alt, len(_airmass))) 
        sky_data['sun_sep'].append(_sun_sep) 

    f = h5py.File(os.path.join(dir_coadd, 'sky_fibers.cmx.v1.hdf5'), 'w') 
    for k in sky_data.keys():
        f.create_dataset(k, data=np.concatenate(sky_data[k], axis=0)) 
    f.create_dataset('wave_b', data=wave_b)
    f.create_dataset('wave_r', data=wave_r)
    f.create_dataset('wave_z', data=wave_z)
    f.close() 
    return None


def convert_sky_rate(inc, good_sky, _exptime, band):
    # The incident rate is in phot per 0.8A wavelength bin.
    # Convert to surface brightness units (sbunit) suitable as input to specsim.
    _wave = inc.wave * u.Angstrom
    dwave = np.gradient(_wave)
    
    ephot = (const.h * const.c / _wave / u.ph).to(u.erg / u.ph)
    etendue = (instrument.effective_area * mean_fiber_area).to(u.cm ** 2 * u.arcsec ** 2)
    
    # Calculate surface brightness on the 0.8A reduction grid.
    surfbrightness = ((inc.flux[good_sky,:] * u.ph / (_exptime * u.s)) / dwave * ephot / etendue).to(sbunit)
    
    # correct for throughput 
    through = load_throughput(band).hardware_throughput(inc.wave)
    return surfbrightness / through 


def _get_obs_param(ra, dec, mjd):
    ''' get observing condition given tileid and time of observation 
    '''
    # get observing conditions
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg) 
    utc_time = Time(mjd, format='mjd') # observed time (UTC)          

    kpno_altaz = AltAz(obstime=utc_time, location=kpno) 
    coord_altaz = coord.transform_to(kpno_altaz)

    airmass = coord_altaz.secz

    # sun
    sun         = get_sun(utc_time) 
    sun_altaz   = sun.transform_to(kpno_altaz) 
    sun_alt     = sun_altaz.alt.deg
    sun_sep     = sun.separation(coord).deg # sun separation
    # moon
    moon        = get_moon(utc_time)
    moon_altaz  = moon.transform_to(kpno_altaz) 
    moon_alt    = moon_altaz.alt.deg 
    moon_sep    = moon.separation(coord).deg #coord.separation(self.moon).deg
            
    elongation  = sun.separation(moon)
    phase       = np.arctan2(sun.distance * np.sin(elongation), moon.distance - sun.distance*np.cos(elongation))
    moon_phase  = phase.value
    moon_ill    = (1. + np.cos(phase))/2.
    return airmass, moon_ill.value, moon_alt, moon_sep, sun_alt, sun_sep


def get_dates(tileid): 
    dates = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), "*"))
    dates = [int(os.path.basename(date)) for date in dates]
    return dates 


def get_exposures(tileid, date): 
    cframes = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), str(date),
        'cframe-b*.fits')) 
    exps = [int(cframe.split('-')[-1].split('.fits')[0]) for cframe in cframes]
    return np.unique(exps)


def get_spectograph(tileid, date, exp): 
    cframes = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), str(date),
        'cframe-b*-%s.fits' % str(exp).zfill(8)))
    spectographs = [int(os.path.basename(cframe).split('-')[1].split('b')[-1])
            for cframe in cframes]
    return spectographs


if __name__=='__main__': 
    compile_skies()
