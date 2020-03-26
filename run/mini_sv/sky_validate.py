#!/bin/python
'''

script to validate the BGS sky model

'''
import os 
import h5py 
import glob
import fitsio
import numpy as np 

from desitarget.cmx import cmx_targetmask
# -- astorpy -- 
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon

import warnings, astropy._erfa.core
warnings.filterwarnings('ignore', category=astropy._erfa.core.ErfaWarning)


kpno = EarthLocation.of_site('kitt peak')

dir_coadd = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'


def compile_skies(): 
    ''' compile all the sky fibers 
    '''
    # read observing conditions compiled from GFAs
    gfa = fitsio.read(os.path.join(
        '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions', 
        'offline_all_guide_ccds_thru_20200315.fits'))
    f_coadds = glob.glob(os.path.join(dir_coadd, 'coadd*fits')) 
    
    sky_data = {} 
    for k in ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt',
            'sun_sep', 'flux', 'tileid', 'date', 'expid', 'mjd']: 
        sky_data[k] = [] 

    for f_coadd in f_coadds: 
        coadd = fitsio.read(f_coadd)
        flux = fitsio.read(f_coadd, ext=3)  
        #ivar = fitsio.read(f_coadd, ext=4) 
        #mask = fitsio.read(f_coadd, ext=5) 
        #_res = fitsio.read(f_coadd, ext=6) 
    
        goodfiber = (coadd['FIBERSTATUS'] == 0)
        issky = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SKY')) != 0 
        print('--- %s ---' % f_coadd) 
        print('%i sky fibers' % np.sum(issky)) 
        print('%i good sky fibers' % np.sum(goodfiber & issky))


        # match to GFA obs condition using NIGHT and EXPID
        date    = int(f_coadd.split('-')[2]) 
        tileid  = int(f_coadd.split('-')[1]) 
        expid   = int(f_coadd.split('-')[-1].replace('.fits',''))
        
        # match to GFA 
        m_gfa = ((gfa['NIGHT'] == date) & (gfa['EXPID'] == expid))
        if np.sum(m_gfa) == 0: 
            continue 
        print('%i GFA exposure matches' % np.sum(m_gfa)) 
        assert (gfa['MJD'][m_gfa].max() - gfa['MJD'][m_gfa].min()) < 0.08333333333
        
        # median MJD of GFA data 
        mjd_mid = np.median(gfa['MJD'][m_gfa])
        
        # these values more or less agree with the GFA values 
        _airmass, _moon_ill, _moon_alt, _moon_sep, _sun_alt, _sun_sep = \
                _get_obs_param(coadd['TARGET_RA'][goodfiber & issky],
                        coadd['TARGET_DEC'][goodfiber & issky], mjd_mid)

        sky_data['flux'].append(flux[goodfiber & issky,:]) 
        sky_data['tileid'].append(np.repeat(tileid, len(_airmass)))
        sky_data['date'].append(np.repeat(date, len(_airmass)))
        sky_data['expid'].append(np.repeat(expid, len(_airmass))) 
        sky_data['mjd'].append(np.repeat(mjd_mid, len(_airmass)))

        sky_data['airmass'].append(_airmass)
        sky_data['moon_ill'].append(np.repeat(_moon_ill, len(_airmass))) 
        sky_data['moon_alt'].append(np.repeat(_moon_alt, len(_airmass))) 
        sky_data['moon_sep'].append(_moon_sep) 
        sky_data['sun_alt'].append(np.repeat(_sun_alt, len(_airmass))) 
        sky_data['sun_sep'].append(_sun_sep) 

    wave = fitsio.read(f_coadd, ext=2)  
    
    f = h5py.File(os.path.join(dir_coadd, 
        'sky_fibers.coadd_gfa.minisv2_sv0.hdf5'), 'w') 
    for k in sky_data.keys():
        f.create_dataset(k, data=np.concatenate(sky_data[k], axis=0)) 
    f.create_dataset('wave', data=wave)
    f.close() 
    return None


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


if __name__=='__main__': 

    compile_skies()
