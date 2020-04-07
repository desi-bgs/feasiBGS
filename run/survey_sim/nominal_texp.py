'''

script to decide on the nominal exposure time based on the survey strategy
simulation and completeness simulations. 

1. Compile 8 exposures from the v2 strategy simulation outputs. 

2. Construct spectral completeness simulations for the 8 exposures using the
CMX updated sky model for t_nominal = 130, 150, 180, 200

3. RUn redrock on the spectra and determine minimum t_nominal for 95% overall
redshift success (L2.X.5 requirement). 


Notes: 
* make sure that z success is consistent for the different exposures.


'''
import os 
import sys 
import numpy as np 
from astropy.io import fits
# -- desihub -- 
from desisurvey.utils import get_date



def compile_exposures(): 
    ''' compile 8 BGS exposures from v2 survey strategy simulations. Generates
    a plot of the observing conditions for all BGS exposures and 
    '''
    # 150s nominal exposure time with twilight 
    name = '150s_skybranch_v2.twilight.brightsky'

    # read in exposures surveysim output  
    f_exp = os.path.join(os.environ['DESISURVEY_OUTPUT'], 
            'exposures_%s.fits' % name)
    exposures = fits.getdata(f_exp, 'exposures') 
    tilestats = fits.getdata(f_exp, 'tiledata')

    finish_snr = (tilestats >= 1) 
    print('Survey runs {} to {} and observes {} tiles with {} exposures.'.format(
          get_date(np.min(exposures['mjd'])),
          get_date(np.max(exposures['mjd'])), 
          np.sum(finish_snr), len(exposures)))
    
    # get observing conditions for the BGS exposures 
    isbgs, airmass, moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = \
            _get_obs_param(exposures['TILEID'], exposures['MJD']) 
    # check that airmass is consistent 
    print(airmass[::100]) 
    print(exposures['AIRMASS'][::100]) 

    print(isbgs) 
    return None 


def _get_obs_param(tileid, mjd):
    ''' get observing condition given tileid and time of observation 
    '''
    # read tiles and get RA and Dec
    tiles = desisurvey.tiles.get_tiles()
    indx = np.array([list(tiles.tileID).index(id) for id in tileid]) 
    # pass number
    tile_passnum = tiles.passnum[indx]
    # BGS passes only  
    isbgs = (tile_passnum > 4) 
    
    tile_ra     = tiles.tileRA[indx][isbgs]
    tile_dec    = tiles.tileDEC[indx][isbgs]
    mjd         = mjd[isbgs]

    # get observing conditions
    coord = SkyCoord(ra=tile_ra * u.deg, dec=tile_dec * u.deg) 
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
    return isbgs, airmass, moon_ill, moon_alt, moon_sep, sun_alt, sun_sep



if __name__=="__main__": 
    compile_exposures()
