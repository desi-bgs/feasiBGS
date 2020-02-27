#!/bin/python
'''
    
notes
----
*   desisurvey.utils.freeze_iers is finicky about version of astropy. It
    requires version 2 rather than the now commonly used 4
'''
import os 
import numpy
from astropy.io import fits
# --- desihub --- 
import desimodel
import surveysim.stats
import desisurvey.utils
import desisurvey.plots

#import warnings, matplotlib.cbook, astropy._erfa.core
#warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
#warnings.filterwarnings('ignore', category=astropy._erfa.core.ErfaWarning)

# directoy for surveysim outs 
os.environ['DESISURVEY_OUTPUT'] = os.path.join(os.environ['CSCRATCH'], 
        'desisurvey_output')


def run_surveysim(name, fconfig): 
    '''
    '''
    _dir    = os.path.dirname(os.path.realpath(__file__))
    fconfig = os.path.join(_dir, fconfig)
    print('surveysim --name %s --config-file %s' % (name, fconfig)) 
    os.system('surveysim --name %s --config-file %s' % (name, fconfig)) 
    return None 


def surveyinit(): 
    ''' check that the ephemerides and surveyinit files are in
    `DESISURVEY_OUTPUT` directory. Otherwise generate new files (this takes a
    long time)  
    '''
    # tabulated ephemerides during 2019-25
    f_ephem = os.path.join(os.environ['DESISURVEY_OUTPUT'],
            'ephem_2019-01-01_2025-12-31.fits') 
    # estimated average weather and optimized initial hour angle (HA)
    # assignments for each tile.c
    f_init = os.path.join(os.environ['DESISURVEY_OUTPUT'],
            'surveyinit.fits') 

    if not os.path.isfile(f_ephem) or not os.path.isfile(f_init): 
        os.system('surveyinit --verbose') 
    else: 
        print('\t%s\n\t%s\nalready exist' % (f_ephem, f_init))
    return None


def tutorial(): 
    os.environ['DESISURVEY_OUTPUT'] = '/global/cfs/cdirs/desi/datachallenge/surveysim2018/weather/000'  

    exposures = fits.getdata(
            os.path.join(os.environ['DESISURVEY_OUTPUT'], 'exposures.fits'), 
            'exposures') 

    tilestats = fits.getdata(
            os.path.join(os.environ['DESISURVEY_OUTPUT'], 'exposures.fits'), 
            'tiledata')

    print('Survey runs {} to {} and observes {} tiles with {} exposures.'.format(
          desisurvey.utils.get_date(numpy.min(exposures['mjd'])),
          desisurvey.utils.get_date(numpy.max(exposures['mjd'])), 
          numpy.sum(tilestats['snr2frac'] >= 1), len(exposures)))

    print(repr(exposures[:3]))
    print(repr(tilestats[:3]))
    tiles = desisurvey.tiles.get_tiles()
    print(tiles.tileRA.shape, tilestats.shape)

    print(tilestats[:3]) 
    return None 


if __name__=="__main__": 
    surveyinit()
    run_surveysim('default', 'config_default.yaml')
