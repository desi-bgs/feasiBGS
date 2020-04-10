#!/bin/python
'''
    
notes
----
*   desisurvey.utils.freeze_iers is finicky about version of astropy. It
    requires version 2 rather than the now commonly used 4
'''
import os 
import sys 
import numpy as np 
import scipy.stats as scistats
# -- astropy --
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon
# --- desihub --- 
import desimodel
import surveysim.stats
import desisurvey.plots
from desisurvey.utils import get_date
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
if 'NERSC_HOST' not in os.environ.keys():  
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

import warnings, astropy._erfa.core
warnings.filterwarnings('ignore', category=astropy._erfa.core.ErfaWarning)

# --- some global variables ---
kpno = EarthLocation.of_site('kitt peak')
# parent dir
_dir = os.path.dirname(os.path.realpath(__file__))
# directoy for surveysim outs 
os.environ['DESISURVEY_OUTPUT'] = os.path.join(os.environ['CSCRATCH'], 
        'desisurvey_output')


def stats_surveysim(name): 
    ''' generate some plots assessing the surveysim run 

    notes 
    -----
    *   https://github.com/desihub/tutorials/blob/master/survey-simulations.ipynb
    '''
    # read in exposures surveysim output  
    f_exp = os.path.join(os.environ['DESISURVEY_OUTPUT'], 
            'exposures_%s.fits' % name)
    exposures = fits.getdata(f_exp, 'exposures') 
    tilestats = fits.getdata(f_exp, 'tiledata')

    # read in stats surveysim output  
    f_stats = os.path.join(os.environ['DESISURVEY_OUTPUT'], 
            'stats_%s.fits' % name)
    stats = surveysim.stats.SurveyStatistics(restore=f_stats)

    print('Survey runs {} to {} and observes {} tiles with {} exposures.'.format(
          get_date(np.min(exposures['mjd'])),
          get_date(np.max(exposures['mjd'])), 
          np.sum(tilestats['snr2frac'] >= 1), len(exposures)))

    print('Number of nights: {}'.format(len(stats._data)))
    stats.summarize()
    
    # -- plot SNR(actual) / SNR (goal) histogram 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    sub.hist(tilestats['snr2frac'], range=(0.75, 1.25), bins=25)
    sub.axvline(np.median(tilestats['snr2frac']), c='r') 
    sub.set_xlabel('Tile SNR(actual) / SNR (goal)')
    sub.set_xlim(0.75, 1.25) 
    fig.savefig(os.path.join(_dir, 'figs', '%s.snr2frac.png' % name),
            bbox_inches='tight') 
    plt.close() 

    # -- plot total exposure time of tiles histogram 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    sub.hist(tilestats['exptime'] / 60, range=(0, 60), bins=30)
    sub.axvline(np.median(tilestats['exptime'] / 60), c='r');
    sub.set_xlabel('Tile Total Exposure Time [min]')
    sub.set_xlim(0., 60.) 
    fig.savefig(os.path.join(_dir, 'figs', '%s.texp.png' % name),
            bbox_inches='tight') 
    plt.close() 

    # plot survey completion as a function of time 
    fig, sub = stats.plot() 
    sub[1].text(0.98, 0.98, name.upper(), ha='right', va='top',
                transform=sub[1].transAxes, fontsize=20)
    fig.savefig(os.path.join(_dir, 'figs', '%s.completion.png' % name),
            bbox_inches='tight') 
    plt.close() 

    # plot exposure time as a function of obsering parameters
    fig = plot_bgs_obs(exposures)  
    fig.savefig(os.path.join(_dir, 'figs', '%s.bgs_obs.png' % name),
            bbox_inches='tight')
    plt.close() 
    return None 


def plot_bgs_obs(exposures): 
    ''' given exposures select BGS exposures and plot them as a function of
    various observational parameters
    '''
    # get observing conditions 
    isbgs, airmass, moon_ill, moon_alt, moon_sep, sun_alt, sun_sep =\
            _get_obs_param(exposures['TILEID'], exposures['MJD']) 
    # check that airmasses are somewhat consistent 
    discrepant = (np.abs(airmass - exposures['AIRMASS'][isbgs]) > 0.1)
    if np.sum(discrepant) > 0: 
        print('%i of %i exposures with discrepant airmass' %
                (np.sum(discrepant), np.sum(isbgs)))

    props = [exposures['AIRMASS'][isbgs], exposures['SEEING'][isbgs], moon_ill,
            moon_alt,  moon_sep, sun_alt, sun_sep] 
    lbls = ['airmass', 'seeing', 'moon illumination', 'moon alitutde', 
            'moon separation', 'sun altitude', 'sun separation'] 
    lims = [(1.,2.), (0., 3.), (0., 1.), (-30., 90.), (30., 180.), 
            (-90., 0.), (30., 180.)]

    fig = plt.figure(figsize=(12,7))
    bkgd = fig.add_subplot(111, frameon=False) 
    for i, prop, lbl, lim in zip(range(len(props)), props, lbls, lims): 
        sub = fig.add_subplot(2,4,i+1) 
        sub.scatter(prop, exposures['EXPTIME'][isbgs]/60., c='k', s=1)
        # plot the median values as well 
        med, bins, _ = scistats.binned_statistic(
                prop, exposures['EXPTIME'][isbgs]/60.,
                statistic='median', bins=10) 
        sub.scatter(0.5*(bins[1:] + bins[:-1]), med, c='C1', s=5)
        sub.set_xlabel(lbl, fontsize=15) 
        sub.set_xlim(lim) 
        sub.set_ylim(0., 30.) 
        if i not in [0,4]: sub.set_yticklabels([]) 

    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_ylabel('exposure time [min]', fontsize=25) 
    fig.subplots_adjust(hspace=0.3)
    return fig 


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


def run_surveysim(name, fconfig, twilight=False, brightsky=False): 
    ''' run surveysim for specified configuration file 
    '''
    fconfig = os.path.join(_dir, fconfig)
    
    flag_twilight = ''
    if twilight: 
        flag_twilight = ' --twilight' 

    flag_brightsky = ''
    if brightsky: 
        flag_brightsky = ' --brightsky' 

    print('surveysim --name %s --config-file %s%s%s' % 
            (name, fconfig, flag_twilight, flag_brightsky)) 
    os.system('surveysim --name %s --config-file %s%s%s' % 
            (name, fconfig, flag_twilight, flag_brightsky)) 
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
        print('already exists:\n\t%s\n\t%s' % (f_ephem, f_init))
    return None


if __name__=="__main__": 
    '''
        >>> python survey_sim.py name fconfig twilight 
    '''
    name        = sys.argv[1]
    fconfig     = sys.argv[2]
    twilight    = sys.argv[3] == 'True'
    brightsky   = sys.argv[4] == 'True'

    if twilight: name += '.twilight'
    if brightsky: name += '.brightsky'

    # check survey init
    #surveyinit()
    # run surveysim
    #run_surveysim(name, fconfig, twilight=twilight, brightsky=brightsky) 
    # get summary statistics of surveysim run
    stats_surveysim(name)
