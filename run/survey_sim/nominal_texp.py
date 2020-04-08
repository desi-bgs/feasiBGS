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
import h5py 
import numpy as np 
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
# -- feasibgs -- 
from feasibgs import util as UT 
from feasibgs import skymodel as Sky
from feasibgs import forwardmodel as FM 
# -- desihub -- 
from desisurvey.utils import get_date
from desisurvey.etc import exposure_factor
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
warnings.filterwarnings('ignore', category=FutureWarning)

# parent dir
_dir = os.path.dirname(os.path.realpath(__file__))
# directoy for surveysim outs 
os.environ['DESISURVEY_OUTPUT'] = os.path.join(os.environ['CSCRATCH'], 
        'desisurvey_output')


def compile_exposures(): 
    ''' compile 8 BGS exposures from v2 survey strategy simulations. Generates
    a plot of the observing conditions for all BGS exposures and 
    '''
    # 150s nominal exposure time with twilight 
    name = '150s_skybranch_v2.twilight.brightsky'
    tfid = 150.

    # read in exposures surveysim output  
    f_exp = os.path.join(os.environ['DESISURVEY_OUTPUT'], 
            'exposures_%s.fits' % name)
    exposures = fits.getdata(f_exp, 'exposures') 
    tilestats = fits.getdata(f_exp, 'tiledata')

    finish_snr = (tilestats['snr2frac'] >= 1) 
    print('Survey runs {} to {} and observes {} tiles with {} exposures.'.format(
          get_date(np.min(exposures['mjd'])),
          get_date(np.max(exposures['mjd'])), 
          np.sum(finish_snr), len(exposures)))
    
    # get observing conditions for the BGS exposures 
    isbgs, airmass, moon_ill, moon_alt, moon_sep, sun_alt, sun_sep = \
            _get_obs_param(exposures['TILEID'], exposures['MJD']) 
    # calculate exposure factor (airmass and bright exposure factors) 
    f_exp = np.array([exposure_factor(np.array([airmass[i]]), moon_ill[i],
        np.array([moon_sep[i]]), moon_alt[i], np.array([sun_sep[i]]),
        sun_alt[i]) for i in range(np.sum(isbgs))]).flatten() 

    # randomly select 8 BGS exposures where the exposures
    i_rand = np.random.choice(np.arange(np.sum(isbgs)), 8, replace=False)

    # concatenate appropriate observing conditions 
    exps = Table(exposures[isbgs][i_rand]) 
    exps['airmass']     = np.array(airmass[i_rand]) 
    exps['moon_ill']    = np.array(moon_ill[i_rand])
    exps['moon_alt']    = np.array(moon_alt[i_rand])
    exps['moon_sep']    = np.array(moon_sep[i_rand])
    exps['sun_alt']     = np.array(sun_alt[i_rand])
    exps['sun_sep']     = np.array(sun_sep[i_rand])
    exps['exposure_factor'] = np.array(f_exp[i_rand])
    # write out exposures 
    exps.write(os.path.join(_dir, 'bgs_exps.8random.fits'))

    # plot random exposures 
    fig = plt.figure(figsize=(20,5))

    props   = [airmass, moon_ill, moon_sep]
    lbls    = ['airmass', 'moon ill', 'moon sep']
    lims    = [(1., 2.), (0., 1.), (0., 180.)]
    for i, prop in enumerate(props):  
        sub = fig.add_subplot(1,4,i+1) 
        sub.scatter(prop, moon_alt, c='k', s=2) 
        sub.scatter(prop[i_rand], moon_alt[i_rand], c='C1', s=4) 

        sub.set_xlabel(lbls[i], fontsize=15) 
        sub.set_xlim(lims[i])
        if i == 0: sub.set_ylabel('moon alt', fontsize=15) 
        sub.set_ylim(-90., 90.) 

    sub = fig.add_subplot(144) 
    sub.scatter(sun_sep, sun_alt, c='k', s=2) 
    sub.scatter(sun_sep[i_rand], sun_alt[i_rand], c='C1', s=4) 
    sub.set_xlabel('sun sep.', fontsize=15)
    sub.set_ylabel('sun alt.', fontsize=15) 
    sub.set_xlim(0., 180.) 
    sub.set_ylim(-90., -10.) 
    fig.savefig(os.path.join(_dir, 'figs', 'tnom.compile_exps.%s.png' % name),
            bbox_inches='tight') 
    plt.close() 
    return None 


def construct_comp_sims(tnom): 
    ''' Construct BGS spectral simulations for 5000 galaxies in the GAMA G15 field using
    observing conditions sampled from surveysim output exposures.
    '''
    # read in 8 sampled bright exposures
    exps = Table.read(os.path.join(_dir, 'bgs_exps.8random.fits'))
    airmass     = exps['AIRMASS']
    moon_ill    = exps['moon_ill']
    moon_alt    = exps['moon_alt']
    moon_sep    = exps['moon_sep']
    sun_alt     = exps['sun_alt']
    sun_sep     = exps['sun_sep']
    seeing      = exps['SEEING']
    transp      = exps['TRANSP']
    exp_factor  = exps['exposure_factor'] 
    n_exps      = len(airmass) 

    # calculate v2 sky brightness 
    Iskies = [] 
    for i in range(n_exps): 
        wave_sky, Isky  = Sky.Isky_regression(airmass[i], moon_ill[i], moon_alt[i], moon_sep[i], sun_alt[i], sun_sep[i]) 
        Iskies.append(Isky * 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second) 
    Iskies = np.array(Iskies) 
    
    # scale up the exposure times
    texp = exp_factor * tnom 
        
    # read in noiseless spectra
    specfile = os.path.join(UT.dat_dir(), 'survey_sim', 'GALeg.g15.sourceSpec.5000.seed0.hdf5')
    fspec = h5py.File(specfile, 'r') 
    wave = fspec['wave'][...]
    flux = fspec['flux'][...] 
    
    for iexp in range(len(airmass)):
        _fexp = os.path.join(UT.dat_dir(), 'survey_sim', 
                'comp_sim.tnom%.f.exp%i.fits' % (tnom, iexp))

        if not os.path.isfile(_fexp):
            print('--- constructing %s ---' % _fexp) 
            print('\tt_exp=%.f (factor %.fx)' % (texp[iexp],exp_factor[iexp]))
            print('\tairmass=%.2f, seeing=%.2f' % (airmass[iexp], seeing[iexp]))
            print('\tmoon ill=%.2f alt=%.f, sep=%.f' % (moon_ill[iexp], moon_alt[iexp], moon_sep[iexp]))
            print('\tsun alt=%.f, sep=%.f' % (sun_alt[iexp], sun_sep[iexp]))
            
            # iexp-th sky spectra 
            Isky = [wave_sky, Iskies[iexp]]
            
            # simulate the exposures 
            fdesi = FM.fakeDESIspec()
            bgs = fdesi.simExposure(wave, flux, 
                exptime=texp[iexp],
                airmass=airmass[iexp], 
                seeing=seeing[iexp], 
                Isky=Isky, 
                filename=_fexp) 

            # --- check sims --- 
            fig = plt.figure(figsize=(10,20))
            sub = fig.add_subplot(411) 
            sub.plot(wave_sky, Iskies[iexp], c='C1') 
            sub.text(0.05, 0.95, 
                '\n'.join([
                    'texp=%.f, exp. factor=%.1f' % (texp[iexp], exp_factor[iexp]), 
                    'airmass=%.2f, seeing=%.1f' % (airmass[iexp], seeing[iexp]), 
                    'moon ill=%.2f, alt=%.f, sep=%.f' % (moon_ill[iexp], moon_alt[iexp], moon_sep[iexp]), 
                    'sun alt=%.f, sep=%.f' % (sun_alt[iexp], sun_sep[iexp])]),
                    ha='left', va='top', transform=sub.transAxes, fontsize=15)
            sub.legend(loc='upper right', frameon=True, fontsize=20) 
            sub.set_xlim([3e3, 1e4]) 
            sub.set_ylim([0., 20.]) 
            for i in range(3): 
                sub = fig.add_subplot(4,1,i+2)
                for band in ['b', 'r', 'z']: 
                    sub.plot(bgs.wave[band], bgs.flux[band][i], c='C1') 
                sub.plot(wave, flux[i], c='k', ls=':', lw=1, label='no noise')
                if i == 0: sub.legend(loc='upper right', fontsize=20)
                sub.set_xlim([3e3, 1e4]) 
                sub.set_ylim([0., 15.]) 
            bkgd = fig.add_subplot(111, frameon=False) 
            bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            bkgd.set_xlabel('rest-frame wavelength [Angstrom]', labelpad=10, fontsize=25) 
            bkgd.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', labelpad=10, fontsize=25) 
            fig.savefig(_fexp.replace('.fits', '.png'), bbox_inches='tight') 
            plt.close()

        # run redrock 
        run_redrock(_fexp, clobber=False)

    return None 


def run_redrock(fspec, clobber=False): 
    ''' run redrock on completeness simulation constructed using
    `construct_comp_sims` above. 
    '''
    _dir = os.path.dirname(os.path.abspath(fspec))
    name = os.path.basename(fspec).split('.fits')[0]
    frr     = os.path.join(_dir, 'redrock.%s.h5' % name)
    fzbest  = os.path.join(_dir, 'zbest.%s.fits' % name)

    if os.path.isfile(fzbest) and not clobber: 
        return None  

    print('running redrock on %s' % os.path.basename(fspec))
    print('\tconstruction %s' % frr)
    print('\tconstruction %s' % fzbest)
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -N 1", 
        "#SBATCH -C haswell", 
        "#SBATCH -q regular", 
        '#SBATCH -J rr_%s' % name,
        '#SBATCH -o _rr_%s.o' % name,
        "#SBATCH -t 01:00:00", 
        "", 
        "export OMP_NUM_THREADS=1", 
        "export OMP_PLACES=threads", 
        "export OMP_PROC_BIND=spread", 
        "", 
        "", 
        "conda activate desi", 
        "", 
        "srun -n 32 -c 2 --cpu-bind=cores rrdesi_mpi -o %s -z %s %s" % (frr, fzbest, fspec), 
        ""]) 
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()

    os.system('sbatch script.slurm') 
    os.system('rm script.slurm') 
    return None 


def compare_zsuccess(deltachi2=40.): 
    ''' compare the redshift success rates for the different observing
    conditions and nominal exposure times 
    '''
    tnoms = [130., 150., 180.]
    # read in 8 sampled bright exposures
    exps = Table.read(os.path.join(_dir, 'bgs_exps.8random.fits'))
    airmass     = exps['AIRMASS']
    moon_ill    = exps['moon_ill']
    moon_alt    = exps['moon_alt']
    moon_sep    = exps['moon_sep']
    sun_alt     = exps['sun_alt']
    sun_sep     = exps['sun_sep']
    seeing      = exps['SEEING']
    transp      = exps['TRANSP']
    exp_factor  = exps['exposure_factor'] 
    n_exps      = len(airmass) 

    # read true redshifts and r magnitude 
    spec = h5py.File(os.path.join(UT.dat_dir(), 'survey_sim', 
        'GALeg.g15.sourceSpec.5000.seed0.hdf5'), 'r') 
    ztrue = spec['zred'][...]
    r_mag = UT.flux2mag(spec['legacy-photo']['flux_r'][...], method='log') 

    fig = plt.figure(figsize=(20,10))
    for iexp in range(n_exps): 
        sub = fig.add_subplot(2,4,iexp+1) 
        sub.plot([15., 22.], [1., 1.], c='k', ls='--', lw=2)
    
        zsucc_bright, zsucc_faint, _plts = [], [], [] 
        for i, tnom in enumerate(tnoms): 
            # read in redrock output
            rr = Table.read(os.path.join(UT.dat_dir(), 'survey_sim',
                    'zbest.comp_sim.tnom%.f.exp%i.fits' % (tnom, iexp)), 
                    hdu=1)
            _zsuc   = UT.zsuccess(rr['Z'], ztrue, rr['ZWARN'],
                    deltachi2=rr['DELTACHI2'], min_deltachi2=deltachi2)
            wmean, rate, err_rate = UT.zsuccess_rate(r_mag, _zsuc,
                    range=[15,22], nbins=28, bin_min=10) 

            _plt = sub.errorbar(wmean, rate, err_rate, 
                    fmt='.C%i' % i, elinewidth=2, markersize=10)
            _plts.append(_plt) 

            # redshift success for bright and faint samples
            # for now lets assume that we get all the redshifts for r < 18
            # objects 
            _zsuc[r_mag < 18.] = True

            bright = (r_mag < 19.5) 
            zsucc_bright.append(100 *
                    float(np.sum(_zsuc[bright]))/float(np.sum(bright)))
            faint = (r_mag < 20.) 
            zsucc_faint.append(100 *
                    float(np.sum(_zsuc[faint]))/float(np.sum(faint)))
        
        obs_cond = '\n'.join([
            r'$f_{\rm exp}=%.1f\times$' % exp_factor[iexp], 
            'airmass=%.2f, seeing=%.2f' % (airmass[iexp], seeing[iexp]), 
            'moon ill=%.2f, alt=%.f, sep=%.f' % (moon_ill[iexp], moon_alt[iexp], moon_sep[iexp]),
            'sun alt=%.f, sep=%.f' % (sun_alt[iexp], sun_sep[iexp])])
        sub.text(0.02, 0.02, obs_cond, ha='left', va='bottom',
                transform=sub.transAxes, fontsize=10)
        for i in range(len(tnoms)): 
            sub.text(19.5, 0.9-0.025*i, '%.1f' % zsucc_bright[i], color='C%i' % i, 
                    ha='right', va='bottom', fontsize=10)
            sub.text(20., 0.9-0.025*i, '%.1f' % zsucc_faint[i], color='C%i' % i, 
                    ha='right', va='bottom', fontsize=10)

        sub.vlines(19.5, 0., 1.2, color='k', linestyle=':', linewidth=1)
        sub.vlines(20.0, 0., 1.2, color='k', linestyle=':', linewidth=1)
        sub.set_xlim([16., 21.]) 
        sub.set_xticks([17, 18, 19, 20, 21]) 
        if iexp < 4: sub.set_xticklabels([]) 
        sub.set_ylim([0.6, 1.1])
        sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
        if iexp not in [0,4]: sub.set_yticklabels([]) 

    sub.legend(_plts, 
            [r'$t_{\rm nom} = %.fs$' % tnom for tnom in tnoms], 
            fontsize=15, handletextpad=0.1, loc='lower right') 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'Legacy DR7 $r$ magnitude', labelpad=10, fontsize=30)
    bkgd.set_ylabel(r'redrock $z$ success rate', labelpad=10, fontsize=30)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(os.path.join(_dir, 'figs',
        'tnom.compare_zsuccess.dchi2_%.f.png' % deltachi2), 
        bbox_inches='tight') 
    return None 


def _get_obs_param(tileid, mjd):
    ''' get observing condition given tileid and time of observation 
    '''
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon
    import desisurvey.tiles 
    kpno = EarthLocation.of_site('kitt peak')
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
    #compile_exposures()
    #for tnom in [130., 150., 180., 200.]:
    #    construct_comp_sims(tnom)
    compare_zsuccess(deltachi2=40.)
