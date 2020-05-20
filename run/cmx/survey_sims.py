'''


validate survey simulations using CMX data. 


updates
-------
* 5/19/2020: created script and test to compare which wavelength range I should
  use for the exposure time correction factor 
'''
import os 
import h5py 
import numpy as np 
import astropy.units as u 
# -- feasibgs --
from feasibgs import util as UT
from feasibgs import forwardmodel as FM 
# -- desihub -- 
import desispec.io 
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
if 'NERSC_HOST' not in os.environ: 
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


dir = '/global/cscratch1/sd/chahah/feasibgs/cmx/survey_sims/'


def texp_factor_wavelength(): 
    ''' Q: Should the exposure time correction factor be determined by sky
    surface brightness ratio at 5000A or 6500A? 

    sky surface brightness ratio = (sky surface brightness) / (nominal dark sky) 
    
    We will answer this by constructing a set of nominal dark sky exposure
    spectra  with 150s exposure time, getting the redshift success rate for
    these spectra. Then we'll compare the redshift success rate for 
      1. exposure spectra constructed with CMX sky brightness and 
         texp = 150s x (sky ratio at 5000A) 
      2. exposure spectra constructed with CMX sky brightness and 
         texp = 150s x (sky ratio at 6500A) 

    We use CMX sky brightness during bright exposures.

    Whichever redshift success rate is coser to the nominal dark exposure z
    success rate will determine the exposure factor
    '''
    # generate spectra for nominal dark sky exposure as reference
    spec_nom = nomdark_spectra() 
    # run redrock on nominal dark sky exposure spectra 
    frr_nom = run_redrock(os.path.join(dir, 'exp_spectra.nominal_dark.150s.fits'), overwrite=False)
    
    # read in CMX sky data 
    skies = cmx_skies()
    # select CMX exposures when the sky was substantially brighter than dark time
    # arbitrarily chose 3x brighter 
    bright = (skies['sky_ratio_b'] > 3) 
    expids = np.random.choice(np.unique(skies['expid'][bright]), size=5,
            replace=False) 
    print('%i CMX exposures')

    # generate exposure spectra for select CMX sky surface brightnesses with
    # exposure times scaled by (1) sky ratio at 5000A (2) sky ratio at 6500A
    for expid in expids:
        print('--- expid = %i ---' % expid) 
        is_exp = (skies['expid'] == expid) 
        # get median sky ratios for the exposure 
        fexp_b = np.median(skies['sky_ratio_b'][is_exp]) 
        fexp_r = np.median(skies['sky_ratio_r'][is_exp]) 
        print('  fexp_b = %.2f' % fexp_b) 
        print('  fexp_r = %.2f' % fexp_r) 
        # get median sky surface brightnesses for exposure 
        Isky = bs_coadd(
                [skies['wave_b'], skies['wave_r'], skies['wave_z']], 
                [
                    np.median(skies['sky_sb_b'][is_exp], axis=0), 
                    np.median(skies['sky_sb_r'][is_exp], axis=0), 
                    np.median(skies['sky_sb_z'][is_exp], axis=0)]
                )
    
        # generate exposure spectra for expid CMX sky  
        _fspec_b = os.path.join(dir, 'exp_spectra.exp%i.fexp_b.fits' % expid)
        _fspec_r = os.path.join(dir, 'exp_spectra.exp%i.fexp_r.fits' % expid)
        spec_b = exp_spectra(
                Isky,           # sky surface brightness 
                150. * fexp_b,  # exposure time 
                1.1,            # same airmass 
                _fspec_b    
                )
        spec_r = exp_spectra(
                Isky, 
                150. * fexp_r, 
                1.1,
                _fspec_r
                )
        # run redrock on the exposure spectra 
        frr_b = run_redrock(_fspec_b, overwrite=False)
        frr_r = run_redrock(_fspec_r, overwrite=False)

        # plot comparing the exp spectra to the nominal dark spectra 
        fig = plt.figure(figsize=(15,5)) 
        sub = fig.add_subplot(111)
        sub.plot(Isky[0], Isky[1], c='C0', lw=0.5) 
        for band in ['b', 'r', 'z']: 
            sub.scatter(spec_nom.wave[band], spec_nom.flux[band][0,:], c='k', s=1) 
        for band in ['b', 'r', 'z']: 
            sub.scatter(spec_b.wave[band], spec_b.flux[band][0,:], c='C0', s=1) 
        for band in ['b', 'r', 'z']: 
            sub.scatter(spec_r.wave[band], spec_r.flux[band][0,:], c='C1', s=1) 
        sub.set_xlabel('wavelength', fontsize=20) 
        sub.set_xlim(3.6e3, 9.8e3) 
        sub.set_ylabel('flux', fontsize=20) 
        fig.savefig(_fspec_b.replace('fexp_b.fits', 'fexp_br.png'),
                bbox_inches='tight') 
        plt.close() 
    raise ValueError
    
    _, _, meta = source_spectra() 
    ztrue = meta['zred']  # true redshifts 
    r_mag = meta['r_mag'] 
    dchi2 = 40. # minimum delta chi2

    # read redrock outputs and compare which exposure factor does better
    # at reproducing the nomimal dark exposure redshift success rate. 
    rr_nom = fitsio.read(frr_nom) 
    zs_nom = UT.zsuccess(rr_nom['Z'], ztrue, rr_nom['ZWARN'],
            deltachi2=rr_nom['DELTACHI2'], min_deltachi2=dchi2)
    print('nominal z-success = %.2f' % (np.sum(zs_nom)/float(len(zs_nom))))

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)

    wmean, rate, err_rate = UT.zsuccess_rate(r_mag, zs_nom, range=[15,22], 
            nbins=28, bin_min=10) 
    sub.errorbar(wmean, rate, err_rate, fmt='.k', elinewidth=2, markersize=10)

    zs_b, zs_r = [], []  
    for expid in expids:
        rr_b = fitsio.read(os.path.join(dir, 'zbest.exp_spectra.exp%i.fexp_b.fits')) 
        rr_r = fitsio.read(os.path.join(dir, 'zbest.exp_spectra.exp%i.fexp_r.fits')) 

        _zs_b = UT.zsuccess(rr_b['Z'], ztrue, rr_b['ZWARN'],
                deltachi2=rr_b['DELTACHI2'], min_deltachi2=dchi2)
        _zs_r = UT.zsuccess(rr_r['Z'], ztrue, rr_r['ZWARN'],
                deltachi2=rr_r['DELTACHI2'], min_deltachi2=dchi2)
        zs_b.append(_zs_b)
        zs_r.append(_zs_r) 
        print('  fexp_b z-success = %.2f' % (np.sum(_zs_b)/float(len(_zs_b))))
        print('  fexp_r z-success = %.2f' % (np.sum(_zs_r)/float(len(_zs_r))))

        wmean, rate, err_rate = UT.zsuccess_rate(r_mag, _zs_b, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.plot(wmean, rate, c='C0')
        wmean, rate, err_rate = UT.zsuccess_rate(r_mag, _zs_r, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.plot(wmean, rate, c='C1')

    zs_b = np.concatenate(zs_b) 
    zs_r = np.concatenate(zs_r) 
    print('-----------------------')
    print('nominal z-success = %.2f' % (np.sum(zs_nom)/float(len(zs_nom))))
    print('fexp_b z-success = %.2f ' % (np.sum(zs_b)/float(len(zs_b))))
    print('fexp_r z-success = %.2f ' % (np.sum(zs_r)/float(len(zs_r))))

    sub.set_xlabel(r'Legacy $r$ magnitude', fontsize=20)
    sub.set_xlim([16., 21.]) 
    sub.set_ylabel(r'redrock $z$ success rate', fontsize=20)
    sub.set_ylim([0.6, 1.1])
    sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
    fig.savefig(os.path.join(dir, 'zsuccess.exp_spectra.fexp_br.png'),
            bbox_inches='tight') 
    return None 


def cmx_skies(): 
    ''' read in CMX sky data. The sky surface brightnesses are generated
    from the flat fielded sky data that's throughput corrected. 
    '''
    fskies = h5py.File('/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/sky_fibers.cmx.v1.hdf5', 'r')
    skies = {}
    for k in fskies.keys(): 
        skies[k] = fskies[k][...]
    return skies


def source_spectra(): 
    ''' read GAMA-matched fiber-magnitude scaled BGS source spectra 
    These source spectra are created for GAMA objects. their spectra is 
    constructed from continuum that's template matched to the broadband
    colors and emission lines from GAMA data (properly flux calibrated). 
    Then the spectra is scaled down to the r-band fiber magnitude. They 
    therefore do not require fiber acceptance fractions. 
    '''
    # read in source
    fsource = h5py.File(os.path.join(dir, 'GALeg.g15.sourceSpec.1000.seed0.hdf5'), 'r')
    wave_s = fsource['wave'][...]
    flux_s = fsource['flux'][...]
    
    meta = {} 
    for k in ['r_mag_apflux', 'r_mag_gama', 'zred', 'absmag_ugriz']: 
        meta[k] = fsource[k][...]
    meta['r_mag'] = UT.flux2mag(fsource['legacy-photo']['flux_r'][...], method='log') 

    fsource.close()
    return wave_s, flux_s, meta


def nomdark_spectra(): 
    ''' spectra observed during nominal dark sky for 150s. This will
    serve as the reference spectra for a number of tests. 
    '''
    fexp = os.path.join(dir, 'exp_spectra.nominal_dark.150s.fits') 

    if os.path.isfile(fexp): 
        bgs = desispec.io.read_spectra(fexp) 
    else: 
        import desisim.simexp
        from desimodel.io import load_throughput

        # read nominal dark sky surface brightness
        wavemin = load_throughput('b').wavemin - 10.0
        wavemax = load_throughput('z').wavemax + 10.0
        wave = np.arange(round(wavemin, 1), wavemax, 0.8) * u.Angstrom

        config = desisim.simexp._specsim_config_for_wave(wave.to('Angstrom').value, dwave_out=0.8, specsim_config_file='desi')

        nominal_surface_brightness_dict = config.load_table(config.atmosphere.sky, 'surface_brightness', as_dict=True)
        Isky = [wave, nominal_surface_brightness_dict['dark']] 
        
        # read in source spectra 
        wave_s, flux_s, _ = source_spectra() 

        # simulate the exposures and save to file 
        fdesi = FM.fakeDESIspec()
        bgs = fdesi.simExposure(
                wave_s, 
                flux_s, 
                exptime=150., 
                airmass=1.1, 
                Isky=Isky, 
                filename=fexp) 
    return bgs 


def exp_spectra(Isky, exptime, airmass, fexp): 
    ''' spectra observed at the specified 
    - sky surface brightness 
    - exposure time 
    - airmass 
    '''
    if os.path.isfile(fexp): 
        bgs = desispec.io.read_spectra(fexp) 
    else: 
        import desisim.simexp
        from desimodel.io import load_throughput

        # read in source spectra 
        wave_s, flux_s, _ = source_spectra() 

        # simulate the exposures and save to file 
        fdesi = FM.fakeDESIspec()
        bgs = fdesi.simExposure(
                wave_s, 
                flux_s, 
                exptime=exptime, 
                airmass=airmass, 
                Isky=Isky, 
                filename=fexp) 
    return bgs 


def run_redrock(fspec, overwrite=False): 
    ''' run redrock on given spectra file
    '''
    frr = os.path.join(os.path.dirname(fspec), 
            'redrock.%s' % os.path.basename(fspec).replace('.fits', '.h5')) 
    fzb = os.path.join(os.path.dirname(fspec), 
            'zbest.%s' % os.path.basename(fspec)) 

    if not os.path.isfile(fzb) or overwrite: 
        print('running redrock on %s' % os.path.basename(fspec))
        script = '\n'.join([
            "#!/bin/bash", 
            "#SBATCH -N 1", 
            "#SBATCH -C haswell", 
            "#SBATCH -q debug", 
            '#SBATCH -J rr_%s' % os.path.basename(fspec)[:5],
            '#SBATCH -o _rr_%s.o' % os.path.basename(fspec)[:5],
            "#SBATCH -t 00:30:00", 
            "", 
            "export OMP_NUM_THREADS=1", 
            "export OMP_PLACES=threads", 
            "export OMP_PROC_BIND=spread", 
            "", 
            "", 
            "conda activate desi", 
            "", 
            "srun -n 32 -c 2 --cpu-bind=cores rrdesi_mpi -o %s -z %s %s" % (frr, fzb, fspec), 
            ""]) 
        # create the script.sh file, execute it and remove it
        f = open('script.slurm','w')
        f.write(script)
        f.close()

        os.system('sbatch script.slurm') 
        os.system('rm script.slurm') 
    return fzb 


def bs_coadd(waves, sbrights): 
    ''' bullshit hack to combine wavelengths and surface brightnesses of the 3
    cameras...
    '''
    from scipy.interpolate import interp1d
    from desimodel.io import load_throughput
    # read nominal dark sky surface brightness
    wavemin = load_throughput('b').wavemin - 10.0
    wavemax = load_throughput('z').wavemax + 10.0
    outwave = np.arange(round(wavemin, 1), wavemax, 0.8) 
    
    sbrights_interp = [] 
    for wave, sbright in zip(waves, sbrights): 
        fintrp = interp1d(wave, sbright, fill_value=0., bounds_error=False) 
        sbrights_interp.append(fintrp(outwave))
    
    outsbright = np.amax(sbrights_interp, axis=0) 
    return outwave, outsbright 


if __name__=="__main__": 
    texp_factor_wavelength()
