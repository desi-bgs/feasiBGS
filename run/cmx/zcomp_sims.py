'''


script for generating redshift completeness simulations directly using 
CMX sky data rather than sky model 


'''
import os 
import h5py 
import glob 
import fitsio 
import numpy as np 
import astropy.units as u 
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import skymodel as Sky 
from feasibgs import catalogs as Cat
from feasibgs import forwardmodel as FM 
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


if 'NERSC_HOST' in os.environ: 
    dir_srp = '/global/cfs/cdirs/desi/users/chahah/cmx/zcomp_sims/'
    dir_cmx = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'
    dir_zcomp = '/global/cfs/cdirs/desi/users/chahah/cmx/zcomp_sims/'
    dir_coadd = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'
else: 
    dir_srp = os.path.join(UT.dat_dir(), 'srp')
    dir_cmx = '/Users/ChangHoon/data/feasiBGS/cmx/'
    dir_zcomp = '/Users/ChangHoon/data/feasiBGS/cmx/zcomp_sims/'


def GALeg_G15_noisySpec5000(): 
    ''' Construct BGS spectral simulations for 5000 galaxies in the GAMA G15
    field using CMX sky brightness measurements. This is to validate the z
    completeness simulations 

    :param t_fid: 
        fiducial dark exposure time in seconds. This exposure time is scaled up 
        by the ETC to calculate all the exposure times. (default: 150)
    '''
    # read in CMX BGS exposures
    exps = cmx_exposures()
    tileid      = exps['tileid']
    date        = exps['date']
    expid       = exps['expid']
    airmass     = exps['airmass']
    moon_ill    = exps['moon_ill']
    moon_alt    = exps['moon_alt']
    moon_sep    = exps['moon_sep']
    sun_alt     = exps['sun_alt']
    sun_sep     = exps['sun_sep']
    transp      = exps['transparency']
    texp        = exps['exptime'] 
    n_sample    = len(airmass) 
    
    # assume seeing is 1.
    seeing      = np.ones(n_sample) 

    # read in sky brightness 
    wave_sky    = exps['wave']
    u_sb        = 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second
    sky_sbright = exps['sky']

    print('%i BGS CMX exposures' % n_sample)
    
    # read in noiseless spectra
    specfile = os.path.join(dir_srp, 'GALeg.g15.sourceSpec.5000.seed0.hdf5')
    fspec = h5py.File(specfile, 'r') 
    wave = fspec['wave'][...]
    flux = fspec['flux'][...] 
    
    #for iexp in np.random.choice(np.arange(n_sample), size=5, replace=False):
    for iexp in np.arange(n_sample):
        _fexp = os.path.join(dir_zcomp,
                'bgs_cmx.%i-%i-%i.GALeg.g15.5000.seed0.fits' % 
                (tileid[iexp], date[iexp], expid[iexp]))
        if os.path.isfile(_fexp): continue 
        print('--- constructing %s ---' % _fexp) 
        print('t_exp=%.f' % texp[iexp])
        print('airmass=%.2f' % airmass[iexp])
        print('moon ill=%.2f alt=%.f, sep=%.f' % (moon_ill[iexp], moon_alt[iexp], moon_sep[iexp]))
        print('sun alt=%.f, sep=%.f' % (sun_alt[iexp], sun_sep[iexp]))
        print('seeing=%.2f, transp=%.2f' % (seeing[iexp], transp[iexp]))
        
        # iexp-th sky spectra 
        Isky = [wave_sky, sky_sbright[iexp]]

        # simulate the exposures 
        fdesi = FM.fakeDESIspec()
        bgs = fdesi.simExposure(wave, flux, exptime=texp[iexp], airmass=airmass[iexp], Isky=Isky, filename=_fexp) 

        # --- some Q/A plots --- 
        fig = plt.figure(figsize=(10,20))
        sub = fig.add_subplot(411) 
        sub.plot(wave_sky, sky_sbright[iexp], c='C1') 
        sub.text(0.05, 0.95, 
                'texp=%.f, airmass=%.2f\nmoon ill=%.2f, alt=%.f, sep=%.f\nsun alt=%.f, sep=%.f\nseeing=%.1f, transp=%.1f' % 
                (texp[iexp], airmass[iexp], moon_ill[iexp], moon_alt[iexp], moon_sep[iexp], 
                    sun_alt[iexp], sun_sep[iexp], seeing[iexp], transp[iexp]), 
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
    return None 


def run_redrock(clobber=False): 
    ''' run redrock on spectral simulation generated from
    GALeg_G15_noisySpec5000() above. 
    '''
    # read in CMX BGS exposures
    exps = cmx_exposures()
    tileids     = exps['tileid']
    dates       = exps['date']
    expids      = exps['expid']
    n_sample    = len(expids) 
    
    for tileid, date, expid in zip(tileids, dates, expids): 
        fspec = os.path.join(dir_zcomp, 
                'bgs_cmx.%i-%i-%i.GALeg.g15.5000.seed0.fits' % 
                (tileid, date, expid))
        frr     = os.path.join(dir_zcomp, 
                'redrock.bgs_cmx.%i-%i-%i.GALeg.g15.5000.seed0.h5' % 
                (tileid, date, expid))
        fzbest  = os.path.join(dir_zcomp, 
                'zbest.bgs_cmx.%i-%i-%i.GALeg.g15.5000.seed0.fits' % 
                (tileid, date, expid))
        if os.path.isfile(fzbest) and not clobber: 
            continue 
        print('running redrock on %s' % os.path.basename(fspec))
        script = '\n'.join([
            "#!/bin/bash", 
            "#SBATCH -N 1", 
            "#SBATCH -C haswell", 
            "#SBATCH -q debug", 
            '#SBATCH -J rr_%i' % expid,
            '#SBATCH -o _rr_%i.o' % expid,
            "#SBATCH -t 00:30:00", 
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


def cmx_exposures(): 
    ''' read CMX sky fibers and return median sky surface brightness
    measurements for each exposure 
    '''
    # desi fiber area in arcsec^2
    desi_fiber_area = (1.46/2.)**2 * np.pi

    fsky = h5py.File(os.path.join(dir_cmx, 
        'sky_fibers.coadd_gfa.minisv2_sv0.hdf5'), 'r')

    sky_data = {}
    for k in fsky.keys():
        sky_data[k] = fsky[k][...]
        
    #bad_seeing = (sky_data['tileid'] == 70502) | (sky_data['date'] == 20200314) #bad seeing on Feb 25 and 27

    #exp_cuts = ~bad_seeing

    #for k in sky_data.keys(): 
    #    if 'wave' not in k: 
    #        sky_data[k] = sky_data[k][exp_cuts]
    #    else:
    #        sky_data[k] = sky_data[k]
       
    uniq_exps, i_uniq = np.unique(sky_data['expid'], return_index=True)

    # compile median observing conditions for each unique exposure and
    # get the median sky fluxes of all sky fibers
    sky_uniq_exps = {} 
    for k in ['tileid', 'date', 'expid', 'airmass', 'moon_ill', 'moon_alt',
            'moon_sep', 'sun_alt', 'sun_sep', 'exptime', 'transparency', 
            'transp_min', 'transp_max']: 
        sky_uniq_exps[k] = np.zeros(len(uniq_exps))

    sky_uniq_exps['wave_b'] = sky_data['wave_b']
    sky_uniq_exps['wave_r'] = sky_data['wave_r']
    sky_uniq_exps['wave_z'] = sky_data['wave_z']
    wave_sort = np.argsort(np.concatenate([sky_data['wave_b'],
        sky_data['wave_r'], sky_data['wave_z']]))
    sky_uniq_exps['wave'] = np.concatenate([sky_data['wave_b'],
        sky_data['wave_r'], sky_data['wave_z']])[wave_sort]
    
    sky_uniq_exps['sky_b'] = np.zeros((len(uniq_exps), len(sky_data['wave_b'])))
    sky_uniq_exps['sky_r'] = np.zeros((len(uniq_exps), len(sky_data['wave_r'])))
    sky_uniq_exps['sky_z'] = np.zeros((len(uniq_exps), len(sky_data['wave_z'])))
    sky_uniq_exps['sky'] = np.zeros((len(uniq_exps),
        len(sky_uniq_exps['wave'])))
    
    print('date \t\t tile \t exp \t airmass \t moon_ill \t moon_alt \t moon_sep')
    for _i, _i_uniq, _exp in zip(range(len(i_uniq)), i_uniq, uniq_exps): 
        _is_exp = (sky_data['expid'] == _exp)
        
        sky_uniq_exps['tileid'][_i] = sky_data['tileid'][_is_exp][0]
        sky_uniq_exps['date'][_i]   = sky_data['date'][_is_exp][0]
        sky_uniq_exps['expid'][_i]  = _exp.astype(int)
        for k in ['airmass', 'moon_ill', 'moon_alt', 'moon_sep', 'sun_alt', 'sun_sep', 'exptime', 
                'transparency', 'transp_min', 'transp_max']:
            sky_uniq_exps[k][_i] = np.median(sky_data[k][_is_exp])
        
        sky_uniq_exps['sky_b'][_i] = np.median(sky_data['sky_b'][_is_exp], axis=0) / desi_fiber_area
        sky_uniq_exps['sky_r'][_i] = np.median(sky_data['sky_r'][_is_exp], axis=0) / desi_fiber_area
        sky_uniq_exps['sky_z'][_i] = np.median(sky_data['sky_z'][_is_exp], axis=0) / desi_fiber_area
        sky_uniq_exps['sky'][_i] = np.concatenate([sky_uniq_exps['sky_b'][_i],
            sky_uniq_exps['sky_r'][_i], sky_uniq_exps['sky_z'][_i]])[wave_sort]

        print('%i \t %i \t %i \t %.2f \t\t %.2f \t\t %.1f \t\t %f' % 
                (sky_data['date'][_i_uniq], sky_data['tileid'][_i_uniq], sky_data['expid'][_i_uniq], 
               sky_uniq_exps['airmass'][_i], sky_uniq_exps['moon_ill'][_i], 
               sky_uniq_exps['moon_alt'][_i], sky_uniq_exps['moon_sep'][_i]))
    return sky_uniq_exps


def zsuccess(deltachi2=40.): 
    ''' compare the redshift completeness of the completeness simulations
    versus the actual exposures 
    '''
    ###########################################################################
    # get true redshifts and magnitude of the completness sims
    ###########################################################################
    specfile = os.path.join(dir_srp, 'GALeg.g15.sourceSpec.5000.seed0.hdf5')
    fspec = h5py.File(specfile, 'r') 
    ztrue = fspec['zred'][...]
    r_mag = UT.flux2mag(fspec['legacy-photo']['flux_r'][...], method='log') 

    ###########################################################################
    # read in CMX BGS exposures
    ###########################################################################
    exps = cmx_exposures()
    tileids     = exps['tileid']
    dates       = exps['date']
    expids      = exps['expid'].astype(int)
    n_sample    = len(expids) 
    
    ###########################################################################
    # loop through expsoures and compare z success of sims to exposures 
    ###########################################################################
    for _i, tileid, date, expid in zip(range(n_sample), tileids, dates, expids): 
        frr_sim = os.path.join(dir_zcomp, 
                'zbest.bgs_cmx.%i-%i-%i.GALeg.g15.5000.seed0.fits' % 
                (tileid, date, expid))
        if not os.path.isfile(frr_sim): 
            print('... no %s' % os.path.basename(frr_sim))
            continue 

        rr_sim = fitsio.read(frr_sim) # read redrock file of sim 
        
        # get spectographs for exposure 
        fcoadds = glob.glob(os.path.join(dir_cmx, 'coadd-%i-%i-*-%s.fits' % 
            (tileid, date, str(expid).zfill(8))))
        n_specs = [os.path.basename(fcoadd).split('-')[3] for fcoadd in fcoadds]
        if len(n_specs) == 0: 
            print('no coadds for %i, %i, %s' % (tileid, date,
                str(expid).zfill(8)))
            continue 

        r_mag_cmx, ztrue_cmx, rrock_cmx = [], [], [] 
        for n_spec in n_specs: 
            # compile rmag or w.e. from coadds
            fcoadd = os.path.join(dir_cmx, 
                    'coadd-%i-%i-%s-%s.fits' % 
                    (tileid, date, n_spec, str(expid).zfill(8)))
            coadd = fitsio.read(fcoadd)
            r_mag_cmx.append(22.5 - 2.5 * np.log10(coadd['FLUX_R']))

            # compile redshift truths
            fztrue = os.path.join(dir_cmx, 
                    'ztrue-%i-%i-%s-%s.hdf5' % 
                    (tileid, date, n_spec, str(expid).zfill(8)))
            zt = h5py.File(fztrue, 'r') 
            ztrue_cmx.append(zt['ztrue'][...])
            # compile redrock fits
            frrock = os.path.join(dir_cmx, 
                    'zbest-%i-%i-%s-%s.fits' % 
                    (tileid, date, n_spec, str(expid).zfill(8)))
            rrock_cmx.append(fitsio.read(frrock))

        r_mag_cmx = np.concatenate(r_mag_cmx)
        ztrue_cmx = np.concatenate(ztrue_cmx)
        rrock_cmx = np.concatenate(rrock_cmx, axis=0) 

        has_zt = (ztrue_cmx != -999.)
        print('%i %i %i' % (tileid, date, expid))
        print('     %i of %i targets with true redshifts' % (np.sum(has_zt),
            len(ztrue_cmx)))
        #######################################################################
        fig = plt.figure(figsize=(5,5))
        sub = fig.add_subplot(111) 
        sub.plot([15., 22.], [1., 1.], c='k', ls='--', lw=2)
        # completeness sim z-success 
        _zsuc   = UT.zsuccess(rr_sim['Z'], ztrue, rr_sim['ZWARN'],
                deltachi2=rr_sim['DELTACHI2'], min_deltachi2=deltachi2)
        wmean, rate, err_rate = UT.zsuccess_rate(r_mag, _zsuc, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.errorbar(wmean, rate, err_rate, fmt='.C0', elinewidth=2, 
                markersize=10)
        # CMX z-success 
        _zsuc   = UT.zsuccess(rrock_cmx['Z'][has_zt], ztrue_cmx[has_zt],
                rrock_cmx['ZWARN'][has_zt], deltachi2=rrock_cmx['DELTACHI2'][has_zt],
                min_deltachi2=deltachi2)
        wmean, rate, err_rate = UT.zsuccess_rate(r_mag_cmx[has_zt], _zsuc, 
                range=[15,22], nbins=28, bin_min=10) 
        sub.errorbar(wmean, rate, err_rate, fmt='.k', elinewidth=2, 
                markersize=10)
        sub.vlines(19.5, 0., 1.2, color='k', linestyle=':', linewidth=1)
       
        exp_info = '\n'.join([
            'tile=%i, date=%i, exp=%i' % (tileid, date, expid),
            'airmass=%.1f' % exps['airmass'][_i], 
            'moon ill=%.2f, moon alt=%.f, moon sep=%.f' % 
            (exps['moon_ill'][_i], exps['moon_alt'][_i], exps['moon_sep'][_i]), 
            r'$t_{\rm exp}= %.f$, %.2f < transparency < %.2f' % 
            (exps['exptime'][_i], exps['transp_min'][_i], exps['transp_max'][_i])])

        sub.text(0.05, 0.05, exp_info, ha='left', va='bottom',
                transform=sub.transAxes, fontsize=10)
        
        sub.text(0.95, 0.95, r'$\Delta \chi^2 = %.f$' % deltachi2, ha='right',
                va='top', transform=sub.transAxes, fontsize=15)

        sub.set_xlabel(r'Legacy $r$ magnitude', fontsize=20)
        sub.set_xlim([16., 21.]) 
        sub.set_ylabel(r'redrock $z$ success rate', fontsize=20)
        sub.set_ylim([0.6, 1.1])
        sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
        fig.savefig(os.path.join(dir_zcomp, 
            'zsuccess.%i-%i-%s.zcomp_sim.png' % 
            (tileid, date, str(expid).zfill(8))), bbox_inches='tight') 
        plt.close() 
    return None 


def compare_spectra(): 
    ''' compare completeness simulations constructed from real sky flux to
    coadd spectra of similar color and magnitude 
    '''
    # read in CMX BGS exposures
    exps = cmx_exposures()
    tileid      = exps['tileid']
    date        = exps['date']
    expid       = exps['expid']
    airmass     = exps['airmass']
    moon_ill    = exps['moon_ill']
    moon_alt    = exps['moon_alt']
    moon_sep    = exps['moon_sep']
    sun_alt     = exps['sun_alt']
    sun_sep     = exps['sun_sep']
    transp      = exps['transparency']
    texp        = exps['exptime'] 
    n_sample    = len(airmass) 
    
    # assume seeing is 1.
    seeing      = np.ones(n_sample) 

    # read in sky brightness 
    wave_sky    = exps['wave']
    sky_sbright = exps['sky'] # 1e-17 * u.erg / u.angstrom / u.arcsec**2 / u.cm**2 / u.second

    print('%i BGS CMX exposures' % n_sample)
    
    # read in noiseless spectra
    specfile = os.path.join(dir_srp, 'GALeg.g15.sourceSpec.5000.seed0.hdf5')
    fspec = h5py.File(specfile, 'r') 
    wave = fspec['wave'][...]
    flux = fspec['flux'][...] 
    gmag = UT.flux2mag(fspec['legacy-photo']['flux_g'][...], method='log') 
    rmag = UT.flux2mag(fspec['legacy-photo']['flux_r'][...], method='log') 

    # pick a galaxy with r~19.
    r19 = np.arange(len(rmag))[(rmag > 18.9) & (rmag < 19.1)]
    i_rand = np.random.choice(r19, replace=False) 
    i_color = gmag[i_rand] - rmag[i_rand]
    i_rmag = rmag[i_rand]
    
    for iexp in np.arange(n_sample):
        # read in coadds
        fcoadds = glob.glob(os.path.join(dir_coadd,
            'coadd-%i-%i-*-%s.fits' % 
            (tileid[iexp], date[iexp], str(int(expid[iexp])).zfill(8))))
        
        for i, fcoadd in enumerate(fcoadds): 
            coadd = fitsio.read(fcoadd)
            coadd_wave = fitsio.read(fcoadd, ext=2)
            coadd_flux = fitsio.read(fcoadd, ext=3) 
            if i == 0: 
                coadds = coadd
                coadds_flux = coadd_flux
            else:
                coadds = np.concatenate([coadds, coadd]) 
                coadds_flux = np.concatenate([coadds_flux, coadd_flux]) 

        # pick similar brightness and color galaxies 
        coadds_gmag = UT.flux2mag(coadds['FLUX_G'], method='log') 
        coadds_rmag = UT.flux2mag(coadds['FLUX_R'], method='log') 
        
        similar = np.arange(len(coadds))[(
            (coadds_rmag > i_rmag - 0.05) & 
            (coadds_rmag < i_rmag + 0.05) & 
            (coadds_gmag - coadds_rmag > i_color - 0.05) & 
            (coadds_gmag - coadds_rmag < i_color + 0.05) & 
            (np.sum(coadds_flux, axis=1) > 0))]
        assert np.sum(similar) > 0 
        i_coadd = np.random.choice(similar, replace=False)  
        print(i_rmag, i_color) 
        print(coadds_rmag[i_coadd], 
                coadds_gmag[i_coadd] - coadds_rmag[i_coadd])

        # read completeness sim 
        fsim = os.path.join(dir_zcomp,
                'bgs_cmx.%i-%i-%i.GALeg.g15.5000.seed0.fits' % 
                (tileid[iexp], date[iexp], expid[iexp]))
        sim = UT.readDESIspec(fsim)

        # --- some Q/A plots --- 
        fig = plt.figure(figsize=(10,10))
        # --- plot sky brightness --- 
        sub = fig.add_subplot(211) 
        sub.plot(wave_sky, sky_sbright[iexp], c='C1') 
        sub.text(0.05, 0.95, 
                'texp=%.f, airmass=%.2f\nmoon ill=%.2f, alt=%.f, sep=%.f\nsun alt=%.f, sep=%.f\nseeing=%.1f, transp=%.1f' % 
                (texp[iexp], airmass[iexp], moon_ill[iexp], moon_alt[iexp], moon_sep[iexp], 
                    sun_alt[iexp], sun_sep[iexp], seeing[iexp], transp[iexp]), 
                ha='left', va='top', transform=sub.transAxes, fontsize=15)
        sub.legend(loc='upper right', frameon=True, fontsize=20) 
        sub.set_xlim([3e3, 1e4]) 
        sub.set_ylim([0., 20.]) 

        # --- compare spectra --- 
        sub = fig.add_subplot(212)
        sub.plot(coadd_wave, coadds_flux[i_coadd], c='k', label='cmx spectra') 
        for band in ['b', 'r', 'z']: 
            sub.plot(sim['wave_%s' % band], sim['flux_%s' % band][i_rand],
                    c='C1', label='sim') 
        sub.plot(wave, flux[i_rand], c='k', ls=':', lw=1)
        sub.set_xlim([3e3, 1e4]) 
        sub.set_ylim([0., 15.]) 
        bkgd = fig.add_subplot(111, frameon=False) 
        bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        bkgd.set_xlabel('rest-frame wavelength [Angstrom]', labelpad=10, fontsize=25) 
        bkgd.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', labelpad=10, fontsize=25) 
        fig.savefig(fsim.replace('.fits', '.comparison.png'), bbox_inches='tight') 
    return None 



if __name__=="__main__": 
    #GALeg_G15_noisySpec5000()
    #run_redrock()
    #zsuccess()
    compare_spectra()
