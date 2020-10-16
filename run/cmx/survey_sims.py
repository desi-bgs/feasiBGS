'''


validate survey simulations using CMX data. 


updates
-------
* 5/19/2020: created script and test to compare which wavelength range I should
  use for the exposure time correction factor 
'''
import os 
import h5py 
import fitsio
import numpy as np 
import astropy.units as u 
# -- feasibgs --
from feasibgs import util as UT
from feasibgs import catalogs as Cat
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


def validate_spectral_pipeline(): 
    ''' validate the spectral pipeline by 
    1. constructing spectra from fiber acceptance fraction scaled smoothed CMX
       spectra with CMX sky surface brightness 
    2. compare noise levels to CMX observations 
    '''
    from scipy.signal import medfilt
    import desisim.simexp
    import specsim.instrument
    from desitarget.cmx import cmx_targetmask

    np.random.seed(0) 

    tileid = 70502
    date = 20200225
    expid = 52113
    ispec = 0
    
    dir_gfa = '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions'
    dir_redux = "/global/cfs/cdirs/desi/spectro/redux/daily"
    dir_coadd = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'
    
    # get sky surface brightness by correcting for the throughput on the CMX
    # sky data 
    f_sky = lambda band: os.path.join(dir_redux, 
            'exposures', str(date), str(expid).zfill(8),
            'sky-%s%i-%s.fits' % (band, ispec, str(expid).zfill(8)))
    sky_b = desispec.io.read_sky(f_sky('b')) 
    sky_r = desispec.io.read_sky(f_sky('r')) 
    sky_z = desispec.io.read_sky(f_sky('z')) 

    wave, sky_electrons = bs_coadd(
            [sky_b.wave, sky_r.wave, sky_z.wave], 
            [sky_b.flux, sky_r.flux, sky_z.flux]) 

    # exposure time
    _frame = desispec.io.read_frame(f_sky('b').replace('sky-', 'frame-'))
    exptime = _frame.meta['EXPTIME']
    print('exp.time = %.fs' % exptime) 

    # get which are good fibers from coadd file
    f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
    coadd = fitsio.read(f_coadd)

    is_good = (coadd['FIBERSTATUS'] == 0)
    is_sky  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SKY')) != 0
    good_sky = is_good & is_sky
    
    # get throughput for the cameras 
    config = desisim.simexp._specsim_config_for_wave(wave, dwave_out=0.8, specsim_config_file='desi')
    instrument = specsim.instrument.initialize(config, True)
    throughput = np.amax([instrument.cameras[0].throughput, instrument.cameras[1].throughput, instrument.cameras[2].throughput], axis=0)

    desi_fiber_area = 1.862089 # fiber area 

    # calculate sky brightness
    sky_bright = np.median(sky_electrons[good_sky,:], axis=0) / throughput / instrument.photons_per_bin / exptime * 1e17

    # get fiber acceptance fraction and airmass  
    gfa = fitsio.read(os.path.join(dir_gfa,
        'offline_all_guide_ccds_thru_20200315.fits')) 
    isexp = (gfa['EXPID'] == expid)
    fibloss = gfa['TRANSPARENCY'][isexp] * gfa['FIBER_FRACFLUX'][isexp]
    fibloss = np.median(fibloss[~np.isnan(fibloss)])
    print('fiber loss = (TRANSP) x (FFRAC) = %f' % fibloss) 
    airmass = np.median(gfa['AIRMASS'][isexp]) 
    print('airmass = %.2f' % airmass) 

    # select BGS spectra
    coadd_wave = fitsio.read(f_coadd, ext=2)
    coadd_flux = fitsio.read(f_coadd, ext=3)
    coadd_ivar = fitsio.read(f_coadd, ext=4)

    is_BGS  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
    gal_cut = is_BGS & (np.sum(coadd_flux, axis=1) != 0)
    igals = np.random.choice(np.arange(len(gal_cut))[gal_cut], size=5,
            replace=False)

    igals = np.arange(len(coadd['FIBER']))[coadd['FIBER'] == 143]
    
    for igal in igals: 
        # source flux is the smoothed CMX spetra
        source_flux = np.clip(np.interp(wave, coadd_wave,
            medfilt(coadd_flux[igal,:], 101)), 0, None) 

        # simulate the exposures using the spectral simulation pipeline  
        fdesi = FM.fakeDESIspec()
        bgs = fdesi.simExposure(
                wave, 
                np.atleast_2d(source_flux * fibloss), # scale by fiber acceptance fraction 
                exptime=exptime, 
                airmass=airmass, 
                Isky=[wave, sky_bright], 
                dwave_out=0.8, 
                filename=None) 
    
        # barebone specsim pipeline for comparison 
        from specsim.simulator import Simulator
        desi = Simulator(config, num_fibers=1)
        desi.observation.exposure_time = exptime * u.s
        desi.atmosphere._surface_brightness_dict[desi.atmosphere.condition] = \
                np.interp(desi.atmosphere._wavelength, wave, sky_bright) * \
                desi.atmosphere.surface_brightness.unit
        desi.atmosphere._extinct_emission = False
        desi.atmosphere._moon = None 
        desi.atmosphere.airmass = airmass # high airmass

        desi.simulate(source_fluxes=np.atleast_2d(source_flux) * 1e-17 * desi.simulated['source_flux'].unit, 
            fiber_acceptance_fraction=np.tile(fibloss,
                np.atleast_2d(source_flux).shape))

        random_state = np.random.RandomState(0)
        desi.generate_random_noise(random_state, use_poisson=True)

        scale=1e17

        waves, fluxes, ivars, ivars_electron = [], [], [], [] 
        for table in desi.camera_output:
            _wave = table['wavelength'].astype(float)
            _flux = (table['observed_flux']+table['random_noise_electrons']*table['flux_calibration']).T.astype(float)
            _flux = _flux * scale

            _ivar = table['flux_inverse_variance'].T.astype(float)
            _ivar = _ivar / scale**2

            waves.append(_wave)
            fluxes.append(_flux[0])
            ivars.append(_ivar[0])

        fig = plt.figure(figsize=(15,10))
        sub = fig.add_subplot(211)
        sub.plot(coadd_wave, coadd_flux[igal,:] * fibloss, c='C0', lw=1,
                label='(coadd flux) x (fib.loss)')
        for i_b, band in enumerate(['b', 'r', 'z']): 
            lbl = None
            if band == 'b': lbl = 'spectral sim.'
            sub.plot(bgs.wave[band], bgs.flux[band][0], c='C1', lw=1,
                    label=lbl)
            sub.plot(waves[i_b], fluxes[i_b] *fibloss, c='C2', lw=1, ls=':')
        sub.plot(wave, source_flux * fibloss, c='k', lw=1, ls='--', 
                label='source flux')
        sub.legend(loc='upper right', frameon=True, fontsize=20)
        sub.set_xlim(3600, 9800)
        sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=25) 
        sub.set_ylim(-1., 5.)
        
        sub = fig.add_subplot(212)
        sub.plot(coadd_wave, coadd_ivar[igal,:] * fibloss**-2, c='C0', lw=1,
                label=r'(coadd ivar) / (fib.loss$)^2$')
        for i_b, band in enumerate(['b', 'r', 'z']): 
            sub.plot(bgs.wave[band], bgs.ivar[band][0], c='C1', lw=1)
            sub.plot(waves[i_b], ivars[i_b] * fibloss**-2, c='C2', lw=1, ls=':')
        sub.legend(loc='upper right', frameon=True, fontsize=20)
        sub.set_xlabel('wavelength [$A$]', fontsize=20)
        sub.set_xlim(3600, 9800)
        sub.set_ylabel('ivar', fontsize=25) 
        sub.set_ylim(0., None)
        fig.savefig(os.path.join(dir, 'valid.spectral_pipeline.exp%i.%i.png' % (expid, igal)),
            bbox_inches='tight') 
    return None 


def validate_spectral_pipeline_GAMA_source(): 
    ''' compare the fiber flux scaled source spectra from spectral simulations
    pipeline to fiber loss corrected cframes CMX data for overlapping GAMA G12
    galaxies.
    '''
    import glob 
    from scipy.signal import medfilt
    from scipy.interpolate import interp1d 
    from desitarget.cmx import cmx_targetmask
    from pydl.pydlutils.spheregroup import spherematch

    np.random.seed(0) 

    tileid  = 70502  #[66014, 70502] #66014 is with low transparency
    date    = 20200225
    expids  = [52112]#, 52113, 52114, 52115, 52116] # terrible FWHM 

    #tileid  = 66014 # low transparency
    #date    = 20200314
    #expids  = [55432]
    
    dir_gfa = '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions'
    dir_redux = "/global/cfs/cdirs/desi/spectro/redux/daily"
    dir_coadd = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'

    # read in GAMA + Legacy catalo g 
    cata = Cat.GamaLegacy()
    g12 = cata.Read('g12', dr_gama=3, dr_legacy=7) 

    g12_ra  = g12['legacy-photo']['ra']
    g12_dec = g12['legacy-photo']['dec']
    Ng12 = len(g12_ra)
    
    # match GAMA galaxies to templates 
    bgs3 = FM.BGStree()
    template_match = bgs3._GamaLegacy(g12)
    hasmatch = (template_match != -999)
    
    # ra/dec cut for GAMA so we only keep ones near the tile
    cut_gama = ((g12_ra > 174.0) & (g12_ra < 186.0) & (g12_dec > -3.0) & (g12_dec < 2.0) & hasmatch)
    g12_ra  = g12_ra[cut_gama]
    g12_dec = g12_dec[cut_gama] 
    g12_z   = g12['gama-spec']['z'][cut_gama] 

    g12_rfib        = UT.flux2mag(g12['legacy-photo']['fiberflux_r'])[cut_gama]
    g12_rmag_gama   = g12['gama-photo']['r_model'][cut_gama] # r-band magnitude from GAMA (SDSS) photometry

    print('%i galaxies in GAMA G12 + Legacy' % len(g12_ra)) 

    # match coadd objects to G12+legacy catalog based on RA and Dec
    for expid in expids: 
        print('--- %i ---' % expid) 
        # get fiber acceptance fraction for exposure from GFA
        gfa = fitsio.read(os.path.join(dir_gfa,
            'offline_all_guide_ccds_thru_20200315.fits')) 
        isexp = (gfa['EXPID'] == expid)

        fwhm = gfa['FWHM_ASEC'][isexp]
        print('  (FWHM) = %f' % np.median(fwhm[~np.isnan(fwhm)]))

        transp = gfa['TRANSPARENCY'][isexp]
        transp = np.median(transp[~np.isnan(transp)])
        print('  (TRANSP) = %f' % transp) 

        fibloss = gfa['TRANSPARENCY'][isexp] * gfa['FIBER_FRACFLUX'][isexp]
        fibloss = np.median(fibloss[~np.isnan(fibloss)])
        print('  fiber loss = (TRANSP) x (FFRAC) = %f' % fibloss) 
    
        # spectrographs available for the exposure
        ispecs = np.sort([int(os.path.basename(fframe).split('-')[1].replace('z', '')) 
                for fframe in glob.glob(os.path.join(dir_redux, 
                    'exposures', str(date), str(expid).zfill(8),
                    'frame-z*.fits'))])
    
        match_gama, coadd_fluxes = [], []
        for ispec in ispecs:  
            # select BGS galaxies from the coadds 
            f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
            coadd = fitsio.read(f_coadd)
            coadd_wave = fitsio.read(f_coadd, ext=2)
            coadd_flux = fitsio.read(f_coadd, ext=3)

            is_BGS  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
            gal_cut = is_BGS & (np.sum(coadd_flux, axis=1) != 0)
            
            # select ones that are in GAMA by matching RA and Dec 
            match = spherematch(g12_ra, g12_dec, 
                    coadd['TARGET_RA'][gal_cut], coadd['TARGET_DEC'][gal_cut], 
                    0.000277778)
            m_gama = match[0] 
            m_coadd = match[1] 

            match_gama.append(m_gama) 
            coadd_fluxes.append(coadd_flux[gal_cut,:][m_coadd])

        match_gama = np.concatenate(match_gama) 
        coadd_fluxes = np.concatenate(coadd_fluxes, axis=0)
        print('  %i matches to G12' % len(match_gama))

        # generate spectra for the following overlapping galaxies
        gama_samp = np.arange(Ng12)[cut_gama][match_gama]

        s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=15000) 

        emline_flux = s_bgs.EmissionLineFlux(g12, index=gama_samp, dr_gama=3, silent=True) # emission lines from GAMA 

        s_flux, s_wave, magnorm_flag = s_bgs.Spectra(
                g12_rfib[match_gama], 
                g12_z[match_gama],
                np.repeat(100.0, len(match_gama)),
                seed=1, 
                templateid=template_match[gama_samp],
                emflux=emline_flux, 
                mag_em=g12_rmag_gama[match_gama]
                )

        igals = np.random.choice(np.arange(len(match_gama))[magnorm_flag], size=5, replace=False)

        fig = plt.figure(figsize=(15,20))

        for i, igal in enumerate(igals):
            sub = fig.add_subplot(5,1,i+1)
            #sub.plot(coadd_wave, medfilt(coadd_fluxes[igal,:], 101), c='k',
            #        ls=':', lw=0.5, label='smoothed (coadd flux)') 

            sub.plot(coadd_wave, coadd_fluxes[igal,:] * transp * 0.775 ,
                    c='C0', lw=0.1) 
            sub.plot(coadd_wave, medfilt(coadd_fluxes[igal,:], 101) * transp * 0.775 , c='C0', 
                    label='(coadd flux) x (TRANSP) x (0.775)') 
            sub.plot(coadd_wave, coadd_fluxes[igal,:] * fibloss, 
                    c='C1', lw=0.1)
            sub.plot(coadd_wave, medfilt(coadd_fluxes[igal,:], 101) * fibloss, c='C1', 
                    label='(coadd flux) x (TRANSP) x (FIBER FRACFLUX)') 

            sub.plot(s_wave, s_flux[igal,:] * transp, c='k', ls='--',
                    label='(sim source flux) x (TRANSP)') 

            sub.set_xlim(3600, 9800)
            if i < 4: sub.set_xticklabels([]) 
            if i == 1: sub.set_ylabel('inciddent flux [$10^{-17} erg/s/cm^2/A$]', fontsize=25) 
            if expid == 55432: 
                sub.set_ylim(-0.5, 3.)
            else: 
                sub.set_ylim(-0.5, 10.)
            #sub.set_ylim(1e-1, None)
            #sub.set_yscale('log') 
        sub.legend(loc='upper right', handletextpad=0.1, fontsize=20)
        sub.set_xlabel('wavelength', fontsize=25) 
        fig.savefig(os.path.join(dir, 
            'valid.spectral_pipeline_source_flux.exp%i.png' % expid), bbox_inches='tight') 
        plt.close() 
    return None 


def validate_spectral_pipeline_source(): 
    ''' compare the color-matched and fiber flux scaled source spectra from the
    spectral simulation to the fiber loss corrected cframes CMX data. This is
    because the GAMA comparison was a bust. 
    '''
    import glob 
    from scipy.signal import medfilt
    from scipy.interpolate import interp1d 
    from scipy.spatial import cKDTree as KDTree
    from desitarget.cmx import cmx_targetmask
    from pydl.pydlutils.spheregroup import spherematch

    np.random.seed(0) 

    tileid = 66003 
    date = 20200315
    expids = [55654, 55655, 55656]
    
    dir_gfa     = '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions'
    dir_redux   = "/global/cfs/cdirs/desi/spectro/redux/daily"
    dir_coadd   = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'

    # read VI redshifts, which will be used for constructing the source spectra
    fvi = os.path.join('/global/cfs/cdirs/desi/sv/vi/TruthTables/',
            'truth_table_BGS_v1.2.csv') 
    vi_id, ztrue, qa_flag = np.genfromtxt(fvi, delimiter=',', skip_header=1, unpack=True, 
            usecols=[0, 2, 3]) 
    good_z = (qa_flag >= 2.5) 
    vi_id = vi_id[good_z].astype(int)
    ztrue = ztrue[good_z]
            
    mbgs = FM.BGStree()

    for expid in expids: 
        print('--- %i ---' % expid) 
        # get fiber acceptance fraction for exposure from GFA
        gfa = fitsio.read(os.path.join(dir_gfa,
            'offline_all_guide_ccds_thru_20200315.fits')) 
        isexp = (gfa['EXPID'] == expid)

        fwhm = gfa['FWHM_ASEC'][isexp]
        print('  (FWHM) = %f' % np.median(fwhm[~np.isnan(fwhm)]))

        transp = gfa['TRANSPARENCY'][isexp]
        transp = np.median(transp[~np.isnan(transp)])
        print('  (TRANSP) = %f' % transp) 

        fibloss = gfa['TRANSPARENCY'][isexp] * gfa['FIBER_FRACFLUX'][isexp]
        fibloss = np.median(fibloss[~np.isnan(fibloss)])
        print('  fiber loss = (TRANSP) x (FFRAC) = %f' % fibloss) 
    
        # spectrographs available for the exposure
        ispecs = np.sort([int(os.path.basename(fframe).split('-')[1].replace('z', '')) 
                for fframe in glob.glob(os.path.join(dir_redux, 
                    'exposures', str(date), str(expid).zfill(8),
                    'frame-z*.fits'))])

        coadd_fluxes, s_fluxes = [], [] 
        for ispec in ispecs:  
            # read coadd file 
            f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
            coadd = fitsio.read(f_coadd)
            coadd_wave = fitsio.read(f_coadd, ext=2)
            coadd_flux = fitsio.read(f_coadd, ext=3)

            is_BGS  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
            gal_cut = is_BGS & (np.sum(coadd_flux, axis=1) != 0)
    
            targetid = coadd['TARGETID'][gal_cut] 
            rmag = UT.flux2mag(coadd['FLUX_R'], method='log')[gal_cut]
            gmag = UT.flux2mag(coadd['FLUX_G'], method='log')[gal_cut]
            rfib = UT.flux2mag(coadd['FIBERFLUX_R'], method='log')[gal_cut]

            _, m_vi, m_coadd = np.intersect1d(vi_id, targetid, return_indices=True) 
            print('  %i matches to VI' % len(m_vi))

            # match to templates 
            temp_rmag = mbgs.meta['SDSS_UGRIZ'].data[:,2]
            temp_gmag = mbgs.meta['SDSS_UGRIZ'].data[:,1]

            temp_meta = np.vstack([ 
                mbgs.meta['Z'].data,
                temp_rmag, 
                temp_gmag - temp_rmag]).T 
            tree = KDTree(temp_meta) 
            
            # match CMX galaxies to templates 
            _, match_temp = tree.query(np.vstack([
                ztrue[m_vi], rmag[m_coadd], (gmag - rmag)[m_coadd]]).T)
            # in some cases there won't be a match from  KDTree.query
            # we flag these with -999 
            has_match = ~(match_temp >= len(mbgs.meta['TEMPLATEID']))

            s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=15000) 
            s_flux, s_wave, magnorm_flag = s_bgs.Spectra(
                    rfib[m_coadd][has_match], 
                    ztrue[m_vi][has_match],
                    np.repeat(100.0, np.sum(has_match)),
                    seed=1, 
                    templateid=match_temp[has_match],
                    emflux=None, 
                    mag_em=None)
            coadd_fluxes.append(coadd_flux[gal_cut][m_coadd][has_match])
            s_fluxes.append(s_flux)

        coadd_fluxes = np.concatenate(coadd_fluxes, axis=0) 
        s_fluxes = np.concatenate(s_fluxes, axis=0) 

        igals = np.random.choice(np.arange(s_fluxes.shape[0]), size=5, replace=False)

        fig = plt.figure(figsize=(15,20))
        for i, igal in enumerate(igals): 
            sub = fig.add_subplot(5,1,i+1)
            sub.plot(coadd_wave, coadd_fluxes[igal,:] * transp * 0.775, c='C0', lw=0.1) 
            sub.plot(coadd_wave, medfilt(coadd_fluxes[igal,:], 101) * transp * 0.775 , c='C0', 
                    label='(coadd flux) x (TRANSP) x (0.775)') 
            sub.plot(coadd_wave, coadd_fluxes[igal,:] * fibloss, c='C1', lw=0.1)
            sub.plot(coadd_wave, medfilt(coadd_fluxes[igal,:], 101) * fibloss, c='C1', 
                    label='(coadd flux) x (TRANSP) x (FIBER FRACFLUX)') 

            sub.plot(s_wave, s_fluxes[igal,:] * transp, c='k', ls='--',
                    label='(sim source flux) x (TRANSP)') 

            sub.set_xlim(3600, 9800)
            if i < 4: sub.set_xticklabels([]) 
            if i == 1: sub.set_ylabel('inciddent flux [$10^{-17} erg/s/cm^2/A$]', fontsize=25) 
            sub.set_ylim(-0.5, 6)

        sub.legend(loc='upper right', handletextpad=0.1, fontsize=20)
        sub.set_xlabel('wavelength', fontsize=25) 
        fig.savefig(os.path.join(dir, 
            'valid.spectral_pipeline_source.exp%i.png' % expid),
            bbox_inches='tight') 
        plt.close() 
    return None 


def validate_cmx_zsuccess_specsim_discrepancy(dchi2=40.): 
    ''' This ended up being a useless test because the smoothed CMX spectra
    that I was using as the source spectra has no features to fit the redshfits!
    
    currently we know that the spectral simulation pipeline does not fuly
    reproduce the noise level of CMX spectra even when we use the smoothed out
    fiber loss corrected CMX spectra as input. This script is to check whether
    this discrepancy significantly impacts the redshift success rates. 

    So we'll be comparing
    - z-success rate of observe CMX exposure with VI truth table 
    - z-success rate of simulated CMX exposure (smoothed fib.loss corrected
      source spectra + CMX sky) 

    VI is currently available for tile 66033 and night 20200315. 
        
    '''
    import glob 
    from scipy.signal import medfilt
    import desisim.simexp
    import specsim.instrument
    from desitarget.cmx import cmx_targetmask

    np.random.seed(0) 

    tileid = 66003 
    date = 20200315
    expids = [55654, 55655, 55656]
    
    dir_gfa = '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions'
    dir_redux = "/global/cfs/cdirs/desi/spectro/redux/daily"
    dir_coadd = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'


    fvi = os.path.join('/global/cfs/cdirs/desi/sv/vi/TruthTables/',
            'truth_table_BGS_v1.2.csv') 
    vi_id, ztrue, qa_flag = np.genfromtxt(fvi, delimiter=',', skip_header=1, unpack=True, 
            usecols=[0, 2, 3]) 
    good_z = (qa_flag >= 2.5) 
    vi_id = vi_id[good_z].astype(int)
    ztrue = ztrue[good_z]

    for expid in expids: 
        print('--- %i ---' % expid) 
        # get fiber acceptance fraction and airmass  
        gfa = fitsio.read(os.path.join(dir_gfa,
            'offline_all_guide_ccds_thru_20200315.fits')) 
        isexp = (gfa['EXPID'] == expid)
        fibloss = gfa['TRANSPARENCY'][isexp] * gfa['FIBER_FRACFLUX'][isexp]
        fibloss = np.median(fibloss[~np.isnan(fibloss)])
        print('  fiber loss = (TRANSP) x (FFRAC) = %f' % fibloss) 
        airmass = np.median(gfa['AIRMASS'][isexp]) 
        print('  airmass = %.2f' % airmass) 

        ispecs = np.sort([int(os.path.basename(fframe).split('-')[1].replace('z', '')) 
                for fframe in glob.glob(os.path.join(dir_redux, 
                    'exposures', str(date), str(expid).zfill(8),
                    'frame-z*.fits'))])

        # exposure time
        _frame = desispec.io.read_frame(os.path.join(dir_redux, 
                    'exposures', str(date), str(expid).zfill(8),
                    'frame-b%i-%s.fits' % (ispecs[0], str(expid).zfill(8))))
        exptime = _frame.meta['EXPTIME']
        print('  exp.time = %.fs' % exptime) 

        for ispec in ispecs: 
            print('  petal %i' % ispec) 
            fexp = os.path.join(dir, 'sim_cmx_spectra.exp%i.petal%i.texp%.fs.fits'
                    % (expid, ispec, exptime)) 

            # get target id 
            f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
            coadd = fitsio.read(f_coadd)
            coadd_wave = fitsio.read(f_coadd, ext=2)
            coadd_flux = fitsio.read(f_coadd, ext=3)
            
            is_BGS  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
            gal_cut = is_BGS & (np.sum(coadd_flux, axis=1) != 0)
            igals = np.arange(len(gal_cut))[gal_cut]
            print('  %i BGS galaxies' % np.sum(gal_cut)) 

            if os.path.isfile(fexp): continue

            # get sky surface brightness for petal
            f_sky = lambda band: os.path.join(dir_redux, 
                    'exposures', str(date), str(expid).zfill(8),
                    'sky-%s%i-%s.fits' % (band, ispec, str(expid).zfill(8)))
            sky_b = desispec.io.read_sky(f_sky('b')) 
            sky_r = desispec.io.read_sky(f_sky('r')) 
            sky_z = desispec.io.read_sky(f_sky('z')) 

            wave, sky_electrons = bs_coadd(
                    [sky_b.wave, sky_r.wave, sky_z.wave], 
                    [sky_b.flux, sky_r.flux, sky_z.flux]) 

            # get which are good fibers from coadd file
            is_good = (coadd['FIBERSTATUS'] == 0)
            is_sky  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SKY')) != 0
            good_sky = is_good & is_sky
            
            # get throughput for the cameras 
            config = desisim.simexp._specsim_config_for_wave(wave, dwave_out=0.8, specsim_config_file='desi')
            instrument = specsim.instrument.initialize(config, True)
            throughput = np.amax([instrument.cameras[0].throughput, instrument.cameras[1].throughput, instrument.cameras[2].throughput], axis=0)

            desi_fiber_area = 1.862089 # fiber area 

            # calculate sky brightness
            sky_bright = np.median(sky_electrons[good_sky,:], axis=0) / throughput / instrument.photons_per_bin / exptime * 1e17

            # source flux is the smoothed CMX spetra
            source_flux = np.zeros((len(igals), len(wave)))
            for i in range(len(igals)): 
                source_flux[i,:] = np.clip(np.interp(wave, coadd_wave,
                    medfilt(coadd_flux[igals[i],:], 101)), 0, None) 

            # simulate the exposures using the spectral simulation pipeline  
            fdesi = FM.fakeDESIspec()
            bgs = fdesi.simExposure(
                    wave, 
                    source_flux * fibloss, # scale by fiber acceptance fraction 
                    exptime=exptime, 
                    airmass=airmass, 
                    Isky=[wave, sky_bright], 
                    dwave_out=0.8, 
                    filename=fexp) 

            frr = run_redrock(fexp, overwrite=False)

        for ispec in ispecs: 
            print('  petal %i' % ispec) 

            # get target id 
            f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
            coadd = fitsio.read(f_coadd)
            coadd_wave = fitsio.read(f_coadd, ext=2)
            coadd_flux = fitsio.read(f_coadd, ext=3)
            coadd_ivar = fitsio.read(f_coadd, ext=4)
            
            is_BGS  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
            gal_cut = is_BGS & (np.sum(coadd_flux, axis=1) != 0)

            fexp = os.path.join(dir, 'sim_cmx_spectra.exp%i.petal%i.texp%.fs.fits'
                    % (expid, ispec, exptime)) 
            sim = desispec.io.read_spectra(fexp) 
            
            # randomly check 3 galaxies 
            igals = np.random.choice(np.arange(np.sum(gal_cut)), size=3, replace=False)

            fig = plt.figure(figsize=(15,15))
    
            for i, igal in enumerate(igals):
                sub = fig.add_subplot(3,1,i+1)

                sub.plot(coadd_wave, coadd_flux[gal_cut,:][igal,:], c='C0', label='coadd') 

                for band in ['b', 'r', 'z']: 
                    sub.plot(sim.wave[band], sim.flux[band][igal,:] / fibloss, c='C1',
                            label='sim / fib.loss') 

                sub.set_xlim(3600, 9800)
                if i < 2: sub.set_xticklabels([]) 
                if i == 1: sub.set_ylabel('flux [$10^{-17} erg/s/cm^2/A$]', fontsize=25) 
                sub.set_ylim(-1., None)
            sub.legend(loc='upper right', handletextpad=0.1, fontsize=20)
            sub.set_xlabel('wavelength', fontsize=25) 
            fig.savefig(os.path.join(dir, 
                'valid.spectral_pipeline_zsuccess_flux.exp%i.petal%i.png' %
                (expid, ispec)), bbox_inches='tight') 
            plt.close() 

            fig = plt.figure(figsize=(15,15))
    
            for i, igal in enumerate(igals):
                sub = fig.add_subplot(3,1,i+1)

                sub.plot(coadd_wave, coadd_ivar[gal_cut,:][igal,:], c='C0', label='coadd') 

                for band in ['b', 'r', 'z']: 
                    sub.plot(sim.wave[band], sim.ivar[band][igal,:] *
                            fibloss**2, c='C1', label='sim x (fib.loss$)^2$') 

                sub.set_xlim(3600, 9800)
                if i < 2: sub.set_xticklabels([]) 
                if i == 1: sub.set_ylabel('ivar', fontsize=25) 
                sub.set_ylim(0., None)
            sub.legend(loc='upper right', handletextpad=0.1, fontsize=20)
            sub.set_xlabel('wavelength', fontsize=25) 
            fig.savefig(os.path.join(dir, 
                'valid.spectral_pipeline_zsuccess_ivar.exp%i.petal%i.png' %
                (expid, ispec)), bbox_inches='tight') 
            plt.close() 

        # read in single exposure coadd and redrock output 
        for i, ispec in enumerate(ispecs): 
            # get target id 
            f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
            coadd = fitsio.read(f_coadd)
            coadd_flux = fitsio.read(f_coadd, ext=3)
            
            is_BGS  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
            gal_cut = is_BGS & (np.sum(coadd_flux, axis=1) != 0)
    
            targetid = coadd['TARGETID'][gal_cut] 
            
            # read coadd redrock fits
            rr_coadd = fitsio.read(f_coadd.replace('coadd-', 'zbest-')) 
            rr_coadd_z      = rr_coadd['Z'][gal_cut]
            rr_coadd_zwarn  = rr_coadd['ZWARN'][gal_cut]
            rr_coadd_dchi2  = rr_coadd['DELTACHI2'][gal_cut]

            fexp = os.path.join(dir, 'sim_cmx_spectra.exp%i.petal%i.texp%.fs.fits'
                    % (expid, ispec, exptime)) 
            frr_sim = run_redrock(fexp, overwrite=False)

            rr_sim = fitsio.read(frr_sim)
            rr_sim_z        = rr_sim['Z']
            rr_sim_zwarn    = rr_sim['ZWARN']
            rr_sim_dchi2    = rr_sim['DELTACHI2'] 

            # match VI to exposure based on target ids 
            _, m_vi, m_sim = np.intersect1d(vi_id, targetid, return_indices=True) 
            print('%i matches to VI' % len(m_vi))
            print('  ', ztrue[m_vi][:5])
            print('  ', rr_coadd_z[m_sim][:5])
            print('  ', rr_sim_z[m_sim][:5])
            
            if i == 0: 
                rmags           = [] 
                ztrues          = [] 
                rr_coadd_zs     = []
                rr_coadd_zwarns = []
                rr_coadd_dchi2s = []
                rr_sim_zs       = []
                rr_sim_zwarns   = []
                rr_sim_dchi2s   = []
            rmags.append(UT.flux2mag(coadd['FLUX_R'][gal_cut][m_sim], method='log')) 
            ztrues.append(ztrue[m_vi])
            rr_coadd_zs.append(rr_coadd_z[m_sim])
            rr_coadd_zwarns.append(rr_coadd_zwarn[m_sim])
            rr_coadd_dchi2s.append(rr_coadd_dchi2[m_sim])
            rr_sim_zs.append(rr_sim_z[m_sim])
            rr_sim_zwarns.append(rr_sim_zwarn[m_sim])
            rr_sim_dchi2s.append(rr_sim_dchi2[m_sim])

        rmags           = np.concatenate(rmags)
        ztrues          = np.concatenate(ztrues)
        rr_coadd_zs     = np.concatenate(rr_coadd_zs)
        rr_coadd_zwarns = np.concatenate(rr_coadd_zwarns)
        rr_coadd_dchi2s = np.concatenate(rr_coadd_dchi2s)
        rr_sim_zs       = np.concatenate(rr_sim_zs)
        rr_sim_zwarns   = np.concatenate(rr_sim_zwarns)
        rr_sim_dchi2s   = np.concatenate(rr_sim_dchi2s)

        zs_coadd = UT.zsuccess(rr_coadd_zs, ztrues, rr_coadd_zwarns,
                deltachi2=rr_coadd_dchi2s, min_deltachi2=dchi2)
        zs_sim = UT.zsuccess(rr_sim_zs, ztrues, rr_sim_zwarns,
                deltachi2=rr_sim_dchi2s, min_deltachi2=dchi2)
        print('coadd z-success %.2f' % (np.sum(zs_coadd)/float(len(zs_coadd))))
        print('sim z-success %.2f' % (np.sum(zs_sim)/float(len(zs_sim))))
    
        # compare the two redshift success rates 
        fig = plt.figure(figsize=(6,6))
        sub = fig.add_subplot(111)
    
        sub.plot([16, 21], [1.0, 1.0], c='k', ls='--') 
        wmean, rate, err_rate = UT.zsuccess_rate(rmags, zs_coadd, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.errorbar(wmean, rate, err_rate, fmt='.C0', label='coadd')
        wmean, rate, err_rate = UT.zsuccess_rate(rmags, zs_sim, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.errorbar(wmean, rate, err_rate, fmt='.C1', label='specsim')
        
        sub.text(21., 1.05, r'$\Delta \chi^2 = %.f$' % dchi2, fontsize=20)
        sub.legend(loc='lower left', ncol=3, handletextpad=0.1, fontsize=15)
        sub.set_xlabel(r'Legacy $r$ fiber magnitude', fontsize=20)
        sub.set_xlim(16, 20.5) 

        sub.set_ylabel(r'redrock $z$ success rate', fontsize=20)
        sub.set_ylim([0.6, 1.1])
        sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 

        fig.savefig(os.path.join(dir, 
            'valid.spectral_pipeline_zsuccess.exp%i.png' % expid),
            bbox_inches='tight') 
        plt.close()
    return None 


def validate_cmx_zsuccess(dchi2=40.): 
    ''' currently we know that the spectral simulation pipeline does not fuly
    reproduce the noise level of CMX spectra even when we use the smoothed out
    fiber loss corrected CMX spectra as input. This script is to check whether
    this discrepancy significantly impacts the redshift success rates. 

    So we'll be comparing
    - z-success rate of observe CMX exposure with VI truth table 
    - z-success rate of spectral simulations run with CMX sky and transparency

    VI is currently available for tile 66033 and night 20200315. 
    '''
    import glob 
    from scipy.signal import medfilt
    import desisim.simexp
    import specsim.instrument
    from desitarget.cmx import cmx_targetmask

    np.random.seed(0) 

    tileid = 66003 
    date = 20200315
    expids = [55654, 55655, 55656]
    
    dir_gfa = '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions'
    dir_redux = "/global/cfs/cdirs/desi/spectro/redux/daily"
    dir_coadd = '/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd/'
    
    # read VI table 
    fvi = os.path.join('/global/cfs/cdirs/desi/sv/vi/TruthTables/',
            'truth_table_BGS_v1.2.csv') 
    vi_id, ztrue, qa_flag = np.genfromtxt(fvi, delimiter=',', skip_header=1, unpack=True, 
            usecols=[0, 2, 3]) 
    good_z = (qa_flag >= 2.5) 
    vi_id = vi_id[good_z].astype(int)
    ztrue = ztrue[good_z]

    # read GAMA-Legacy source fluxes
    wave_s, flux_s, meta_s = source_spectra() 

    for expid in expids: 
        print('--- %i ---' % expid) 
        # get fiber acceptance fraction and airmass  
        gfa = fitsio.read(os.path.join(dir_gfa,
            'offline_all_guide_ccds_thru_20200315.fits')) 
        isexp = (gfa['EXPID'] == expid)
        
        fwhm = gfa['FWHM_ASEC'][isexp]
        print('  (FWHM) = %f' % np.median(fwhm[~np.isnan(fwhm)]))

        transp = gfa['TRANSPARENCY'][isexp]
        transp = np.median(transp[~np.isnan(transp)])
        print('  (TRANSP) = %f' % transp) 

        fibloss = transp * gfa['FIBER_FRACFLUX'][isexp]
        fibloss = np.median(fibloss[~np.isnan(fibloss)])
        print('  fiber loss = (TRANSP) x (FFRAC) = %f' % fibloss) 
        airmass = np.median(gfa['AIRMASS'][isexp]) 
        print('  airmass = %.2f' % airmass) 

        # get petals 
        ispecs = np.sort([int(os.path.basename(fframe).split('-')[1].replace('z', '')) 
                for fframe in glob.glob(os.path.join(dir_redux, 
                    'exposures', str(date), str(expid).zfill(8),
                    'frame-z*.fits'))])

        # exposure time
        _frame = desispec.io.read_frame(os.path.join(dir_redux, 
                    'exposures', str(date), str(expid).zfill(8),
                    'frame-b%i-%s.fits' % (ispecs[0], str(expid).zfill(8))))
        exptime = _frame.meta['EXPTIME']
        print('  exp.time = %.fs' % exptime) 

        # simulated exposure
        fexp = os.path.join(dir, 'spectralsim_source.cmx_sky.exp%i.fits' % expid)

        if not os.path.isfile(fexp): 
            # get sky brightness for exposure 
            sky_brights = [] 
            for ispec in ispecs: 
                print('  petal %i' % ispec) 
                f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
                coadd = fitsio.read(f_coadd)

                # get sky surface brightness for petal
                f_sky = lambda band: os.path.join(dir_redux, 
                        'exposures', str(date), str(expid).zfill(8),
                        'sky-%s%i-%s.fits' % (band, ispec, str(expid).zfill(8)))
                sky_b = desispec.io.read_sky(f_sky('b')) 
                sky_r = desispec.io.read_sky(f_sky('r')) 
                sky_z = desispec.io.read_sky(f_sky('z')) 

                wave, sky_electrons = bs_coadd(
                        [sky_b.wave, sky_r.wave, sky_z.wave], 
                        [sky_b.flux, sky_r.flux, sky_z.flux]) 

                # get which are good fibers from coadd file
                is_good = (coadd['FIBERSTATUS'] == 0)
                is_sky  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SKY')) != 0
                good_sky = is_good & is_sky
                
                # get throughput for the cameras 
                config = desisim.simexp._specsim_config_for_wave(wave, dwave_out=0.8, specsim_config_file='desi')
                instrument = specsim.instrument.initialize(config, True)
                throughput = np.amax([instrument.cameras[0].throughput, instrument.cameras[1].throughput, instrument.cameras[2].throughput], axis=0)

                desi_fiber_area = 1.862089 # fiber area 

                # calculate sky brightness
                sky_bright = np.median(sky_electrons[good_sky,:], axis=0) / throughput / instrument.photons_per_bin / exptime * 1e17
                sky_brights.append(sky_bright) 

            sky_brights = np.array(sky_brights)
            # median sky brightness of the petals
            sky_bright = np.median(sky_brights, axis=0) 

            # simulate the exposures using the spectral simulation pipeline  
            fdesi = FM.fakeDESIspec()
            bgs = fdesi.simExposure(
                    wave_s, 
                    flux_s * transp, # scale by transparency
                    exptime=exptime, 
                    airmass=airmass, 
                    Isky=[wave, sky_bright], 
                    dwave_out=0.8, 
                    filename=fexp) 

        # run redrock 
        frr_sim = run_redrock(fexp, overwrite=False)
        rr_sim  = fitsio.read(frr_sim)
        rr_sim_z      = rr_sim['Z']
        rr_sim_zwarn  = rr_sim['ZWARN']
        rr_sim_dchi2  = rr_sim['DELTACHI2']

        # compile single exposure coadd and redrock output 
        for i, ispec in enumerate(ispecs): 
            # get target id 
            f_coadd = os.path.join(dir_coadd, 'coadd-%i-%i-%i-%s.fits' % (tileid, date, ispec, str(expid).zfill(8)))
            coadd = fitsio.read(f_coadd)
            coadd_flux = fitsio.read(f_coadd, ext=3)
            
            is_BGS  = (coadd['CMX_TARGET'] & cmx_targetmask.cmx_mask.mask('SV0_BGS')) != 0
            gal_cut = is_BGS & (np.sum(coadd_flux, axis=1) != 0)
    
            targetid = coadd['TARGETID'][gal_cut] 
            
            # read coadd redrock fits
            rr_coadd = fitsio.read(f_coadd.replace('coadd-', 'zbest-')) 
            rr_coadd_z      = rr_coadd['Z'][gal_cut]
            rr_coadd_zwarn  = rr_coadd['ZWARN'][gal_cut]
            rr_coadd_dchi2  = rr_coadd['DELTACHI2'][gal_cut]

            # match VI to exposure based on target ids 
            _, m_vi, m_coadd = np.intersect1d(vi_id, targetid, return_indices=True) 
            
            if i == 0: 
                rmags           = [] 
                ztrues          = [] 
                rr_coadd_zs     = []
                rr_coadd_zwarns = []
                rr_coadd_dchi2s = []
            rmags.append(UT.flux2mag(coadd['FLUX_R'][gal_cut][m_coadd], method='log')) 
            ztrues.append(ztrue[m_vi])
            rr_coadd_zs.append(rr_coadd_z[m_coadd])
            rr_coadd_zwarns.append(rr_coadd_zwarn[m_coadd])
            rr_coadd_dchi2s.append(rr_coadd_dchi2[m_coadd])
        print('%i matches to VI' % len(rmags))

        rmags           = np.concatenate(rmags)
        ztrues          = np.concatenate(ztrues)
        rr_coadd_zs     = np.concatenate(rr_coadd_zs)
        rr_coadd_zwarns = np.concatenate(rr_coadd_zwarns)
        rr_coadd_dchi2s = np.concatenate(rr_coadd_dchi2s)

        zs_coadd = UT.zsuccess(rr_coadd_zs, ztrues, rr_coadd_zwarns,
                deltachi2=rr_coadd_dchi2s, min_deltachi2=dchi2)
        zs_sim = UT.zsuccess(rr_sim_z, meta_s['zred'], rr_sim_zwarn,
                deltachi2=rr_sim_dchi2, min_deltachi2=dchi2)
        print('coadd z-success %.2f' % (np.sum(zs_coadd)/float(len(zs_coadd))))
        print('sim z-success %.2f' % (np.sum(zs_sim)/float(len(zs_sim))))
    
        # compare the two redshift success rates 
        fig = plt.figure(figsize=(6,6))
        sub = fig.add_subplot(111)
    
        sub.plot([16, 21], [1.0, 1.0], c='k', ls='--') 
        wmean, rate, err_rate = UT.zsuccess_rate(rmags, zs_coadd, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.errorbar(wmean, rate, err_rate, fmt='.C0', label='coadd')
        wmean, rate, err_rate = UT.zsuccess_rate(meta_s['r_mag'], zs_sim, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.errorbar(wmean, rate, err_rate, fmt='.C1', label='spectral sim')
        
        sub.text(19.5, 1.05, r'$\Delta \chi^2 = %.f$' % dchi2, fontsize=20)
        sub.legend(loc='lower left', ncol=3, handletextpad=0.1, fontsize=15)
        sub.set_xlabel(r'Legacy $r$ fiber magnitude', fontsize=20)
        sub.set_xlim(16, 20.5) 

        sub.set_ylabel(r'redrock $z$ success rate', fontsize=20)
        sub.set_ylim([0.6, 1.1])
        sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 

        fig.savefig(os.path.join(dir, 
            'valid.spectralsim_source.cmx_sky.zsuccess.exp%i.png' % expid),
            bbox_inches='tight') 
        plt.close()
    return None 


def tnom(dchi2=40.):
    ''' Calculate z-success rate for nominal dark time exposure with different
    tnom exposure times. For each tnom, use the z-success rate to determine
    r_lim, the r magnitude that gets 95% completeness. 
    '''
    np.random.seed(0) 
    
    # nominal exposure times
    if dchi2 == 40: 
        texps = [100 + 20 * i for i in range(11)][::2]
    elif dchi2 == 100: 
        texps = [200 + 10 * i for i in range(11)][::2]
   
    # true redshift and r-magnitude 
    _, _, meta = source_spectra() 
    ztrue = meta['zred']  # true redshifts 
    r_mag = meta['r_mag'] 
    r_fib = meta['r_mag_apflux']

    # generate spectra for nominal dark sky exposures and run redrock 
    frr_noms = [] 
    for texp in texps: 
        spec_nom = nomdark_spectra(texp) 
        # run redrock on nominal dark sky exposure spectra 
        frr_nom = run_redrock(
                os.path.join(dir, 'exp_spectra.nominal_dark.%.fs.fits' % texp), 
                overwrite=False)
        frr_noms.append(frr_nom) 

    rmags = np.linspace(17, 20, 31)

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot([16, 21], [1., 1.], c='k', ls=':') 
    
    # for each tnom, calculate rlim from the z-sucess rates 
    for i, texp, frr_nom in zip(range(len(texps)), texps, frr_noms): 
        # read redrock output and calculate z-success 
        rr_nom = fitsio.read(frr_nom) 
        zs_nom = UT.zsuccess(rr_nom['Z'], ztrue, rr_nom['ZWARN'],
                deltachi2=rr_nom['DELTACHI2'], min_deltachi2=dchi2)
    
        # ignore redshift failtures for bright r < 18.2 galaxies, since this is
        # likely an issue with the emission line 
        zs_nom[r_mag < 18.2] = True

        # determine rlim 
        zs_rmag = [] 
        for _r in rmags: 
            brighter = (r_mag < _r) 
            zs_rmag.append(np.sum(zs_nom[brighter]) / np.sum(brighter))
    
        crit = (np.array(zs_rmag) < 0.95) & (rmags > 18) 
        if np.sum(crit) > 0: 
            rlim = np.min(rmags[crit])
        else: 
            rlim = np.max(rmags) 
        print('--- tnom = %.fs ---' % texp) 
        print('  total z-success = %.2f' % (np.sum(zs_nom)/float(len(zs_nom))))
        print('  95percent complete rlim = %.1f' % rlim) 

        wmean, rate, err_rate = UT.zsuccess_rate(r_mag, zs_nom, range=[15,22], 
                nbins=28, bin_min=10) 
        sub.plot(wmean, rate, label=r'%.fs; $r_{\rm lim}= %.1f$' % (texp, rlim))
    
    sub.text(19., 1.05, r'$\Delta \chi^2 = %.f$' % dchi2, fontsize=20)
    sub.legend(loc='lower left', handletextpad=0.1, fontsize=15)
    sub.set_xlabel(r'Legacy $r$ magnitude', fontsize=20)
    sub.set_xlim([16., 20.5]) 
    sub.set_ylabel(r'redrock $z$ success rate', fontsize=20)
    sub.set_ylim([0.6, 1.1])
    sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
    fig.savefig(os.path.join(dir, 'zsuccess.tnom.dchi2_%i.png' % dchi2),
            bbox_inches='tight') 
    plt.close() 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.plot([18, 25], [1., 1.], c='k', ls=':') 
    
    # nominal exposure z-success rate as a function of fiber magnitude 
    for i, texp, frr_nom in zip(range(len(texps)), texps, frr_noms): 
        # read redrock output and calculate z-success 
        rr_nom = fitsio.read(frr_nom) 
        zs_nom = UT.zsuccess(rr_nom['Z'], ztrue, rr_nom['ZWARN'],
                deltachi2=rr_nom['DELTACHI2'], min_deltachi2=dchi2)
    
        # ignore redshift failtures for bright r < 18.2 galaxies, since this is
        # likely an issue with the emission line 
        zs_nom[r_mag < 18.2] = True

        wmean, rate, err_rate = UT.zsuccess_rate(r_fib, zs_nom, range=[18,23], 
                nbins=28, bin_min=10) 
        sub.plot(wmean, rate, err_rate, label=r'%.fs' % texp)

    sub.text(21., 1.05, r'$\Delta \chi^2 = %.f$' % dchi2, fontsize=20)
    sub.legend(loc='lower left', ncol=3, handletextpad=0.1, fontsize=15)
    sub.set_xlabel(r'Legacy $r$ fiber magnitude', fontsize=20)
    sub.set_xlim([18., 22.5]) 
    sub.set_ylabel(r'redrock $z$ success rate', fontsize=20)
    sub.set_ylim([0.6, 1.1])
    sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
    fig.savefig(os.path.join(dir, 'zsuccess.tnom.r_fib.dchi2_%i.png' % dchi2),
            bbox_inches='tight') 
    return None 
    

def texp_factor_wavelength(emlines=True): 
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

    updates
    -------
    * David Schlegel was surprised that 6500A agreed better. He finds that
      5000A agrees better. He suggested I run this test without emission lines 
    * 06/11/2020: Read noise term in the SNR calculation cannot be ignored when
      our nominal exposure time is low. New fsky values calculated for CMX
      exposures including read noise. 
    '''
    np.random.seed(0) 
    
    import desisim.simexp
    from desimodel.io import load_throughput
    wavemin = load_throughput('b').wavemin - 10.0
    wavemax = load_throughput('z').wavemax + 10.0
    wave = np.arange(round(wavemin, 1), wavemax, 0.8) * u.Angstrom
    config = desisim.simexp._specsim_config_for_wave(wave.to('Angstrom').value, dwave_out=0.8, specsim_config_file='desi')
    nominal_surface_brightness_dict = config.load_table(
            config.atmosphere.sky, 'surface_brightness', as_dict=True)
    Isky_nom = [wave, nominal_surface_brightness_dict['dark']] 
    # generate spectra for nominal dark sky exposure as reference
    spec_nom = nomdark_spectra(150, emlines=emlines) 
    # run redrock on nominal dark sky exposure spectra 
    frr_nom = run_redrock(os.path.join(dir, 
        'exp_spectra.nominal_dark%s.150s.fits' % ['.noemission', ''][emlines]), 
        overwrite=False)
    
    # read in CMX sky data 
    skies = cmx_skies()
    # select CMX exposures when the sky was brighter than dark time. In
    # principle we should focus on bright exposures (i.e. 2.5x nominal).
    # we also remove exposures from 20200314 which has strange sky fluxes.
    #bright = (((skies['sky_ratio_5000'] > 1.) | (skies['sky_ratio_7000'] > 1.)) 
    #        & (skies['date'] != 20200314)) 
    #print('%i exposures with sky ratios > 1 and not taken during March 14' % len(expids))
    bright = (((skies['fsky_5000'] > 1.5) | (skies['fsky_7000'] > 1.5)) 
            & (skies['date'] != 20200314)) 
    expids = np.unique(skies['expid'][bright])[:5]
    print('%i exposures with fsky > 1.5 and not taken during March 14' % len(expids))
    #np.random.choice(np.unique(skies['expid'][bright]), size=5, replace=False) 

    # generate exposure spectra for select CMX sky surface brightnesses with
    # exposure times scaled by (1) sky ratio at 5000A (2) sky ratio at 6500A
    for expid in expids:
        print('--- expid = %i ---' % expid) 
        is_exp = (skies['expid'] == expid) 
        # get median sky surface brightnesses for exposure 
        Isky = bs_coadd(
                [skies['wave_b'], skies['wave_r'], skies['wave_z']], 
                [
                    np.median(skies['sky_sb_b'][is_exp], axis=0), 
                    np.median(skies['sky_sb_r'][is_exp], axis=0), 
                    np.median(skies['sky_sb_z'][is_exp], axis=0)]
                )

        fig = plt.figure(figsize=(15,10)) 
        sub = fig.add_subplot(211)
        sub.plot(Isky_nom[0], Isky_nom[1], c='k', lw=0.5) 
        sub.plot(Isky[0], Isky[1], c='C0', lw=0.5) 
        sub.set_xlabel('wavelength', fontsize=20) 
        sub.set_xlim(3.6e3, 9.8e3) 
        sub.set_ylabel('flux', fontsize=20) 
        sub.set_ylim(0., 10.) 

        sub = fig.add_subplot(212)
        for band in ['b', 'r', 'z']: 
            sub.plot(spec_nom.wave[band], spec_nom.flux[band][0,:], c='k', lw=1) 
    
        # get median sky ratios for the exposure 
        for i, _w in enumerate([5000, 7000]): 
            _fexp = np.median(skies['fsky_%i' % _w ][is_exp]) 
            print('  fexp at %iA = %.2f' % (_w, _fexp))
            print('  sky ratio = %.2f' % (np.median(skies['sky_ratio_%i' % _w][is_exp])))

            # generate exposure spectra for expid CMX sky  
            _fspec = os.path.join(dir, 'exp_spectra.exp%i%s.fexp_%i.fits' %
                (expid, ['.noemission', ''][emlines], _w))

            _spec = exp_spectra(
                    Isky,           # sky surface brightness 
                    150. * _fexp,   # exposure time 
                    1.1,            # same airmass 
                    _fspec,
                    emlines=emlines
                    )
            # run redrock on the exposure spectra 
            frr = run_redrock(_fspec, qos='debug')

            # plot comparing the exp spectra to the nominal dark spectra 
            for band in ['b', 'r', 'z']: 
                lbl = None 
                if band == 'b': 
                    lbl = ('at %iA' % _w)
                sub.plot(_spec.wave[band], _spec.flux[band][0,:], c='C%i' % i,
                        lw=1, label=lbl) 
        sub.set_xlabel('wavelength', fontsize=20) 
        sub.set_xlim(3.6e3, 9.8e3) 
        sub.set_ylabel('flux', fontsize=20) 
        sub.set_ylim(0., 10.) 
        sub.legend(loc='upper right', fontsize=20, ncol=3) 
        fig.savefig(_fspec.replace('.fexp_%i.fits' % _w, '.png'), bbox_inches='tight') 
        plt.close() 

    _, _, meta = source_spectra(emlines=emlines) 
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
    sub.plot([16, 21], [1., 1.], c='k', ls=':') 
    wmean, rate, err_rate = UT.zsuccess_rate(r_mag, zs_nom, range=[15,22], 
            nbins=28, bin_min=10) 
    _plt_nom = sub.errorbar(wmean, rate, err_rate, fmt='.k', elinewidth=2, markersize=10)

    zs_5000, zs_7000 = [], []
    for expid in expids:
        print('--- expid = %i ---' % expid) 
        zss = [] 
        for i, _w in enumerate([5000, 7000]): 
            rr = fitsio.read(os.path.join(dir,
                'zbest.exp_spectra.exp%i%s.fexp_%i.fits' % 
                (expid, ['.noemission', ''][emlines], _w)))

            _zs = UT.zsuccess(rr['Z'], ztrue, rr['ZWARN'],
                    deltachi2=rr['DELTACHI2'], min_deltachi2=dchi2)
            zss.append(_zs)
            
            print('  fexp at %i z-success = %.2f' % (_w, np.sum(_zs)/float(len(_zs))))
            wmean, rate, err_rate = UT.zsuccess_rate(r_mag, _zs, range=[15,22], 
                    nbins=28, bin_min=10) 
            _plt, = sub.plot(wmean, rate, c='C%i' % i)

            if expid == expids[0]: 
                if i == 0: _plts = [_plt_nom]
                _plts.append(_plt) 

        zs_5000.append(zss[0])
        zs_7000.append(zss[1])

    zs_5000 = np.concatenate(zs_5000) 
    zs_7000 = np.concatenate(zs_7000) 
    print('-----------------------')
    print('nominal z-success = %.2f' % (np.sum(zs_nom)/float(len(zs_nom))))
    print('fexp at 5000A z-success = %.2f ' % (np.sum(zs_5000)/float(len(zs_5000))))
    print('fexp at 7000A z-success = %.2f ' % (np.sum(zs_7000)/float(len(zs_7000))))

    sub.text(19., 1.05, r'$\Delta \chi^2 = %.f$' % dchi2, fontsize=20)
    sub.legend(_plts, 
            ['nominal dark 150s', 
                r'CMX exp. $f_{\rm sky}[5000A]$',
                r'CMX exp. $f_{\rm sky}[7000A]$'], 
            loc='lower left', handletextpad=0.1, fontsize=15)
    sub.set_xlabel(r'Legacy $r$ magnitude', fontsize=20)
    sub.set_xlim([16., 20.5]) 
    sub.set_ylabel(r'redrock $z$ success rate', fontsize=20)
    sub.set_ylim([0.6, 1.1])
    sub.set_yticks([0.6, 0.7, 0.8, 0.9, 1.]) 
    fig.savefig(os.path.join(dir, 
        'zsuccess.exp_spectra%s.fsky.png' % ['.noemission', ''][emlines]),
        bbox_inches='tight') 
    return None 


def _SNR_test(): 
    ''' Q: Why is scaling the exposure time by the sky brightness ratio scaling
    not producing spectra with roughly the same SNR? 

    The SNR of the spectra is approximately 
        SNR = S x sqrt(texp/sky)
    This means that if the sky is twice as bright but you increase texp by 2,
    you would get the same SNR. This, however, does not seem to be the case for
    the SNR for the `exp_spectra` output. 

    In this script I will generate spectra with uniform sky brightness  

    '''
    np.random.seed(0) 

    import desisim.simexp
    from desimodel.io import load_throughput
    wavemin = load_throughput('b').wavemin - 10.0
    wavemax = load_throughput('z').wavemax + 10.0
    wave = np.arange(round(wavemin, 1), wavemax, 0.8) * u.Angstrom

    # get throughput for the cameras 
    import specsim.instrument
    from specsim.simulator import Simulator
    config = desisim.simexp._specsim_config_for_wave(wave.value, dwave_out=0.8, specsim_config_file='desi')
    instrument = specsim.instrument.initialize(config, True)
    throughput = np.amax([instrument.cameras[0].throughput, instrument.cameras[1].throughput, instrument.cameras[2].throughput], axis=0)

    fig = plt.figure(figsize=(20,15)) 
    sub0 = fig.add_subplot(321)
    sub1 = fig.add_subplot(323)
    sub2 = fig.add_subplot(325)
    sub3 = fig.add_subplot(322)
    sub4 = fig.add_subplot(324)
    sub5 = fig.add_subplot(326)
    for ii, i in enumerate([0, 5, 10]): 
        # read in source spectra 
        print('sky = %i' % (i+1)) 
        wave_s, flux_s, _ = source_spectra(emlines=False) 
        #'''
        _fspec = os.path.join(dir, 'exp_spectra.snr_test.sky%i.fits' % (i+1))
        Isky = [wave, np.ones(len(wave)) * (i + 1.)]
        _spec = exp_spectra(
                Isky,           # sky surface brightness 
                150. * (i + 1.),   # exposure time 
                1.1,            # same airmass 
                _fspec,
                emlines=False
                )
        # plot comparing the exp spectra to the nominal dark spectra 
        for band in ['b', 'r', 'z']: 
            lbl = None 
            if band == 'b': lbl = ('sky = %i, texp = %.f' % ((i+1), 150.*(i+1.)))
            sub0.plot(_spec.wave[band], _spec.flux[band][0,:], c='C%i' % ii, lw=1, label=lbl) 
            sub1.plot(_spec.wave[band], _spec.flux[band][1,:], c='C%i' % ii, lw=1, label=lbl) 
            sub2.plot(_spec.wave[band], _spec.flux[band][2,:], c='C%i' % ii, lw=1, label=lbl) 
            sub3.plot(_spec.wave[band], _spec.ivar[band][0,:], c='C%i' % ii, lw=1, label=lbl) 
            sub4.plot(_spec.wave[band], _spec.ivar[band][1,:], c='C%i' % ii, lw=1, label=lbl) 
            sub5.plot(_spec.wave[band], _spec.ivar[band][2,:], c='C%i' % ii, lw=1, label=lbl) 
        sub0.plot(wave_s, flux_s[0,:], c='k', lw=1, ls='--') 
        sub1.plot(wave_s, flux_s[1,:], c='k', lw=1, ls='--') 
        sub2.plot(wave_s, flux_s[2,:], c='k', lw=1, ls='--') 
        '''
            # barebone specsim pipeline for comparison 
            desi = Simulator(config, num_fibers=flux_s.shape[0])
            desi.observation.exposure_time = 150. * (i + 1.) * u.s
            desi.atmosphere._surface_brightness_dict[desi.atmosphere.condition] = \
                    np.ones(len(desi.atmosphere._wavelength)) * (i + 1.) * \
                    desi.atmosphere.surface_brightness.unit
            desi.atmosphere._extinct_emission = False
            desi.atmosphere._moon = None 
            desi.atmosphere.airmass = 1.1
            
            source_flux = np.array([np.clip(np.interp(wave, wave_s, _flux_s), 0, None) for _flux_s in flux_s])
            desi.simulate(source_fluxes=source_flux * 1e-17 * desi.simulated['source_flux'].unit) 

            random_state = np.random.RandomState(0)
            desi.generate_random_noise(random_state, use_poisson=True)

            scale=1e17

            waves, fluxes, ivars, ivars_electron = [], [], [], [] 
            lbl = ('sky=%i' % (i+1))
            for table in desi.camera_output:
                print('  source', table['num_source_electrons'][0][:5]) 
                print('  sky', table['num_sky_electrons'][0][:5]) 
                print('  dark', table['num_dark_electrons'][0][:5]) 
                print('  RN', table['read_noise_electrons'][0][:5]**2) 
                _wave = table['wavelength'].astype(float)
                _flux = (table['observed_flux']+table['random_noise_electrons']*table['flux_calibration']).T.astype(float)
                _flux = _flux * scale

                _ivar = table['flux_inverse_variance'].T.astype(float)
                _ivar = _ivar / scale**2

                sub0.plot(_wave, _flux[0], c='C%i' % ii, lw=1, label=lbl) 
                sub1.plot(_wave, _flux[1], c='C%i' % ii, lw=1, label=lbl) 
                sub2.plot(_wave, _flux[2], c='C%i' % ii, lw=1, label=lbl) 
                sub3.plot(_wave, _ivar[0], c='C%i' % ii, lw=1, label=lbl) 
                sub4.plot(_wave, _ivar[1], c='C%i' % ii, lw=1, label=lbl) 
                sub5.plot(_wave, _ivar[2], c='C%i' % ii, lw=1, label=lbl) 
                lbl = None 
        '''
            
    sub2.set_xlabel('wavelength', fontsize=20) 
    sub0.set_xlim(3.6e3, 9.8e3) 
    sub1.set_xlim(3.6e3, 9.8e3) 
    sub2.set_xlim(3.6e3, 9.8e3) 
    sub3.set_xlim(3.6e3, 9.8e3) 
    sub4.set_xlim(3.6e3, 9.8e3) 
    sub5.set_xlim(3.6e3, 9.8e3) 
    sub1.set_ylabel('flux', fontsize=20) 
    sub4.set_ylabel('ivar', fontsize=20) 
    sub0.set_ylim(0., 10.) 
    sub1.set_ylim(0., 10.) 
    sub2.set_ylim(0., 10.) 
    sub0.legend(loc='upper right', fontsize=15) 
    fig.savefig(os.path.join(dir, 'snr_test.png'), bbox_inches='tight') 
    plt.close() 
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


def source_spectra(emlines=True): 
    ''' read GAMA-matched fiber-magnitude scaled BGS source spectra 
    These source spectra are created for GAMA objects. their spectra is 
    constructed from continuum that's template matched to the broadband
    colors and emission lines from GAMA data (properly flux calibrated). 
    Then the spectra is scaled down to the r-band fiber magnitude. They 
    therefore do not require fiber acceptance fractions. 
    '''
    fsource = os.path.join(dir, 
            'GALeg.g15.sourceSpec%s.1000.seed0.hdf5' % ['.noemission', ''][emlines])

    if not os.path.isfile(fsource): 
        seed = 0 
        np.random.seed(seed) 
        # read in GAMA-Legacy catalog with galaxies in both GAMA and Legacy surveys
        cata = Cat.GamaLegacy()
        gleg = cata.Read('g15', dr_gama=3, dr_legacy=7, silent=True)  
        
        # extract meta-data of galaxies 
        redshift        = gleg['gama-spec']['z']
        absmag_ugriz    = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3, galext=False) # ABSMAG k-correct to z=0.1
        r_mag_apflux    = UT.flux2mag(gleg['legacy-photo']['apflux_r'][:,1]) # aperture flux
        r_mag_gama      = gleg['gama-photo']['r_petro'] # r-band magnitude from GAMA (SDSS) photometry
        ha_gama         = gleg['gama-spec']['ha_flux'] # halpha line flux

        ngal = len(redshift) # number of galaxies
        vdisp = np.repeat(100.0, ngal) # velocity dispersions [km/s]

        # match GAMA galaxies to templates 
        bgs3 = FM.BGStree()
        match = bgs3._GamaLegacy(gleg)
        hasmatch = (match != -999)
        criterion = hasmatch 
        
        # randomly pick a few more than 5000 galaxies from the catalog that have 
        # matching templates because some of the galaxies will have issues where the 
        # emission line is brighter than the photometric magnitude.  
        subsamp = np.random.choice(np.arange(ngal)[criterion], int(1.1 * 1000), replace=False) 

        # generate noiseless spectra for these galaxies 
        s_bgs = FM.BGSsourceSpectra(wavemin=1500.0, wavemax=15000) 
        # emission line fluxes from GAMA data  
        if emlines: 
            emline_flux = s_bgs.EmissionLineFlux(gleg, index=subsamp, dr_gama=3, silent=True) # emission lines from GAMA 
            mag_em = r_mag_gama[subsamp]
        else: 
            emline_flux = None 
            mag_em = None 

        flux, wave, magnorm_flag = s_bgs.Spectra(
                r_mag_apflux[subsamp], 
                redshift[subsamp],
                vdisp[subsamp], 
                seed=1, 
                templateid=match[subsamp], 
                emflux=emline_flux, 
                mag_em=mag_em, 
                silent=True)

        # only keep 1000 galaxies
        isubsamp = np.random.choice(np.arange(len(subsamp))[magnorm_flag], 1000, replace=False) 
        subsamp = subsamp[isubsamp]
        
        # save to file  
        fsub = h5py.File(fsource, 'w') 
        fsub.create_dataset('zred', data=redshift[subsamp])
        fsub.create_dataset('absmag_ugriz', data=absmag_ugriz[:,subsamp]) 
        fsub.create_dataset('r_mag_apflux', data=r_mag_apflux[subsamp]) 
        fsub.create_dataset('r_mag_gama', data=r_mag_gama[subsamp]) 
        for grp in gleg.keys(): 
            group = fsub.create_group(grp) 
            for key in gleg[grp].keys(): 
                group.create_dataset(key, data=gleg[grp][key][subsamp])
        fsub.create_dataset('flux', data=flux[isubsamp, :])
        fsub.create_dataset('wave', data=wave)
        fsub.close()

    # read in source spectra
    source = h5py.File(fsource, 'r')

    wave_s = source['wave'][...]
    flux_s = source['flux'][...]
    
    meta = {} 
    for k in ['r_mag_apflux', 'r_mag_gama', 'zred', 'absmag_ugriz']: 
        meta[k] = source[k][...]
    meta['r_mag'] = UT.flux2mag(source['legacy-photo']['flux_r'][...], method='log') 
    source.close()

    return wave_s, flux_s, meta


def nomdark_spectra(texp, emlines=True): 
    ''' spectra observed during nominal dark sky for 150s. This will
    serve as the reference spectra for a number of tests. 
    '''
    if emlines: 
        fexp = os.path.join(dir, 'exp_spectra.nominal_dark.%.fs.fits' % texp) 
    else: 
        fexp = os.path.join(dir, 'exp_spectra.nominal_dark.noemission.%.fs.fits' % texp) 

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
        wave_s, flux_s, _ = source_spectra(emlines=emlines) 

        # simulate the exposures and save to file 
        fdesi = FM.fakeDESIspec()
        bgs = fdesi.simExposure(
                wave_s, 
                flux_s, 
                exptime=texp, 
                airmass=1.1, 
                Isky=Isky, 
                filename=fexp) 
    return bgs 


def exp_spectra(Isky, exptime, airmass, fexp, emlines=True, overwrite=False): 
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
        wave_s, flux_s, _ = source_spectra(emlines=emlines) 

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


def run_redrock(fspec, qos='regular', overwrite=False): 
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
            "#SBATCH -q %s" % qos, 
            '#SBATCH -J rr_%s' % os.path.basename(fspec).replace('.fits', ''),
            '#SBATCH -o _rr_%s.o' % os.path.basename(fspec).replace('.fits', ''),
            "#SBATCH -t 00:10:00", 
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
    #_SNR_test()
    texp_factor_wavelength()
    #texp_factor_wavelength(emlines=False) # without emission lines 
    #tnom(dchi2=40)
    #tnom(dchi2=100)
    #validate_spectral_pipeline()
    #validate_spectral_pipeline_source()
    #validate_spectral_pipeline_GAMA_source()
    #validate_cmx_zsuccess_specsim_discrepancy()
    #validate_cmx_zsuccess(dchi2=40.)

