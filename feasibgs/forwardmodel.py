'''

forward model a spectra from meta data. 
most of the functions here serve as wrappers
for `desisim` code 

References: 
    - https://github.com/desihub/desisim/blob/master/doc/nb/bgs-reference-spectra.ipynb
    - https://github.com/desihub/desisim/blob/master/doc/nb/bgs-redshift-efficiency.ipynb 
    - https://github.com/desihub/desisim/blob/master/py/desisim/scripts/quickspectra.py
'''
import os
import time 
import numpy as np 
from scipy.spatial import cKDTree as KDTree
# -- astropy 
from astropy.table import vstack
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
# -- desi -- 
from speclite import filters
import specsim.simulator
from specsim.simulator import Simulator 
import specsim.config
from desimodel.io import load_throughput
from desisim.io import empty_metatable
from desisim.io import read_basis_templates
from desisim.obs import get_night
from desisim.templates import BGS
import desisim.simexp
import desisim.specsim
import desimodel.io
import desispec
import desitarget
from desispec.spectra import Spectra
from desispec.resolution import Resolution
from desitarget.cuts import isBGS_bright, isBGS_faint

# -- local -- 
from feasibgs import catalogs as Cat 


class BGStree(object):
    '''class to deal with KDTree from BGS basis template metadata.
    Read in meta data of BGS basis templates, construct a KDTree
    and then use the KDTree to identify closest templates. 
    '''
    def __init__(self):
        # read in the meta data of the BGS basis templates
        self.meta = read_basis_templates(objtype='BGS', onlymeta=True)
        # construct KDTree from it 
        self.tree = KDTree(self.extractMeta())

    def extractMeta(self):
        ''' Extract quantities used to construct KDTree from the basis
        template meta data: redshift (z), M_0.1r, and 0.1(g-r).
        '''
        # read in necessary meta data
        zobj = self.meta['Z'].data  # redshift 
        mabs = self.meta['SDSS_UGRIZ_ABSMAG_Z01'].data # absolute magnitudes in SDSS ugriz bands kcorrected to z=0.1
        rmabs = mabs[:,2] # r-band absolute magnitude
        gr = mabs[:,1] - mabs[:,2] # g-r color 

        return np.vstack((zobj, rmabs, gr)).T
    
    def Query(self, matrix):
        '''Return the nearest template number based on the KD Tree.

        Parameters
        ----------
          matrix (numpy.ndarray): 
            M x N array (M=number of properties, N=number of objects) 
            in the same format as the corresponding function for each 
            object type (e.g., self.bgs).

        Returns
        -------
          - indx: index of nearest template (main item of interest) 
          - dist: distance to nearest template
        '''
        dist, indx = self.tree.query(matrix)
        return indx, dist

    def _GamaLegacy(self, gleg, index=False): 
        ''' Given `catalogs.GamaLegacy` class object, return matches to  
        template. This is purely for convenience. 
        '''
        # extract necessary meta data 
        redshift = gleg['gama-spec']['z_helio']  # redshift
        # calculate ABSMAG k-correct to z=0.1 
        cata = Cat.GamaLegacy()
        absmag_ugriz = cata.AbsMag(gleg, kcorr=0.1, H0=70, Om0=0.3) 

        if not np.any(index): # match all galaxies in the GamaLegacy object
            ind = np.ones(absmag_ugriz.shape[1], dtype=bool) 
        else: # match only specified galaxies
            ind = np.zeros(absmag_ugriz.shape[1], dtype=bool) 
            ind[index] = True 
        
        # stack meta data -- [z, M_r0.1, 0.1(g-r)]
        gleg_meta = np.vstack([
            redshift[ind], 
            absmag_ugriz[2,ind], 
            absmag_ugriz[1,ind] - absmag_ugriz[2,ind]]).T

        # match to templates 
        match, _ = self.Query(gleg_meta)
        # in some cases there won't be a match from  KDTree.query
        # we flag these with -999 
        nomatch = (match >= len(self.meta['TEMPLATEID']))
        match[nomatch] = -999
        return match 


class BGStemplates(object):
    '''Generate spectra for BGS templates.  
    '''
    def __init__(self, wavemin=None, wavemax=None, dw=0.2):
        ''' initiate BGS template spectra. Mainly for initializing `desisim.templates.BGS`
        '''
        # default (buffered) wavelength vector
        if wavemin is None: self.wavemin = load_throughput('b').wavemin - 10.0
        if wavemax is None: self.wavemax = load_throughput('z').wavemax + 10.0
        self.dw = dw
        self.wave = np.arange(round(wavemin, 1), wavemax, dw)

        # initialize the templates once:
        self.bgs_templates = BGS(wave=self.wave, normfilter='decam2014-r') 
        #normfilter='sdss2010-r') # Need to generalize this!
        self.bgs_templates.normline = None # no emission lines!

    def Spectra(self, r_mag, zred, vdisp, seed=None, templateid=None, silent=True):
        ''' Given data in the output format of `feasibgs.catalog.Read`
        generate spectra given the `templateid`.
        '''
        np.random.seed(seed) # set random seed

        # meta data of 'mag', 'redshift', 'vdisp'
        input_meta = empty_metatable(nmodel=len(r_mag), objtype='BGS')
        input_meta['SEED'] = np.random.randint(2**32, size=len(r_mag)) 
        input_meta['MAG'] = r_mag # r band apparent magnitude
        input_meta['REDSHIFT'] = zred # redshift
        input_meta['VDISP'] = vdisp 
        input_meta['TEMPLATEID'] = templateid

        flux, _, meta = self.bgs_templates.make_templates(input_meta=input_meta,
                                                          nocolorcuts=True, 
                                                          novdisp=False,
                                                          verbose=(not silent))
        return flux, self.wave, meta

    def addEmissionLines(self, wave, flux, gama_data, gama_indices, silent=True): 
        ''' add emission lines to spectra
        '''
        assert flux.shape[0] == len(gama_indices)
        _flux = flux.copy() 
        # emission lines
        emline_keys = ['oii', 'hb',  'oiiib', 'oiiir', 'niib', 'ha', 'niir', 'siib', 'siir']
        emline_lambda = [3727., 4861., 4959., 5007., 6548., 6563., 6584., 6716., 6731.]
    
        # gama spectra data
        gama_sdata = gama_data['gama-spec']

        # redshifts 
        z_gama = gama_data['gama-spec']['z_helio'][gama_indices]

        # get median emission line widths for case where the emission line is measured 
        # but the width is not... 
        emline_sigmed = [] 
        for k in emline_keys: 
            hassig = (gama_sdata[k+'sig'] > 0)
            emline_sigmed.append(np.median(gama_sdata[k+'sig'][hassig]))

        # loop through emission lines and add them on! 
        for i_k, k in enumerate(emline_keys): 
            hasem = np.where(gama_sdata[k][gama_indices] != -99.)[0] # galaxies with measured emission line
            if len(hasem) == 0: 
                if not silent: print('no galaxies with emission line %s' % k) 
                continue
            
            em_lineflux = gama_sdata[k][gama_indices][hasem] # line flux [10^(-17) erg/s/cm^2]
            # width of Gaussian emission line 
            em_sig = gama_sdata[k+'sig'][gama_indices][hasem] # Angstrom
            # fit the width of the emission line is not fit 
            # accurately, then use the median value of the data  
            nosig = (em_sig <= 0.) 
            em_sig[nosig] = emline_sigmed[i_k]

            # normalization of the Gaussian
            A = em_lineflux/np.sqrt(2. * np.pi * em_sig**2)

            lambda_eml_red = emline_lambda[i_k] * (1 + z_gama[hasem]) 

            emline_flux = A[:,None] * np.exp(-0.5*(np.tile(wave, len(hasem)).reshape(len(hasem), len(wave)) - lambda_eml_red[:,None])**2/em_sig[:,None]**2)

            _flux[hasem] += emline_flux 

        return wave, _flux 


class fakeDESIspec(object): 
    ''' simulate exposure for DESI spectrograph. 
    This is sort of a giant wrapper for specsim. 
    '''
    def __init__(self): 
        pass

    def skySurfBright(self, wave, cond='bright'): 
        ''' given wavelengths return realistic surface brightnesses 
        of the sky. 

        notes
        -----
        * at the moment, surface brightness of the sky is determined 
            by some crude fit of the surface brightness residual between 
            a bright sky measured from BOSS and the DESI dark sky default
            surface brightness. 
        '''
        # Generate specsim config object for a given wavelength grid
        config = desisim.simexp._specsim_config_for_wave(wave.to('Angstrom').value,
                specsim_config_file='desi')
        # load surface brightness dictionary 
        surface_brightness_dict = config.load_table(config.atmosphere.sky, 'surface_brightness', as_dict=True)
    
        # dark sky surface brightness 
        dark_sbright = surface_brightness_dict['dark'] 
        if cond == 'dark': 
            # desi dark sky is pretty accurate? 
            return dark_sbright 
        elif cond == 'bright':  
            # hacked together implementation of birght sky 
            # (residual surface-brightness fit) + (dark sky surface brightness)
            # see notebook.notes_sky_brightness.ipynb for details on how the 
            # values are calculated
            SBright_resid = lambda x: np.exp(-0.000488 * (x - 5071.) + 1.388)
            sbright_unit = 1e-17 * u.erg / (u.arcsec**2 * u.Angstrom * u.s * u.cm ** 2 )
            return dark_sbright + SBright_resid(config.wavelength.value) * sbright_unit

    def simExposure(self, wave, flux, airmass=1.0, exptime=1000, 
            moonalt=-60, moonsep=180, moonfrac=0.0, seeing=1.1, 
            seed=1, skyerr=0.0, skycondition='bright', nonoise=False, filename=None): 
        ''' simulate exposure for input source flux(wavelength). keyword arguments (airmass, seeing) 
        specify a number of observational conditions. These are used to calculate the extinction
        factor. The sky surface brightness is dictated by `skyconditions` kwarg. 
        '''
        nspec, _ = flux.shape # number of spectra 

        # observation conditions
        obvs_dict = dict(
                AIRMASS=airmass, 
                EXPTIME=exptime, # s 
                MOONALT=moonalt, # deg
                MOONFRAC=moonfrac,
                MOONSEP=moonsep, # deg
                SEEING=seeing)   # arc sec

        tileid  = 0
        dateobs = time.gmtime()
        night   = get_night(utc=dateobs)
        
        frame_fibermap = desispec.io.fibermap.empty_fibermap(nspec) # empty fibermap ndarray
        frame_fibermap.meta["FLAVOR"] = "custom"
        frame_fibermap.meta["NIGHT"] = night
        frame_fibermap.meta["EXPID"] = 0 
        # add DESI_TARGET
        tm = desitarget.targetmask.desi_mask
        frame_fibermap['DESI_TARGET'] = tm.BGS_ANY
        frame_fibermap['TARGETID'] =  np.arange(nspec).astype(int)
    
        # spectra fibermap has two extra fields : night and expid
        # This would be cleaner if desispec would provide the spectra equivalent
        # of desispec.io.empty_fibermap()
        spectra_fibermap = desispec.io.empty_fibermap(nspec)
        spectra_fibermap = desispec.io.util.add_columns(spectra_fibermap,
                           ['NIGHT', 'EXPID', 'TILEID'],
                           [np.int32(night), np.int32(0), np.int32(tileid)],
                           )
        for s in range(nspec):
            for tp in frame_fibermap.dtype.fields:
                spectra_fibermap[s][tp] = frame_fibermap[s][tp]

        # ccd wavelength limit 
        params = desimodel.io.load_desiparams()
        wavemin = params['ccd']['b']['wavemin']
        wavemax = params['ccd']['z']['wavemax']

        if wave[0] > wavemin or wave[-1] < wavemax:
            raise ValueError

        wlim = (wavemin <= wave) & (wave <= wavemax) # wavelength limit 
        wave = wave[wlim]*u.Angstrom

        flux_unit = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm ** 2 )
        flux = flux[:,wlim]*flux_unit

        sim = self._simulate_spectra(wave, flux, fibermap=frame_fibermap, 
                skycondition=skycondition, obsconditions=obvs_dict, 
                redshift=None, seed=seed, psfconvolve=True)
        
        # put in random noise 
        random_state = np.random.RandomState(seed)
        if not nonoise: 
            sim.generate_random_noise(random_state)

        scale=1e17
        specdata = None

        resolution={}
        for camera in sim.instrument.cameras:
            R = Resolution(camera.get_output_resolution_matrix())
            resolution[camera.name] = np.tile(R.to_fits_array(), [nspec, 1, 1])
    
        skyscale = skyerr * random_state.normal(size=sim.num_fibers)

        for table in sim.camera_output :
            wave = table['wavelength'].astype(float)
            flux = (table['observed_flux']+table['random_noise_electrons']*table['flux_calibration']).T.astype(float)
            if np.any(skyscale):
                flux += ((table['num_sky_electrons']*skyscale)*table['flux_calibration']).T.astype(float)
            
            ivar = table['flux_inverse_variance'].T.astype(float)
            
            band  = table.meta['name'].strip()[0]
            
            flux = flux * scale
            ivar = ivar / scale**2
            mask  = np.zeros(flux.shape).astype(int)
            
            spec = Spectra([band], {band : wave}, {band : flux}, {band : ivar}, 
                           resolution_data={band : resolution[band]}, 
                           mask={band : mask}, 
                           fibermap=spectra_fibermap, 
                           meta=None,
                           single=True)
            if specdata is None :
                specdata = spec
            else:
                specdata.update(spec)
        if filename is None: 
            return specdata  
        else: 
            desispec.io.write_spectra(filename, specdata)
            return specdata  

    def _simulate_spectra(self, wave, flux, fibermap=None, skycondition='bright', obsconditions=None, 
            redshift=None, dwave_out=None, seed=None, psfconvolve=True, specsim_config_file = "desi"):
        '''
        A more streamlined BGS version of the method `desisim.simexp.simulate_spectra`, which 
        simulates an exposure 

        Args:
            wave (array): 1D wavelengths in Angstroms
            flux (array): 2D[nspec,nwave] flux in 1e-17 erg/s/cm2/Angstrom
                or astropy Quantity with flux units

        Optional:
            fibermap: table from fiberassign or fibermap; uses X/YFOCAL_DESIGN, TARGETID, DESI_TARGET

            skycondition: (str) 
                specifies the sky condition. At the moment only supports 'bright' or 'dark'. 
                Default is 'bright'

            obsconditions: (dict-like) observation metadata including
                SEEING (arcsec), EXPTIME (sec), AIRMASS,
                MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)
            redshift : list/array with each index being the redshifts for that target
            seed: (int) random seed
            psfconvolve: (bool) passed to simspec.simulator.Simulator camera_output.
                if True, convolve with PSF and include per-camera outputs
            specsim_config_file: (str) path to DESI instrument config file.
                default is desi config in specsim package.
        TODO: galsim support

        Returns a specsim.simulator.Simulator object
        '''
        # Input cosmology to calculate the angular diameter distance of the galaxy's redshift
        LCDM = FlatLambdaCDM(H0=70, Om0=0.3)
        ang_diam_dist = LCDM.angular_diameter_distance
        
        random_state = np.random.RandomState(seed)

        nspec, nwave = flux.shape

        #- Convert to unit-ful quantities for specsim
        #if not isinstance(flux, u.Quantity):
        #    fluxunits = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm**2)
        #    flux = flux * fluxunits

        #if not isinstance(wave, u.Quantity):
        #    wave = wave * u.Angstrom

        # Generate specsim config object for a given wavelength grid
        config = desisim.simexp._specsim_config_for_wave(wave.to('Angstrom').value, 
                dwave_out=dwave_out, specsim_config_file=specsim_config_file)

        # get sky surface brightness
        sky_surface_brightness = self.skySurfBright(wave, cond=skycondition)
        
        #- Create simulator
        desi = SimulatorHacked(config, num_fibers=nspec, camera_output=psfconvolve)

        if obsconditions is None: raise ValueError

        desi.atmosphere.seeing_fwhm_ref = obsconditions['SEEING'] * u.arcsec
        desi.observation.exposure_time = obsconditions['EXPTIME'] * u.s
        desi.atmosphere.airmass = obsconditions['AIRMASS']
        desi.atmosphere.moon.moon_phase = np.arccos(2*obsconditions['MOONFRAC']-1)/np.pi
        desi.atmosphere.moon.moon_zenith = (90 - obsconditions['MOONALT']) * u.deg
        desi.atmosphere.moon.separation_angle = obsconditions['MOONSEP'] * u.deg

        #- Set fiber locations from meta Table or default fiberpos
        fiberpos = desimodel.io.load_fiberpos()
        if len(fiberpos) != len(fibermap):
            ii = np.in1d(fiberpos['FIBER'], fibermap['FIBER'])
            fiberpos = fiberpos[ii]

        #- Extract fiber locations from meta Table -> xy[nspec,2]
        assert np.all(fibermap['FIBER'] == fiberpos['FIBER'][0:nspec])
        if 'XFOCAL_DESIGN' in fibermap.dtype.names:
            xy = np.vstack([fibermap['XFOCAL_DESIGN'], fibermap['YFOCAL_DESIGN']]).T * u.mm
        elif 'X' in fibermap.dtype.names:
            xy = np.vstack([fibermap['X'], fibermap['Y']]).T * u.mm
        else:
            xy = np.vstack([fiberpos['X'], fiberpos['Y']]).T * u.mm

        if 'TARGETID' in fibermap.dtype.names:
            unassigned = (fibermap['TARGETID'] == -1)
            if np.any(unassigned):
                #- see https://github.com/astropy/astropy/issues/5961
                #- for the units -> array -> units trick
                xy[unassigned,0] = np.asarray(fiberpos['X'][unassigned], dtype=xy.dtype) * u.mm
                xy[unassigned,1] = np.asarray(fiberpos['Y'][unassigned], dtype=xy.dtype) * u.mm
            
        #- Determine source types
        source_types = desisim.simexp.get_source_types(fibermap)
        if np.any(source_types != "bgs"): raise ValueError("source types are not all BGS!") 

        desi.instrument.fiberloss_method = 'fastsim'

        source_fraction=None
        source_half_light_radius=None

        # BGS parameters based on SDSS main sample, in g-band
        # see analysis from J. Moustakas in
        # https://github.com/desihub/desitarget/blob/master/doc/nb/bgs-morphology-properties.ipynb 
        # B/T (bulge-to-total ratio): 0.48 (0.36 - 0.59).
        # Bulge Sersic n: 2.27 (1.12 - 3.60).
        # log10 (Bulge Half-light radius): 0.11 (-0.077 - 0.307) arcsec
        # log10 (Disk Half-light radius): 0.67 (0.54 - 0.82) arcsec
        # This gives
        # bulge_fraction = 0.48
        # disk_fraction  = 0.52
        # bulge_half_light_radius = 1.3 arcsec
        # disk_half_light_radius  = 4.7 arcsec
        # note we use De Vaucouleurs' law , which correspond to a Sersic index n=4
        
        # source_fraction[:,0] is DISK profile (exponential) fraction
        # source_fraction[:,1] is BULGE profile (devaucouleurs) fraction
        # 1 - np.sum(source_fraction,axis=1) is POINT source profile fraction
        # see specsim.GalsimFiberlossCalculator.create_source routine
        source_fraction=np.zeros((nspec,2)) 
        source_fraction[:,0]=0.52 # disk comp in BGS
        source_fraction[:,1]=0.48 # bulge comp in BGS       

        # source_half_light_radius[:,0] is the half light radius in arcsec for the DISK profile
        # source_half_light_radius[:,1] is the half light radius in arcsec for the BULGE profile        
        # see specsim.GalsimFiberlossCalculator.create_source routine
        source_half_light_radius=np.zeros((nspec,2))
        
        # 4.7 is angular size of z=0.1 disk, and 1.3 is angular size of z=0.1 bulge
        bgs_disk_z01 = 4.7  # in arcsec
        bgs_bulge_z01 = 1.3 # in arcsec
        
        # Convert to angular size of the objects in this sample with given redshifts
        if redshift is None:
            angscales = np.ones(len(source_types))
        else:
            # Avoid infinities
            if np.any(redshift <= 0.):
                bgs_redshifts[redshift <= 0.] = 0.0001
            angscales = ( ang_diam_dist(0.1) / ang_diam_dist(redshift) ).value
        source_half_light_radius[:,0]= bgs_disk_z01 * angscales # disk comp in BGS, arcsec
        source_half_light_radius[:,1]= bgs_bulge_z01 * angscales  # bulge comp in BGS, arcsec
            
        #- Work around randomness in specsim quickfiberloss calculations
        #- while not impacting global random state.
        #- See https://github.com/desihub/specsim/issues/83
        randstate = np.random.get_state()
        np.random.seed(seed)
        desi.simulate(sky_surface_brightness, source_fluxes=flux, focal_positions=xy, source_types=source_types,
                source_fraction=source_fraction,
                source_half_light_radius=source_half_light_radius,
                source_minor_major_axis_ratio=None,
                source_position_angle=None)
        np.random.set_state(randstate)
        return desi


class SimulatorHacked(Simulator): 
    def __init__(self, config, num_fibers=2, camera_output=True, verbose=False):
        super(SimulatorHacked, self).__init__(config, num_fibers=num_fibers, camera_output=camera_output, 
                verbose=verbose) 

    def simulate(self, sky_surface_brightness, sky_positions=None, focal_positions=None,
            source_fluxes=None, source_types=None, source_fraction=None,
            source_half_light_radius=None,
            source_minor_major_axis_ratio=None, source_position_angle=None,
            save_fiberloss=None):
        ''' The main function that I'm going to hack. 
        
        notes
        ----- 
        * calibration feature removed
        * fiberloss calculation also removed #commented out 
        '''
        # Get references to our results columns. Since table rows index
        # wavelength, the shape of each column is (nwlen, nfiber) and
        # therefore some transposes are necessary to match with the shape
        # (nfiber, nwlen) of source_fluxes and fiber_acceptance_fraction,
        # and before calling the camera downsample() and apply_resolution()
        # methods (which expect wavelength in the last index).
        wavelength = self.simulated['wavelength']
        source_flux = self.simulated['source_flux']
        fiberloss = self.simulated['fiberloss']
        source_fiber_flux = self.simulated['source_fiber_flux']
        sky_fiber_flux = self.simulated['sky_fiber_flux']
        num_source_photons = self.simulated['num_source_photons']
        num_sky_photons = self.simulated['num_sky_photons']
        nwlen = len(wavelength)

        # Position each fiber.
        if focal_positions is not None:
            if len(focal_positions) != self.num_fibers:
                raise ValueError(
                    'Expected {0:d} focal_positions.'.format(self.num_fibers))
            try:
                focal_positions = focal_positions.to(u.mm)
            except (AttributeError, u.UnitConversionError):
                raise ValueError('Invalid units for focal_positions.')
            self.focal_x, self.focal_y = focal_positions.T
            on_sky = False
            if self.verbose:
                print('Fibers positioned with focal_positions array.')
        elif sky_positions is not None:
            if len(sky_positions) != self.num_fibers:
                raise ValueError(
                    'Expected {0:d} sky_positions.'.format(self.num_fibers))
            self.focal_x, self.focal_y = self.observation.locate_on_focal_plane(
                sky_positions, self.instrument)
            on_sky = True
            if self.verbose:
                print('Fibers positioned with sky_positions array.')
        elif self.source.focal_xy is not None:
            self.focal_x, self.focal_y = np.tile(
                self.source.focal_xy, [self.num_fibers, 1]).T
            on_sky = False
            if self.verbose:
                print('All fibers positioned at config (x,y).')
        elif self.source.sky_position is not None:
            focal_x, focal_y = self.observation.locate_on_focal_plane(
                self.source.sky_position, self.instrument)
            self.focal_x = np.tile(focal_x, [self.num_fibers])
            self.focal_y = np.tile(focal_y, [self.num_fibers])
            on_sky = True
            if self.verbose:
                print('All fibers positioned at config (ra,dec).')
        else:
            raise RuntimeError('No fiber positioning info available.')

        if on_sky:
            # Set the observing airmass in the atmosphere model using
            # Eqn.3 of Krisciunas & Schaefer 1991.
            obs_zenith = 90 * u.deg - self.observation.boresight_altaz.alt
            obs_airmass = (1 - 0.96 * np.sin(obs_zenith) ** 2) ** -0.5
            self.atmosphere.airmass = obs_airmass
            if self.verbose:
                print('Calculated alt={0:.1f} az={1:.1f} airmass={2:.3f}'
                      .format(self.observation.boresight_altaz.alt,
                              self.observation.boresight_altaz.az, obs_airmass))

        # Check that all sources are within the field of view.
        focal_r = np.sqrt(self.focal_x ** 2 + self.focal_y ** 2)
        if np.any(focal_r > self.instrument.field_radius):
            raise RuntimeError(
                'A source is located outside the field of view: r > {0:.1f}'
                .format(self.instrument.field_radius))

        # Calculate the on-sky fiber areas at each focal-plane location.
        radial_fiber_size = (
            0.5 * self.instrument.fiber_diameter /
            self.instrument.radial_scale(focal_r))
        azimuthal_fiber_size = (
            0.5 * self.instrument.fiber_diameter /
            self.instrument.azimuthal_scale(focal_r))
        self.fiber_area = np.pi * radial_fiber_size * azimuthal_fiber_size

        # Get the source fluxes incident on the atmosphere.
        if source_fluxes is None:
            source_fluxes = self.source.flux_out.to(
                source_flux.unit)[np.newaxis, :]
        elif source_fluxes.shape != (self.num_fibers, nwlen):
            raise ValueError('Invalid shape for source_fluxes.')
        try:
            source_flux[:] = source_fluxes.to(source_flux.unit).T
        except AttributeError:
            raise ValueError('Missing units for source_fluxes.')
        except u.UnitConversionError:
            raise ValueError('Invalid units for source_fluxes.')

        # Calculate the source flux entering a fiber.
        source_fiber_flux[:] = (
            source_flux *
            self.atmosphere.extinction[:, np.newaxis] #* fiberloss
            ).to(source_fiber_flux.unit)
    
        # check that input sky_surface_brightness is the same size 
        # as the source_flux 
        assert source_flux.shape[0] == sky_surface_brightness.shape[0] 

        # Calculate the sky flux entering a fiber from input 
        # sky surface_brightness  
        sky_fiber_flux[:] = (
            sky_surface_brightness[:, np.newaxis] *
            self.fiber_area
            ).to(sky_fiber_flux.unit)

        # Calculate the calibration from constant unit source flux above
        # the atmosphere to number of source photons entering the fiber.
        # We use this below to calculate the flux inverse variance in
        # each camera.
        source_flux_to_photons = (
            self.atmosphere.extinction[:, np.newaxis] * # fiberloss *
            self.instrument.photons_per_bin[:, np.newaxis] *
            self.observation.exposure_time
            ).to(source_flux.unit ** -1).value

        # Calculate the mean number of source photons entering the fiber
        # per simulation bin.
        num_source_photons[:] = (
            source_fiber_flux *
            self.instrument.photons_per_bin[:, np.newaxis] *
            self.observation.exposure_time
            ).to(1).value

        # Calculate the mean number of sky photons entering the fiber
        # per simulation bin.
        num_sky_photons[:] = (
            sky_fiber_flux *
            self.instrument.photons_per_bin[:, np.newaxis] *
            self.observation.exposure_time
            ).to(1).value

        # Calculate the high-resolution inputs to each camera.
        for camera in self.instrument.cameras:
            # Get references to this camera's columns.
            num_source_electrons = self.simulated[
                'num_source_electrons_{0}'.format(camera.name)]
            num_sky_electrons = self.simulated[
                'num_sky_electrons_{0}'.format(camera.name)]
            num_dark_electrons = self.simulated[
                'num_dark_electrons_{0}'.format(camera.name)]
            read_noise_electrons = self.simulated[
                'read_noise_electrons_{0}'.format(camera.name)]

            # Calculate the mean number of source electrons detected in the CCD
            # without any resolution applied.
            num_source_electrons[:] = (
                num_source_photons * camera.throughput[:, np.newaxis])

            # Calculate the mean number of sky electrons detected in the CCD
            # without any resolution applied.
            num_sky_electrons[:] = (
                num_sky_photons * camera.throughput[:, np.newaxis])

            # Calculate the mean number of dark current electrons in the CCD.
            num_dark_electrons[:] = (
                camera.dark_current_per_bin[:, np.newaxis] *
                self.observation.exposure_time).to(u.electron).value

            # Copy the read noise in units of electrons.
            read_noise_electrons[:] = (
                camera.read_noise_per_bin[:, np.newaxis].to(u.electron).value)

        if not self.camera_output:
            # All done since no camera output was requested.
            return

        # Loop over cameras to calculate their individual responses
        # with resolution applied and downsampling to output pixels.
        for output, camera in zip(self.camera_output, self.instrument.cameras):

            # Get references to this camera's columns.
            num_source_electrons = self.simulated[
                'num_source_electrons_{0}'.format(camera.name)]
            num_sky_electrons = self.simulated[
                'num_sky_electrons_{0}'.format(camera.name)]
            num_dark_electrons = self.simulated[
                'num_dark_electrons_{0}'.format(camera.name)]
            read_noise_electrons = self.simulated[
                'read_noise_electrons_{0}'.format(camera.name)]

            # Apply resolution to the source and sky detected electrons on
            # the high-resolution grid.
            num_source_electrons[:] = camera.apply_resolution(
                num_source_electrons.T).T
            num_sky_electrons[:] = camera.apply_resolution(
                num_sky_electrons.T).T

            # Calculate the corresponding downsampled output quantities.
            output['num_source_electrons'][:] = (
                camera.downsample(num_source_electrons.T)).T
            output['num_sky_electrons'][:] = (
                camera.downsample(num_sky_electrons.T)).T
            output['num_dark_electrons'][:] = (
                camera.downsample(num_dark_electrons.T)).T
            output['read_noise_electrons'][:] = np.sqrt(
                camera.downsample(read_noise_electrons.T ** 2)).T
            output['variance_electrons'][:] = (
                output['num_source_electrons'] +
                output['num_sky_electrons'] +
                output['num_dark_electrons'] +
                output['read_noise_electrons'] ** 2)

            # Calculate the effective calibration from detected electrons to
            # source flux above the atmosphere, downsampled to output pixels.
            output['flux_calibration'][:] = 1.0 / camera.downsample(
                camera.apply_resolution(
                    source_flux_to_photons.T * camera.throughput)).T

            # Calculate the calibrated flux in this camera.
            output['observed_flux'][:] = (
                output['flux_calibration'] * output['num_source_electrons'])

            # Calculate the corresponding flux inverse variance.
            output['flux_inverse_variance'][:] = (
                output['flux_calibration'] ** -2 *
                output['variance_electrons'] ** -1)

            # Zero our random noise realization column.
            output['random_noise_electrons'][:] = 0.
        return None 
