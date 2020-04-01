#!/bin/python
'''

script for compiling single CMX exposures into coadds and runnign them through
redrock

'''
import os 
import glob
import h5py
import fitsio
import numpy as np 


dir_redux = "/global/cfs/cdirs/desi/spectro/redux/daily" 
dir_output = "/global/cfs/cdirs/desi/users/chahah/bgs_exp_coadd"


def get_dates(tileid): 
    dates = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), "*"))
    dates = [os.path.basename(date) for date in dates]
    return dates 


def get_exposures(tileid, date): 
    cframes = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), date,
        'cframe-b*.fits')) 
    exps = [cframe.split('-')[-1].split('.fits')[0] for cframe in cframes]
    return np.unique(exps)


def get_spectograph(tileid, date, exp): 
    cframes = glob.glob(os.path.join(dir_redux, 'tiles', str(tileid), date,
        'cframe-b*-%s.fits' % exp.zfill(8)))
    spectographs = [os.path.basename(cframe).split('-')[1].split('b')[-1] 
            for cframe in cframes]
    return spectographs


def coadd(tileid, date, exp, spec): 
    ''' combine spectra from b r z spectographs for given (tileid, date, exp,
    spec) into a single coadd. 
    '''
    cframe = os.path.join(dir_redux, 'tiles', str(tileid), date,
            'cframe-[brz]%s-%s.fits' % (spec, exp.zfill(8)))
    fcoadd = os.path.join(dir_output, 
            'coadd-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))
    cmd = 'desi_coadd_spectra --coadd-cameras -i %s -o %s' % (cframe, fcoadd)
    os.system(cmd) 
    return None 


def rr_coadd(tileid, date, exp, spec): 
    ''' run redrock on specified coadd  
    '''
    fcoadd  = os.path.join(dir_output, 
            'coadd-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))
    frr     = os.path.join(dir_output, 
            'redrock-%s-%s-%s-%s.h5' % (str(tileid), date, spec, exp.zfill(8)))
    fzbest  = os.path.join(dir_output, 
            'zbest-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))

    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -N 1", 
        "#SBATCH -C haswell", 
        "#SBATCH -q regular", 
        '#SBATCH -J rr_%s_%s' % (exp, spec),
        '#SBATCH -o _rr_%s_%s.o' % (exp, spec),
        "#SBATCH -t 00:10:00", 
        "", 
        "export OMP_NUM_THREADS=1", 
        "export OMP_PLACES=threads", 
        "export OMP_PROC_BIND=spread", 
        "", 
        "", 
        "conda activate desi", 
        "", 
        "srun -n 32 -c 2 --cpu-bind=cores rrdesi_mpi -o %s -z %s %s" % (frr, fzbest, fcoadd), 
        ""]) 
    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()

    os.system('sbatch script.slurm') 
    os.system('rm script.slurm') 
    return None 


def get_ztrue(tileid, date, exp, spec, clobber=False): 
    ''' compile redshift truth table for each of the (tileid, date, exp, spec)
    spectra.
    '''
    from pydl.pydlutils.spheregroup import spherematch
    fztrue  = os.path.join(dir_output, 
            'ztrue-%s-%s-%s-%s.hdf5' % (str(tileid), date, spec, exp.zfill(8)))
    if os.path.isfile(fztrue) and not clobber: 
        return None 
    # read coadd 
    fcoadd  = os.path.join(dir_output, 
            'coadd-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))
    if not os.path.isfile(fcoadd): return None 
    coadd = fitsio.read(fcoadd) 
    print('coadd --- %s' % os.path.basename(fcoadd))
    # get ra and dec range of coadd
    ra_min, ra_max = coadd['TARGET_RA'].min(), coadd['TARGET_RA'].max() 
    dec_min, dec_max = coadd['TARGET_DEC'].min(), coadd['TARGET_DEC'].max() 
    print('%.f < RA < %.f, %.f < DEC < %.f' % (ra_min, ra_max, dec_min, dec_max))

    has_ztrue = np.zeros(len(coadd['TARGET_RA'])).astype(bool) 
    ztrue = np.repeat(-999., len(coadd['TARGET_RA']))
    
    # only imaging 
    imaging_cat_kw = ['cfhtls-d3', 'hsc-pdr1', 'hsc-pdr2', 'SpIES', 'cosmos-acs']

    # loop through Rongpu's matched truth tables 
    dir_match = lambda ns: \
            '/global/cfs/cdirs/desi/target/analysis/truth/dr9sv/%s/matched' % ns 

    for nors in ['north', 'south']: 
        fmatches = glob.glob(os.path.join(dir_match(nors),  '[!ls-]*match.fits'))
        for fmatch in fmatches: 
            imaging = False
            for k in imaging_cat_kw: 
                if k in fmatch: imaging =True
            if imaging: continue
            # read match 
            truth = fitsio.read(fmatch)
            print('     match to ... %s %s' % (nors, os.path.basename(fmatch)))

            ra_col, dec_col, z_col = None, None, None 
            for k in ['ra', 'RA', 'R.A.', 'RAJ2000', 'ALPHA', 'alpha']: 
                if k in truth.dtype.names: 
                    ra_col = k 

            for k in ['dec', 'DEC', 'Dec', 'Dec.', 'DEJ2000', 'DECJ2000',
                    'DELTA', 'delta']: 
                if k in truth.dtype.names: 
                    dec_col = k 

            for k in ['cz', 'z', 'z1', 'Z', 'ZBEST', 'V', 'ZHELIO']: 
                if k in truth.dtype.names: 
                    z_col = k 
                    z_factor = 1.
                    if k in ['cz', 'V']: 
                        z_factor = 1./3.e5
            
            # check if exposure is within RA, Dec range 
            _ra_min, _ra_max = truth[ra_col].min(), truth[ra_col].max() 
            _dec_min, _dec_max = truth[dec_col].min(), truth[dec_col].max() 

            if ((_ra_min > ra_max) | (_ra_max < ra_min) | (_dec_min > dec_max)
                    | (_dec_max < dec_min)): 
                print('     ... out of range: %.f < RA < %.f, %.f < DEC < %.f' % 
                        (_ra_min, _ra_max, _dec_min, _dec_max))
                continue 
            # match RA/Dec 
            match = spherematch(
                    truth[ra_col], truth[dec_col],
                    coadd['TARGET_RA'], coadd['TARGET_DEC'],
                    0.000277778)
            if len(match[0]) == 0: # no matches
                print('     ... no matches')
                continue 
            print('     ... has %i matches' % len(match[0]))

            conflict = np.zeros(len(match[0])).astype(bool) 
            if np.sum(has_ztrue[match[1]]) > 0: 
                # target already has true redshift 
                already_has = np.arange(len(match[1]))[has_ztrue[match[1]]]
                z_already = ztrue[match[1]][already_has]
                z_new = z_factor * (truth[z_col][match[0]][already_has]).astype(float)
                has_conflict = np.abs(z_already - z_new) > 0.01
                conflict[already_has[has_conflict]] = True 
                print('     has %i conflicting redshifts' % np.sum(conflict))

            ztrue[match[1]]     = z_factor * (truth[z_col][match[0]]).astype(float)
            has_ztrue[match[1]] = True
            ztrue[match[1]][conflict] = -999.

    print('%i total maches' % np.sum(has_ztrue))
    
    f = h5py.File(fztrue, 'w') 
    f.create_dataset('has_ztrue', data=has_ztrue)
    f.create_dataset('ztrue', data=ztrue)
    f.close()
    return None 


if __name__=="__main__": 
    bgs_minisv_tiles    = [70502, 70510] #[70500, 70502, 70510]
    bgs_sv0_tiles       = [66000, 66014, 66003]
    bgs_tiles = bgs_minisv_tiles + bgs_sv0_tiles 
    
    for tile in bgs_tiles: 
        dates = get_dates(tile)
        for date in dates: 
            exps = get_exposures(tile, date)
            for exp in exps: 
                spectographs = get_spectograph(tile, date, exp)
                for spec in spectographs: 
                    #coadd(tile, date, exp, spec)
                    #rr_coadd(tile, date, exp, spec)
                    get_ztrue(tile, date, exp, spec)
