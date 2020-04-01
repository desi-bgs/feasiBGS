#!/bin/python
'''

script for compiling single CMX exposures into coadds and runnign them through
redrock

'''
import os 
import glob
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


def get_ztrue(tileid, date, exp, spec): 
    ''' compile redshift truth table for each of the (tileid, date, exp, spec)
    spectra.
    '''
    from pydl.pydlutils.spheregroup import spherematch
    # read coadd 
    fcoadd  = os.path.join(dir_output, 
            'coadd-%s-%s-%s-%s.fits' % (str(tileid), date, spec, exp.zfill(8)))
    coadd = fitsio.read(fcoadd) 
    print('coadd --- %s' % os.path.basename(fcoadd))
    # get ra and dec range of coadd
    ra_min, ra_max = coadd['TARGET_RA'].min(), coadd['TARGET_RA'].max() 
    dec_min, dec_max = coadd['TARGET_DEC'].min(), coadd['TARGET_DEC'].max() 

    has_ztrue = np.zeros(len(coadd['TARGET_RA'])).astype(bool) 
    ztrue = np.repeat(-999., len(coadd['TARGET_RA']))
    
    # loop through Rongpu's matched truth tables 
    dir_match = lambda ns:
        '/global/cfs/cdirs/desi/target/analysis/truth/dr9sv/%s/matched' % ns 

    for nors in ['north', 'south']: 
        fmatches = glob.glob(os.path.join(dir_match(nors),  '[!ls-]*match.fits'))
        for fmatch in fmatches: 
            # read match 
            truth = fitsio.read(fmatch)
            print('     match to ... %s' % os.path.basename(fmatch))
            
            # check if exposure is within RA, Dec range 
            _ra_min, _ra_max = truth['RA'].min(), truth['RA'].max() 
            _dec_min, _dec_max = truth['DEC'].min(), truth['DEC'].max() 

            if ((_ra_min > ra_max) | (_ra_max < ra_min) | (_dec_min > dec_max)
                    | (_dec_max < dec_min)): 
                print('     out of range')
                continue 
            # match RA/Dec 
            match = spherematch(
                    truth['RA'], truth['DEC'],
                    coadd['TARGET_RA'], coadd['TARGET_DEC'],
                    0.000277778)
            if len(match[0]) == 0: # no matches
                print('     no matches')
                continue 
            print('     has %i matches' % len(match[0]))

            if np.sum(has_ztrue[match[1]]) > 0: 
                # target already has true redshift 
                conflict = np.abs(ztrue[match[1]] - truth['ZBEST'][match[0]]) > 0.01
                print('     has %i conflicting redshifts' % np.sum(conflict))
            else: 
                conflict = np.zeros(len(match[0])).astype(bool) 
            ztrue[match[1]]     = truth['ZBEST'][match[0]]
            has_ztrue[match[1]] = True
            ztrue[match[1]][conflict] = -999.
    
    fztrue  = os.path.join(dir_output, 
            'ztrue-%s-%s-%s-%s.hdf5' % (str(tileid), date, spec, exp.zfill(8)))
    f = h5py.File(fztrue, 'w') 
    f.create_dataset('has_ztrue', data=has_ztrue)
    f.create_dataset('ztrue', data=ztrue)
    f.close()
    return None 


if __name__=="__main__": 
    bgs_minisv_tiles    = [70500, 70502, 70510]
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
                    get_ztrue(tileid, date, exp, spec)
