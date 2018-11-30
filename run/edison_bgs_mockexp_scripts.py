'''

Code for generating and submitting redrock jobs on edison 
for simulated spectra of mock BGS exposures


'''
import numpy as np 
from feasibgs import util as UT 


def bgs_mockexp_simspec_redrock(ibatch, nbatch=10): 
    ''' run slurm script to run redrock on simulated spectra 
    from BGS mock exposures 
    '''
    if ibatch > nbatch: raise ValueError

    # read in exposure numbers 
    f_exp = ''.join(["/global/cscratch1/sd/chahah/feasibgs/bgs_survey_exposures.withsun.iexp_metabin.dat"]) 
    iexps = np.loadtxt(f_exp, unpack=True) 
    nexps = len(iexps) 
    iexp_batch = iexps[(ibatch-1)*(nexps // nbatch):ibatch*(nexps // nbatch)]
    iexp_batch_str = ' '.join([str(ii) for ii in iexp_batch.astype(int)])

    fjob = ''.join(['edison_bgs_mockexp_simspec_', str(ibatch), 'of', str(nbatch), '.slurm']) 
    print('--- writing %s ---' % fjob) 
    
    # estimate the hours required 
    mins_tot = float(len(iexp_batch))*9.*2
    h = str(int(mins_tot // 60)).zfill(2)#int(np.ceil(((nreals[1] - nreals[0])+1)*9./60.))
    m = str(int(10.*np.ceil((mins_tot % 60.)/10.))).zfill(2)
    queue = 'regular'
    if (int(h) == 0) and (int(m) < 30): 
        queue = 'debug'

    jb = '\n'.join([ 
        '#!/bin/bash', 
        '#SBATCH -q '+queue, 
        '#SBATCH -N 1', 
        '#SBATCH -t '+h+':'+m+':00', #'#SBATCH -t 00:10:00'
        '#SBATCH -J bgs_mockexp_'+str(ibatch)+'of'+str(nbatch), 
        '#SBATCH -o bgs_mockexp_'+str(ibatch)+'of'+str(nbatch)+'.o', 
        '', 
        'now=$(date +"%T")',
        'echo "start time ... $now"', 
        '', 
        'module load python/2.7-anaconda', 
        'source activate myenv0', 
        'source /project/projectdirs/desi/software/desi_environment.sh 18.3', 
        '', 
        'export OMP_NUM_THREADS=1', 
        'dir_spec=$FEASIBGS_DIR"spectra/gamadr3_legacydr7/"', 
        '', 
        'for iexp in '+iexp_batch_str+'; do',
        '\tfor skymodel in "KS" "newKS"; do', 
        '\t\techo "-- $skymodel sky model --"', 
        '\t\tf_str="g15.sim_spectra.mockexp_block.2of64.480.iexp$iexp."$skymodel"sky"',
        '\t\tf_spec=$dir_spec$f_str".fits"', 
        '\t\tf_redr=$dir_spec$f_str".rr.fits"', 
        '\t\tf_zout=$dir_spec$f_str".rr.h5"', 
        '\t\techo "-- "$f_spec" --"', 
        '\t\tif [ ! -f $f_spec ]; then', 
        '\t\t\techo "$f_spec not found!"', 
        '\t\tfi', 
        '\t\tif [ ! -f $f_redr ]; then', 
        '\t\t\t# takes ~9 mins',   
        '\t\t\trrdesi --mp 24 --zbest $f_redr --output $f_zout $f_spec', 
        '\t\tfi', 
        '\tdone', 
        'done', 
        'now=$(date +"%T")',
        'echo "end time ... $now"'
        ]) 
    job = open(fjob, 'w') 
    job.write(jb) 
    job.close() 

    UT.nersc_submit_job(fjob) 
    return None 


if __name__=='__main__': 
    for i in range(1,11): 
        bgs_mockexp_simspec_redrock(i, nbatch=10)

