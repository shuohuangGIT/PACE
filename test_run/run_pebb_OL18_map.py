import numpy as np
import scipy.stats as stats
import os
import sys
import subprocess as sp
from multiprocessing import Process, Queue, Pool

srcfile = '../shared_src/'
sys.path.append(srcfile)

from extra_funcs import get_sequential_indices
from venice_pps_setup_pebb_vader_OL18_map import run_single_pps
from extra_funcs import dynamical_mass
from amuse.units import units
from amuse.datamodel import Particles
from amuse.io import read_set_from_file

np.random.seed(0)

datafile = 'planet_evo/'
if ((os.path.exists(datafile))!=1):
    sp.run(['mkdir', datafile])

### Initial conditions
## cluster data/properties
disk_keys = []
M_dot_exts = [] # | units.MSun/units.yr
ages = [] #| units.kyr
star_mass = []

filepath = '/data2/wilhelm/sim_archive/torch/torch_ppds_p2_vm_s1/data/'
i0 = 130
i1 = 273
indexs, time = get_sequential_indices(i0, i1, srcfile+'cluster_data/', dt=0.01|units.Myr)
for i in indexs:
    filename = srcfile+'cluster_data/viscous_particles_plt_i%05d.hdf5'%i
    if (os.path.exists(filename)):
        env_data = read_set_from_file(filename, "hdf5")
        for disk_key in env_data.disk_key:
            if disk_key == -1:
                continue
                
            if disk_key in disk_keys:
                M_dot_exts[disk_keys.index(disk_key)].append(env_data.outer_photoevap_rate[env_data.disk_key==disk_key][0].value_in(units.MSun/units.yr))
                ages[disk_keys.index(disk_key)].append(env_data.age[env_data.disk_key==disk_key][0].value_in(units.kyr))
            else:
                disk_keys.append(disk_key)
                star_mass.append(env_data.mass[env_data.disk_key==disk_key][0].value_in(units.MSun)) # mass bug solved
                M_dot_exts.append([(env_data.outer_photoevap_rate[env_data.disk_key==disk_key][0]).value_in(units.MSun/units.yr)])
                ages.append([env_data.age[env_data.disk_key==disk_key][0].value_in(units.kyr)])

# The number of planets/disks.
N = len(disk_keys)

random_par_name = '../random_par.npz'
if os.path.exists(random_par_name):
    print('Loading parameters...')
    random_par_data = np.load(random_par_name)
    random_par_data.keys()
    FeH_rand = random_par_data['FeH_rand']
    alpha_rand = random_par_data['alpha_rand']
    alpha_acc_rand = random_par_data['alpha_acc_rand']
    Rdisk_in_rand = random_par_data['Rdisk_in_rand']
    Rdisk_out_rand = random_par_data['Rdisk_out_rand']
    tbirth_rand = random_par_data['tbirth_rand']
    beta_L = random_par_data['beta_L']
    pmass_rand = random_par_data['pmass_rand']
    psma_rand = random_par_data['psma_rand']
    fDG = random_par_data['fDG']
    mu = random_par_data['mu']
    beta_T = random_par_data['beta_T']
    stokes_number = random_par_data['stokes_number']
    v_frag = random_par_data['v_frag']
    gamma = random_par_data['gamma']
    del random_par_data.f
    random_par_data.close()
    
else:
    ## steller/disk properties
    FeH_rand = 0.02*np.ones(N)
    lower, upper, mu, sigma = np.log10(0.1), np.log10(100), np.log10(4), 0.5 # Batygin et al. 2023
    logPdisk_in_rand_cal = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)
    Pdisk_in_rand = 10**logPdisk_in_rand_cal.rvs(N) # unit: days

    Rdisk_in_rand = (Pdisk_in_rand/365.24)**(2/3)*np.array(star_mass)**(1/3)
    Rdisk_out_rand = 117 * np.array(star_mass)**0.45  # unit: au

    ## random conditions:
    alpha_rand = 10**(-4)*np.ones(N)
    alpha_acc_rand = 10**(-3)*np.ones(N)
    pmass_rand = 10**-2*np.ones(N)
    tbirth_rand = np.random.uniform(0,0.5,N)
    beta_L = 2 # mass slope of mass-luminosity relation for proto stars.

    # psma_in = np.array(star_mass)**((2*beta_L-1)/3) # outside iceline
    psma_in = (100/365.24)**(2/3)*np.array(star_mass)**(1/3)
    psma_out = Rdisk_out_rand
    psma_rand = 10**np.random.uniform(np.log10(psma_in), np.log10(psma_out))

    # some fixed parameters
    fDG = 0.0149 # solar dust to gas ratio
    mu = 2.3 # mean molecular weight
    # beta_T  = 3/7 # warning, does not have minors sign
    beta_T = 0.5
    stokes_number = 0.001
    v_frag =  1e3 # cm/s
    gamma = 7/5

    print('Generating parameters...')
    np.savez(random_par_name, FeH_rand=FeH_rand, alpha_rand=alpha_rand, alpha_acc_rand=alpha_acc_rand, Rdisk_in_rand=Rdisk_in_rand, 
            Rdisk_out_rand=Rdisk_out_rand, tbirth_rand=tbirth_rand, pmass_rand=pmass_rand, psma_rand=psma_rand, fDG=fDG, mu=mu, beta_T=beta_T, beta_L=beta_L, stokes_number=stokes_number,
            v_frag=v_frag, gamma=gamma)

tbirth_rand = 0.*np.ones(N) # unit: Myr

n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']

print("The number of cores available from SLURM: {}".format(n_cores))

pool = Pool(processes=int(n_cores)) # number of processes <---------------
results=[]

for i in range(N):
    filename = datafile +'%06d'%i
    if ((os.path.exists(filename+'_planet.npz'))*(os.path.exists(filename+'_disk.npz'))==1):
        print(filename, 'exist, continue!')
        continue

    FeH = FeH_rand[i] # initial metalicity, default: 0
    alpha = alpha_rand[i] # alpha parameter, default: 2e-3
    alpha_acc = alpha_acc_rand[i]
    Rdisk_in = Rdisk_in_rand[i] | units.AU # initial disk outer radius, default: 0.03
    Rdisk_out = Rdisk_out_rand[i] | units.AU # initial disk outer radius, default: 30
    t_birth = tbirth_rand[i] | units.Myr # planet birth time

    M = [pmass_rand[i]] | units.MEarth # initial planet mass
    a = [psma_rand[i]] | units.AU # initial planet sma
    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0|units.g,
        semimajor_axis = a,
        isohist = False # it will become true when planet first reach pebble isolation mass
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    # host star properties
    M_star = star_mass[i] | units.MSun
    
    # temp1 = 150*star_mass[i]**((2*beta_L-1)/7) | units.K
    temp1 = 150*star_mass[i]**(1/4) | units.K
    
    M_dot_ph_ex = np.array(M_dot_exts[i]) | units.MSun/units.yr
    
    times = np.array(ages[i]) | units.kyr

    if len(M_dot_ph_ex) != len(times):
        print('error: error with reading cluster data!')
        sys.exit(0)

    dt = 1 | units.kyr # timestep of the matrix (fixed)
    N_plot_disk = 20 # number of saved snapshots

    print('fDG:', fDG, 'FeH:', FeH, 'alpha:', alpha, 'alpha_acc:', alpha_acc, 'gamma:', gamma, 'temp1:', temp1.value_in(units.K), 'betaT:', beta_T, 'R_in:', Rdisk_in_rand[i], 'R_out:', Rdisk_out_rand[i], 
            'St:', stokes_number, 'star_mass:', star_mass[i], 't_birth:', tbirth_rand[i], 'psma:', psma_rand[i])

    argL = (fDG, FeH, mu, v_frag, alpha, alpha_acc, gamma, temp1, beta_T, Rdisk_in, Rdisk_out, 
            stokes_number, planets, M_star, M_dot_ph_ex, t_birth, dt, times, N_plot_disk, filename)
    results.append (pool.apply_async(run_single_pps, argL))

# clean up
pool.close()
pool.join()

planet_results = np.array([i.get() for i in results])
print(planet_results)