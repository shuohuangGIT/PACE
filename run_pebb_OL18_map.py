from venice_pps_setup_pebb_vader_OL18_map import run_single_pps
from extra_funcs import dynamical_mass
from amuse.units import units
from amuse.datamodel import Particles
from amuse.io import read_set_from_file

import numpy as np
import scipy.stats as stats
import os
import sys
import subprocess as sp
from multiprocessing import Process, Queue, Pool

np.random.seed(0)

inputfile = "planet_in_fixTem0.txt"
outputfile = "planet_out_fixTem0.txt"
datafile = 'planet_evo/'
if ((os.path.exists(outputfile))==1):
    sp.run(['rm', outputfile])
if ((os.path.exists(inputfile))==1):
    sp.run(['rm', inputfile])
if ((os.path.exists(datafile))!=1):
    sp.run(['mkdir', datafile])

f_in = open(inputfile, "a")
f_out = open(outputfile, "a") # open the output file

### Initial conditions
## cluster data/properties
disk_keys = []
M_dot_exts = [] # | units.MSun/units.yr
ages = [] #| units.kyr
star_mass = []
for i in range(500):
    filename = 'cluster_data/viscous_particles_plt_i%05d.hdf5'%i
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
                star_mass.append(env_data.mass[0].value_in(units.MSun))
                M_dot_exts.append([(env_data.outer_photoevap_rate[env_data.disk_key==disk_key][0]).value_in(units.MSun/units.yr)])
                ages.append([env_data.age[env_data.disk_key==disk_key][0].value_in(units.kyr)])

# The number of planets/disks.
N = len(disk_keys)
FeH_rand = 0.02*np.ones(N)

## steller/disk properties
lower, upper, mu, sigma = 2, 12, 7, 2 # Batygin et al. 2023
Pdisk_in_rand_cal = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)
Pdisk_in_rand = Pdisk_in_rand_cal.rvs(N) # unit: days

Rdisk_in_rand = (Pdisk_in_rand/365.24)**(2/3)*np.array(star_mass)**(1/3)
Rdisk_out_rand = 200 * np.array(star_mass)**0.45  #unit: au

## random conditions:
alpha_rand = 10**(-6)*np.ones(N)
alpha_acc_rand = 10**(-6)*np.ones(N)
pmass_rand = 10**-2*np.ones(N)
tbirth_rand = np.random.uniform(0,0.5,N)

psma_rand = 10**np.random.uniform(np.log10(Rdisk_in_rand), np.log10(Rdisk_out_rand))

# n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']
n_cores = 8
print("The number of cores available from SLURM: {}".format(n_cores))

pool = Pool(processes=int(n_cores)) # number of processes <---------------
results=[]

for i in range(1):
    FeH = FeH_rand[i] # initial metalicity, default: 0
    alpha = alpha_rand[i] # alpha parameter, default: 2e-3
    alpha_acc = alpha_acc_rand[i]
    Rdisk_in = Rdisk_in_rand[i] | units.AU # initial disk outer radius, default: 0.03
    Rdisk_out = Rdisk_out_rand[i] | units.AU # initial disk outer radius, default: 30
    # t_birth = tbirth_rand[i] | units.Myr # planet birth time
    t_birth = 0. | units.Myr # planet birth time

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
    beta_L = 1 # mass slope of mass-luminosity relation for proto stars.

    temp1 = 150*star_mass[i]**((2*beta_L-1)/7) | units.K
    # M_dot_ph_ex = np.array(M_dot_exts[i]) | units.MSun/units.yr
    
    times = np.array(ages[i]) | units.kyr
    M_dot_ph_ex = 1e-10*np.ones(len(times)) | units.MSun/units.yr

    if len(M_dot_ph_ex) != len(times):
        print('error: error with reading cluster data!')
        sys.exit(0)
    # some fixed parameters
    fDG = 0.0149 # solar dust to gas ratio
    mu = 2.4 # mean molecular weight
    beta_T  = 3/7 # warning, does not have minors sign
    stokes_number = 0.001
    v_frag =  1e3 # cm/s
    gamma = 7/5

    dt = 1 | units.kyr # timestep of the matrix (fixed)
    N_plot_disk = 20 # number of saved snapshots

    filename = datafile +'%06d'%i

    print(fDG, FeH, alpha, alpha_acc, gamma, temp1.value_in(units.K), beta_T, Rdisk_in_rand[i], Rdisk_out_rand[i], 
            stokes_number, star_mass[i], tbirth_rand[i], psma_rand[i])

    # results.append(run_single_pps(fDG, FeH, mu, v_frag, alpha, alpha_acc, gamma, temp1, beta_T, Rdisk_in, Rdisk_out, 
    #         stokes_number, planets, M_star, M_dot_ph_ex, t_birth, dt, times, N_plot_disk, filename))
    argL = (fDG, FeH, mu, v_frag, alpha, alpha_acc, gamma, temp1, beta_T, Rdisk_in, Rdisk_out, 
            stokes_number, planets, M_star, M_dot_ph_ex, t_birth, dt, times, N_plot_disk, filename)
    results.append (pool.apply_async(run_single_pps, argL))

    # write input
    for data in ([fDG, FeH, alpha, alpha_acc, gamma, temp1.value_in(units.K), beta_T, Rdisk_in_rand[i], Rdisk_out_rand[i], 
            stokes_number, star_mass[i], tbirth_rand[i]]):
        f_in.write(str(data)+' ')
    for planet in (planets):
        for data in ([planet.core_mass.value_in(units.MEarth), planet.envelope_mass.value_in(units.MEarth), planet.semimajor_axis.value_in(units.au)]):
            f_in.write(str(data)+' ')
    f_in.write('\n')

planet_results = np.array([i.get() for i in results])

# write output
for planets in (planet_results):
    for planet in (planets):
        for data in planet:
            f_out.write(str(data)+' ')
    f_out.write('\n')


f_in.close() # close the output file
f_out.close() # close the output file
