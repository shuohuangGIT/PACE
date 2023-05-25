from venice_pps_setup_plts import run_single_pps
from extra_funcs import dynamical_mass
from amuse.units import units
from amuse.datamodel import Particles
import numpy as np
import os
import subprocess as sp
from multiprocessing import Process, Queue, Pool

# initialize planets.
N = 500
inputfile = "planet_in.txt"
outputfile = "planet_out.txt"
datafile = 'planet_evo/'
if ((os.path.exists(outputfile))==1):
    sp.run(['rm', outputfile])
if ((os.path.exists(inputfile))==1):
    sp.run(['rm', inputfile])
if ((os.path.exists(datafile))!=1):
    sp.run(['mkdir', datafile])

f_in = open(inputfile, "a")
f_out = open(outputfile, "a") # open the output file

fg_rand = 10**np.random.uniform(-1,1,N)
FeH_rand = np.random.uniform(-1,1,N)
alpha_rand = 10**np.random.uniform(-4,-2,N)
Rdisk_in_rand = 10**np.random.uniform(np.log10(0.04), np.log10(0.25),N)
Rdisk_out_rand = 10**np.random.uniform(np.log10(30), np.log10(50),N)
lifetime_rand = 10**np.random.uniform(-1,0.3,N)
pmass_rand = 10**np.random.uniform(-8,-5,N)
psma_rand = 10**np.random.uniform(np.log10(2.7), np.log10(Rdisk_out_rand*0.9))

pool = Pool(processes=8) # number of processes <---------------
results=[]

for i in range(N):
    fg = fg_rand[i] # initial disk mass, default: 5?
    FeH = FeH_rand[i] # initial metalicity, default: 0
    alpha = alpha_rand[i] # alpha parameter, default: 2e-3
    Rdisk_in = Rdisk_in_rand[i] | units.AU # initial disk outer radius, default: 0.03
    Rdisk_out = Rdisk_out_rand[i] | units.AU # initial disk outer radius, default: 30
    lifetime = lifetime_rand[i] | units.Myr # disk e-folding time (with wind term), default: 1.2

    M = [pmass_rand[i]] | units.MEarth # initial planet mass
    a = [psma_rand[i]] | units.AU # initial planet sma
    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0|units.g,
        semimajor_axis = a
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    # host star properties
    M_star=1|units.MSun
    R_star=1|units.RSun
    T_star=5770|units.K

    # some fixed parameters
    pg0 = -1 # gas density slope 
    fDG = 0.0149 # solar dust to gas ratio
    mu = 2.4 # mean molecular weight

    dt = 10 | units.kyr # timestep of the matrix (fixed)
    end_time = lifetime * 4*fg
    # end_time = 0 |units.s
    dt_plot_planet = end_time/400 # timestep to save the planet data
    N_plot_disk = 40

    filename = datafile +'%.2f'%fg +'_%.2f'%FeH +'_%.5f'%alpha +'_%.4f'%Rdisk_in_rand[i] +'_%.2f'%Rdisk_out_rand[i] +'_%.3f'%lifetime_rand[i] +'_%.9f'%pmass_rand[i] +'_%.3f'%psma_rand[i]
    argL = (fg, pg0, fDG, FeH, mu, alpha, Rdisk_in, Rdisk_out, lifetime, planets, M_star, R_star, T_star, dt, end_time, dt_plot_planet, N_plot_disk, filename)
    results.append (pool.apply_async(run_single_pps, argL))

    # write input
    for data in ([fg, FeH, alpha, Rdisk_in_rand[i], Rdisk_out_rand[i], lifetime_rand[i], pmass_rand[i], psma_rand[i]]):
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
