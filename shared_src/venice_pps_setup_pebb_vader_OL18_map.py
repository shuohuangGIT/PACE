import numpy as np
from amuse.units import units
from amuse.datamodel import Particles, new_regular_grid
from venice_src.venice import Venice

from extra_funcs import *
from module_pebbleaccretion_OL18 import *
from module_diskevolution import *
from module_migration_map import *
import subprocess as sp

def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    pebble_gas_accretion = PebbleGasAccretion()
    disk_gas_evolution = DiskGasDustEvolution()
    migration = nonisothermal_Migration()

    # Add codes
    system.add_code(pebble_gas_accretion)
    system.add_code(disk_gas_evolution)
    system.add_code(migration)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix[0,1] = timestep
    system.timestep_matrix[0,2] = timestep
    system.timestep_matrix[1,2] = timestep

    system.add_channel(0,2, from_attributes = ['core_mass'], 
                to_attributes = ['core_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
                
    system.add_channel(0,2, from_attributes = ['envelope_mass'], 
                to_attributes = ['envelope_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
    
    system.add_channel(1,0, from_attributes = ['surface_solid'], 
                to_attributes = ['surface_solid'], 
                from_set_name = 'disk', to_set_name = 'disk')
    
    system.add_channel(1,0, from_attributes = ['surface_gas'], 
                to_attributes = ['surface_gas'], 
                from_set_name = 'disk', to_set_name = 'disk')
    
    system.add_channel(1,2, from_attributes = ['surface_solid'], 
                to_attributes = ['surface_solid'], 
                from_set_name = 'disk', to_set_name = 'disk')

    system.add_channel(1,2, from_attributes = ['surface_gas'], 
                to_attributes = ['surface_gas'], 
                from_set_name = 'disk', to_set_name = 'disk')
    
    system.add_channel(1,2, from_attributes = ['vd'], 
                to_attributes = ['vd'], 
                from_set_name = 'disk', to_set_name = 'disk')
    
    system.add_channel(1,2, from_attributes = ['st'], 
                to_attributes = ['st'], 
                from_set_name = 'disk', to_set_name = 'disk')

    system.add_channel(2,0, from_attributes=['semimajor_axis'], 
                to_attributes=['semimajor_axis'],
                from_set_name='planets', to_set_name='planets')

    return system, pebble_gas_accretion, disk_gas_evolution, migration

def run_single_pps (fDG, FeH, mu, v_frag, alpha, alpha_acc, gamma, temp1, beta_T, Rdisk_in, Rdisk_out, stokes_number, planets, star_mass, 
                    M_dot_ph_ex, t_birth, dt, times, N_snapshot, filename):
    # initialize venice
    system,_,_,_ = setup_single_pps(dt)

    # initialize disk evolution (code 1)
    viscous = system.codes[1].code
    viscous.initialize_keplerian_grid(
        pre_ndisk,                # grid cells
        False,              # True for linear, False for logarithmic
        Rdisk_in,    # inner disk edge
        1000|units.au,   # outer disk edge
        star_mass     # central mass
    )

    viscous = init_viscous(viscous, alpha, alpha_acc, mu, M_dot_ph_ex[0], star_mass, v_frag)

    disk_mass   = 0.1 * star_mass
    disk_radius = Rdisk_out

    temp = temp1*(viscous.grid.r.value_in(units.au))**(-beta_T)

    sigma0 = disk_mass / (2.*np.pi * disk_radius**2. * (1. - np.exp(-1.)))
    sigma = sigma0 * disk_radius/viscous.grid.r * np.exp(-viscous.grid.r/disk_radius)
    sigma[ viscous.grid.r > disk_radius ] = 1e-12 | units.g/units.cm**2 # sharp edge

    pressure = constants.kB*temp*sigma / (mu*1.008*constants.u)

    viscous.grid.column_density = sigma
    viscous.grid.pressure = pressure
    
    # dust density
    viscous.grid_user[0].value = fDG * sigma.value_in(units.g/units.cm**2)
    # dust initial grain size
    viscous.grid_user[1].value = stokes_number
    # temperature profile
    viscous.grid_user[2].value = temp.value_in(units.K)

    # initialize disk
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)
    disk.position = viscous.grid.r

    disk.surface_gas = sigma
    disk.temperature = temp
    disk.surface_solid = fDG * sigma * 10**FeH
    disk.scale_height = sound_speed(temp, mu)/np.sqrt(constants.G*star_mass/viscous.grid.r**3)

    disk.alpha = alpha
    disk.alpha_acc = alpha_acc
    disk.gamma = gamma
    disk.vd = 0. | units.cm/units.s
    disk.st = stokes_number

    system.codes[1].disk = disk

    system.codes[1].star.mass = star_mass

    # initialize pebble accretion (code 0)
    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass

    # initialize planet migration (code 2)
    system.codes[2].planets.add_particles(planets)
    system.codes[2].disk = disk
    system.codes[2].star.mass = star_mass

    # ================

    N_plot_steps = len(times)
    t = np.zeros((N_plot_steps, len(planets))) | units.Myr
    a = np.zeros((N_plot_steps, len(planets))) | units.au
    Mc = np.zeros((N_plot_steps, len(planets))) | units.MEarth
    Me = np.zeros((N_plot_steps, len(planets))) | units.MEarth
    qdisk = np.zeros((N_plot_steps, len(planets)))
    Mdot_ex = np.zeros((N_plot_steps, len(planets)))
    gas = []
    solid = []
    disk_time = []

    for i, ti in enumerate(times):
        system.codes[1].code.set_parameter(1, ( M_dot_ph_ex[i] ).value_in(units.g/units.s))
        if system.codes[1].disk_gas_mass/star_mass<5e-5:
            pass
        elif ti < t_birth:
            system.codes[1].evolve_model( ti )
            system._sync_code_to_codes(1, [0,2])
            system.codes[0].model_time = system.codes[1].model_time
            system.codes[2].model_time = system.codes[1].model_time
            system.model_time = system.codes[1].model_time

        elif i==0:
            system.codes[1].evolve_model( t_birth )
            system._sync_code_to_codes(1, [0,2])
            system.codes[0].model_time = system.codes[1].model_time
            system.codes[2].model_time = system.codes[1].model_time
            system.model_time = system.codes[1].model_time
            
            system.evolve_model( ti )

        elif times[i-1] <= t_birth:
            system.codes[1].evolve_model( t_birth )
            system._sync_code_to_codes(1, [0,2])
            system.codes[0].model_time = system.codes[1].model_time
            system.codes[2].model_time = system.codes[1].model_time
            system.model_time = system.codes[1].model_time
            
            system.evolve_model( ti )
        else:
            system.evolve_model( ti )

        t[i] = ti
        a[i] = system.codes[0].planets.semimajor_axis
        Mc[i] = system.codes[0].planets.core_mass
        Me[i] = system.codes[0].planets.envelope_mass
        qdisk[i] = system.codes[1].disk_gas_mass/star_mass
        Mdot_ex[i] = M_dot_ph_ex[i].value_in(units.MSun/units.yr)

        # control the snapshots of disk
        if i%(int(N_plot_steps/N_snapshot)) == 0:
            disk_time.append(system.codes[0].model_time.value_in(units.kyr))

            # print(int(system.codes[0].model_time.value_in(units.kyr)), times[-1].value_in(units.kyr), 'kyr')
            # print('total:', np.sum(system.codes[0].planets.dynamical_mass)/(1|units.MEarth),'Mearth')
            # print('envelope:', np.sum(system.codes[0].planets.envelope_mass)/(1|units.MEarth),'Mearth')
            # print('disk to star mass ratio:', system.codes[1].disk_gas_mass/star_mass)
            solid.append(system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2))  
            gas.append(system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2))
        
    np.savez(filename+'_disk.npz', time=disk_time, position = system.codes[0].disk.position.value_in(units.AU), 
                gas=gas, solid=solid)

    np.savez(filename+'_planet.npz', time=t.value_in(units.kyr), Mc=Mc.value_in(units.MEarth), 
                Me=Me.value_in(units.MEarth), a = a.value_in(units.au), qdisk=qdisk, Mdot_ex=Mdot_ex, fDG =fDG, FeH=FeH, mu=mu, 
                vfrag = v_frag, alpha=alpha, alpha_acc=alpha_acc, gamma=gamma, temp1 = temp1.value_in(units.K), 
                betaT = beta_T, Rdisk_in = Rdisk_in.value_in(units.au), Rdisk_out = Rdisk_out.value_in(units.au), 
                stokes_number = stokes_number, star_mass = star_mass.value_in(units.MSun), t_birth=t_birth.value_in(units.Myr))

    system.stop()
    return 0


if __name__ == '__main__':
    
    # initialize planets.
    a = [2.27] | units.AU
    M = [1e-2 for i in range(len(a))] | units.MEarth
    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0|units.g,
        semimajor_axis = a,
        isohist = [False for i in range(len(M))]
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 1 | units.kyr # timestep of the matrix
    end_time = 1000. #| units.kyr
    times = np.linspace(0, end_time, 1001) | units.kyr

    N_plot_disk = 10
    
    M_star=0.4 | units.MSun

    FeH = 0
    mu = 2.3
    alpha = 1e-6
    alpha_acc = 1e-6
    gamma = 7/5
    temp1 = 150|units.K
    beta_T = 3/7

    v_frag = 1e3 # cm/s

    fDG = 0.0134
    Rdisk_in = 0.04 | units.AU
    Rdisk_out = 85 | units.AU
    stokes_number = 1e-3
    # M_dot_ph_in = 1e-8 | units.MSun/units.yr
    M_dot_ph_ex = 1e-10*np.ones(len(times)) | units.MSun/units.yr

    t_birth = 0. | units.Myr
    filename = './'
    system = run_single_pps(fDG, FeH, mu, v_frag, alpha, alpha_acc, gamma, temp1, beta_T, Rdisk_in, Rdisk_out, stokes_number, planets, M_star, M_dot_ph_ex, t_birth, dt, times, N_plot_disk, filename)
    
    sp.run(['python3', 'venice_pps_plot.py'])
