import numpy as np
from amuse.units import units
from amuse.datamodel import Particles, new_regular_grid
from venice_src.venice import Venice

from extra_funcs import *
from module_pebbleaccretion import *
from module_gasevolution import *
from module_migration import *
import subprocess as sp

def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    pebble_gas_accretion = PebbleGasAccretion()
    disk_gas_evolution = DiskGasEvolution()
    typeI_migration = TypeIMigration()

    # Add codes
    system.add_code(pebble_gas_accretion)
    system.add_code(disk_gas_evolution)
    system.add_code(typeI_migration)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix[0,1] = timestep
    system.timestep_matrix[0,2] = timestep/3
    system.timestep_matrix[1,2] = timestep

    system.add_channel(0,2, from_attributes = ['core_mass'], 
                to_attributes = ['core_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
                
    system.add_channel(0,2, from_attributes = ['envelope_mass'], 
                to_attributes = ['envelope_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
    system.add_channel(0,2, from_attributes = ['surface_solid'], 
                to_attributes = ['surface_solid'], 
                from_set_name = 'disk', to_set_name = 'disk')

    system.add_channel(1,0, from_attributes = ['surface_gas'], 
                to_attributes = ['surface_gas'], 
                from_set_name = 'disk', to_set_name = 'disk')
    system.add_channel(1,2, from_attributes = ['surface_gas'], 
                to_attributes = ['surface_gas'], 
                from_set_name = 'disk', to_set_name = 'disk')

    system.add_channel(2,0, from_attributes=['semimajor_axis'], 
                to_attributes=['semimajor_axis'],
                from_set_name='planets', to_set_name='planets')

    return system, pebble_gas_accretion, disk_gas_evolution, typeI_migration


def run_single_pps (fg, beta, xi, fDG, FeH, mu, alpha, Rdisk_in, Rdisk_out, lifetime, stokes_number, planets, star_mass, star_radius, star_teff, dt, end_time, dt_plot, N_snapshot, filename):
    # initialize disk
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)
    # sigmag_0 = 2400 | units.g/units.cm**2
    # disk.position = Rdisk0(Rdisk_in, Rdisk_out, pre_ndisk)
    # disk.surface_gas = fg*sigmag_0 * (disk.position.value_in(units.au))**(-beta)

    # T1 = 117|units.K
    # T = T1*(disk.position.value_in(units.au))**(-xi)
    # disk.surface_solid = disk.surface_gas *fDG * 10**FeH
    # disk.scale_height = scale_height(sound_speed(T, mu), star_mass, disk.position)
    # disk.alpha=alpha

    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)
    disk.position = Rdisk0(Rdisk_in, Rdisk_out, pre_ndisk)
    disk.surface_gas = sigma_g0(fg, -beta, disk.position, Rdisk_in, Rdisk_out)
    T = temperature (disk.position, -xi, star_mass)
    disk.temperature = T
    disk.surface_solid = sigma_d0(disk.surface_gas, fDG, FeH, disk.temperature)
    disk.scale_height = scale_height(sound_speed(T, mu), star_mass, disk.position)
    disk.alpha=alpha

    # disk.surface_solid = disk.surface_gas*fDG*10**FeH
    # dtgr = disk.surface_solid/disk.surface_gas
    # temp_d = np.array(cal_temperature(disk.position.value_in(units.cm),star_mass.value_in(units.g),star_radius.value_in(units.cm),star_teff.value_in(units.K),alpha, 
    # 	disk.surface_gas.value_in(units.g/units.cm**2), dtgr)) |units.K
    # disk.temperature = temp_d.reshape((pre_ndisk,1))

    # initialize venice
    system,_,_,_ = setup_single_pps(dt)
    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass
    system.codes[0].st  = stokes_number

    system.codes[1].disk = disk
    system.codes[1].disk_lifetime = lifetime
    # system.codes[1].sigma_dot_wind = -0 | units.MSun/units.yr/(100*units.au)**2

    system.codes[2].planets.add_particles(planets)
    system.codes[2].disk = disk
    system.codes[2].star.mass = star_mass
    system.codes[2].star.radius = star_radius
    system.codes[2].star_teff = star_teff

    N_plot_steps = int(end_time/dt_plot)
    t = np.zeros((N_plot_steps+1, len(planets))) | units.Myr
    a = np.zeros((N_plot_steps+1, len(planets))) | units.au
    Mc = np.zeros((N_plot_steps+1, len(planets))) | units.MEarth
    Me = np.zeros((N_plot_steps+1, len(planets))) | units.MEarth

    gas = []
    solid = []
    disk_time = []

    for i in range(N_plot_steps+1):

        system.evolve_model( (i) * dt_plot )

        t[i] = system.codes[0].model_time
        a[i] = system.codes[0].planets.semimajor_axis
        Mc[i] = system.codes[0].planets.core_mass
        Me[i] = system.codes[0].planets.envelope_mass
        
        # control the snapshots of disk
        if i%(int(N_plot_steps/N_snapshot)) == 0:
            disk_time.append(system.codes[0].model_time.value_in(units.kyr))

            print(int(system.codes[0].model_time.value_in(units.kyr)), end_time.value_in(units.kyr))
            solid.append(system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2))  
            gas.append(system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2))
        
    np.savez(filename+'_disk.npz', time=disk_time, position = system.codes[0].disk.position.value_in(units.AU), 
                gas=gas, solid=solid)

    np.savez(filename+'_planet.npz', time=t.value_in(units.kyr), Mc=Mc.value_in(units.MEarth), 
                Me=Me.value_in(units.MEarth), a = a.value_in(units.au))
    planets = system.codes[0].planets
    return [[planets[i].core_mass.value_in(units.MEarth), planets[i].envelope_mass.value_in(units.MEarth), planets[i].semimajor_axis.value_in(units.au)] for i in range(len(planets))]


if __name__ == '__main__':
    
    # initialize planets.
    Nplanet = 10
    smas = list(10**np.linspace(-1,1.5))
    a = [0.1, 0.5, 1, 5, 10, 20, 25] | units.AU
    M = [1e-2 for i in range(len(a))] | units.MEarth
    print(a)
    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0|units.g,
        semimajor_axis = a,
        isohist = [False for i in range(len(M))]
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 10 | units.kyr # timestep of the matrix
    end_time = 5000. | units.kyr
    dt_plot = end_time/1000
    N_plot_disk = 40
    
    M_star=1|units.MSun
    R_star=1|units.RSun
    T_star=5770|units.K

    fg = 1
    beta = 15/14
    xi = 3/7
    FeH = 0
    mu = 2.4
    alpha = 1e-3
    fDG = 0.01
    lifetime = 1.2 | units.Myr
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU
    stokes_number = 0.01
    
    filename = './'
    system = run_single_pps(fg, beta, xi, fDG, FeH, mu, alpha, Rdisk_in, Rdisk_out, lifetime, stokes_number, planets, M_star, R_star, T_star, dt, end_time, dt_plot, N_plot_disk, filename)
    
    sp.run(['python3', 'venice_pps_plot.py'])