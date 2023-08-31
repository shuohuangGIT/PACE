# use vader (Krumholz+2015) and pedisk (Birnstiel+2012, Wilhelm+2023) to evolve gas and dust disk
import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid

from venice_src.venice import Venice
from amuse.community.vader.interface import Vader

from extra_funcs import *

class DiskGasDustEvolution:
    def __init__(self):
        self.code = Vader(mode='pedisk_dusty', redirection='none')
        self.code.module_time = 0|units.Myr
        self.model_time = 0 | units.Myr

        self.disk = new_regular_grid(([int(pre_ndisk)]),[1]|units.au)
        self.disk.surface_gas = 1e-20 | units.g/units.cm**2
        self.disk.surface_solid = 1e-20 | units.g/units.cm**2
        self.disk.vd = 0. | units.cm/units.s
        self.disk.st = 1e-3
        
        # need to initialize the grid when calling the function externally!
        # self.code.initialize_keplerian_grid(
        #     300,                # grid cells
        #     False,              # True for linear, False for logarithmic
        #     0.01 | units.au,    # inner disk edge
        #     500. | units.au,   # outer disk edge
        #     1. | units.MSun     # central mass
        # )

    def evolve_model(self, end_time):
        self.code.evolve_model(end_time)
        self.disk.surface_gas = self.code.grid.column_density
        self.disk.surface_solid = self.code.grid_user[0].value | units.g/units.cm**2
        self.disk.vd = self.code.grid_user[3].value | units.cm/units.s
        self.disk.st = self.code.grid_user[4].value
        self.model_time = self.code.model_time

        print(self.code.grid_user[2].value)

# example of calling the function---------------------------------------------

def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    disk_gas_dust_evolution = DiskGasDustEvolution()

    # Add codes
    system.add_code(disk_gas_dust_evolution)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix = timestep
    return system, disk_gas_dust_evolution # core_accretion, typeI_migration

def init_viscous(viscous, alpha, alpha_acc, mu, M_dot_ph_in, M_dot_ph_ex, star_mass, v_frag):
    viscous.parameters.alpha = alpha_acc # should change to alpha_acc if disk accretion is mainly wind-driven.
    # some fixted parameters
    viscous.parameters.inner_boundary_function = True    
    viscous.parameters.inner_pressure_boundary_type = 1     # fixed mass flux
    viscous.parameters.maximum_tolerated_change = 1e99
    viscous.parameters.post_timestep_function = True
    viscous.parameters.number_of_user_outputs = 5
    viscous.parameters.number_of_user_parameters = 14
    # some free (user defined) parameters
    viscous.set_parameter(0, ( M_dot_ph_in ).value_in(units.g/units.s))
    viscous.set_parameter(1, ( M_dot_ph_ex ).value_in(units.g/units.s))
    # minimum gas surface density to photoevaporate to, in g/cm^2
    viscous.set_parameter(2, 1e-12)
    viscous.set_parameter(3, 1e9)
    viscous.set_parameter(4, (mu*1.008*constants.u).value_in(units.g))
    viscous.set_parameter(5, alpha)
    viscous.set_parameter(6, star_mass.value_in(units.MSun))
    viscous.set_parameter(7, v_frag) # dust fragmentation velocity [cm/s]
    viscous.set_parameter(8, 1.) # dust internal density
    viscous.set_parameter(9, 1e-3)
    viscous.set_parameter(10, 1e99)
    return viscous

def run_single_pps (R_in, R_out, star_mass, alpha, alpha_acc, M_dot_ph_in, M_dot_ph_ex, fDG, dt, end_time, dt_plot):
    system, _ = setup_single_pps(dt)
    viscous = system.codes[0].code

    viscous.initialize_keplerian_grid(
        500,                # grid cells
        False,              # True for linear, False for logarithmic
        R_in,    # inner disk edge
        R_out,   # outer disk edge
        star_mass     # central mass
    )

    mu = 2.3 # mean molecular weight
    v_frag = 1e3 # cm/s

    viscous = init_viscous(viscous, alpha, alpha_acc, mu, M_dot_ph_in, M_dot_ph_ex, star_mass, v_frag)

    disk_mass   = 0.1 * star_mass
    disk_radius = R_out
    temp = (150. | units.K) * viscous.grid.r.value_in(units.au)**-(3/7)

    sigma0 = disk_mass / (2.*np.pi * disk_radius**2. * (1. - np.exp(-1.)))
    sigma = sigma0 * disk_radius/viscous.grid.r * np.exp(-viscous.grid.r/disk_radius)
    sigma[ viscous.grid.r > disk_radius*2/3 ] = 1e-12 | units.g/units.cm**2 # sharp edge
    
    pressure = constants.kB*temp*sigma / (mu*1.008*constants.u)
    
    viscous.grid.column_density = sigma
    viscous.grid.pressure = pressure

    # dust density
    viscous.grid_user[0].value = fDG * sigma.value_in(units.g/units.cm**2)
    # dust initial grain size
    viscous.grid_user[1].value = 1e-3
    # temperature profile
    viscous.grid_user[2].value = temp.value_in(units.K)

    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)
    disk.position = system.codes[0].code.grid.r

    disk.surface_gas = sigma
    disk.surface_solid = fDG * sigma
    system.codes[0].disk = disk

    N_plot_steps = int(end_time/dt_plot)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    color = (1,0,0)
    ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2),color = color, label = '%.1f Myr'%system.codes[0].model_time.value_in(units.Myr))
    ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2),color = color, linestyle='--')
    for i in range(N_plot_steps):

        system.codes[0].evolve_model( (i+1) * dt_plot )

        print ("Time(/Myr)", system.codes[0].model_time.value_in(units.Myr), "End Time(/Myr)", end_time.value_in(units.Myr))
        
        if i==N_plot_steps-1: # comment it out if plot more snapshots
            color = (1-(i+1)/N_plot_steps,0,(i+1)/N_plot_steps)
            # ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2),color = color, label = '%.1f Myr'%system.codes[0].model_time.value_in(units.Myr))
            # ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2),color = color, linestyle='--')

            ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.vd.value_in(units.cm/units.s),color = color, label = '%.1f Myr'%system.codes[0].model_time.value_in(units.Myr))
            ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.st,color = color, linestyle='--')


    ax.set_xlabel('$a [au]$')
    ax.set_ylabel(r'$\Sigma_g$[$g/cm^{2}$]')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2,1e2)
    plt.legend(loc='upper right')

    return system


if __name__ == '__main__':
    end_time = 1. | units.Myr
    dt_plot = end_time/10
    dt = dt_plot

    star_mass = 1. | units.MSun
    Rdisk_in = 0.01 | units.AU
    Rdisk_out = 300. | units.AU

    fDG = 0.0134
    alpha = 1e-4
    alpha_acc = 1e-4
    M_dot_ph_in = 0|units.MSun/units.yr
    M_dot_ph_ex = 0|units.MSun/units.yr
    
    system = run_single_pps(Rdisk_in, Rdisk_out, star_mass, alpha, alpha_acc, M_dot_ph_in, M_dot_ph_ex, fDG, dt, end_time, dt_plot)

    plt.show()
