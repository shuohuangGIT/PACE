import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid

from venice_src.venice import Venice

from extra_funcs import *

class DiskGasEvolution:
    def __init__(self):
        self.planets = Particles()
        self.planets.add_calculated_attribute('dynamical_mass', dynamical_mass)
        # self.star = Particle(mass=1|units.MSun)
        self.model_time = 0. | units.Myr
        self.disk = new_regular_grid(([int(pre_ndisk)]),[1]|units.au)
        self.disk.surface_gas = 0 | units.g/units.cm**2
        self.disk_lifetime = 1 | units.Myr
        self.sigma_dot_wind = -1e-7 | units.MSun/units.yr/(100*units.au)**2
        self.sigmag_min = 1e-20|units.g/units.cm**2

    def calculate_dt(self, model_time_i, end_time):
        return min(1 | units.kyr, end_time-model_time_i)
    
    def calculate_sigma_dot(self):
        return -self.disk.surface_gas/self.disk_lifetime+self.sigma_dot_wind

    def evolve_model(self, end_time):
        model_time_i = self.model_time
        while model_time_i< end_time:
            dt = self.calculate_dt(model_time_i, end_time)
            
            # first order
            self.disk.surface_gas += self.calculate_sigma_dot()*dt
            self.disk.surface_gas = np.maximum(self.disk.surface_gas,self.sigmag_min) # positive value
            model_time_i += dt
        self.model_time = end_time

#---------------------------------------------

def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    # core_accretion = CoreAccretion()
    disk_gas_evolution = DiskGasEvolution()

    # Add codes
    # system.add_code(core_accretion)
    system.add_code(disk_gas_evolution)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix = timestep

    return system, disk_gas_evolution # core_accretion, typeI_migration


def run_single_pps (disk, dt, end_time, dt_plot):

    system, _ = setup_single_pps(dt)
    system.codes[0].disk = disk

    N_plot_steps = int(end_time/dt_plot)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(N_plot_steps):

        system.codes[0].evolve_model( (i+1) * dt_plot )

        print (system.codes[0].model_time.value_in(units.Myr), end_time.value_in(units.Myr))
        
        color = (1-(i+1)/N_plot_steps,0,(i+1)/N_plot_steps)
        ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2),color = color, label = '%.1f Myr'%system.codes[0].model_time.value_in(units.Myr))

    ax.set_xlabel('$a [au]$')
    ax.set_ylabel(r'$\Sigma_g$[$g/cm^{2}$]')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2,1e3)
    plt.legend(loc='upper right')

    return system


if __name__ == '__main__':
    dt = 10. | units.kyr
    end_time = 10000. | units.kyr
    dt_plot = 1000. | units.kyr
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)

    sigmag_0 = 2400 | units.g/units.cm**2
    fg = 5
    pg0 = -1
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU

    disk.position = Rdisk0(1e-2|units.au, 1e2|units.au, pre_ndisk)
    disk.surface_gas = sigma_g0(sigmag_0, fg, pg0, disk.position, Rdisk_in, Rdisk_out)    

    system = run_single_pps(disk, dt, end_time, dt_plot)

    plt.show()
