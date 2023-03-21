import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid
from venice_src.venice import Venice

from extra_funcs import *

class TypeIMigration:

    def __init__ (self):
        # need to initialize disk, planet, star, disk lifetime
        self.planets = Particles()
        self.planets.add_calculated_attribute('dynamical_mass', dynamical_mass)
        self.star = Particle(mass=1|units.MSun)
        self.model_time = 0. | units.Myr

        self.disk = new_regular_grid(([int(pre_ndisk)]),[1])
        self.disk.surface_gas = 0 | units.g/units.cm**2
        self.disk.surface_solid = 0 | units.g/units.cm**2
        self.disk.scale_height = 0.03 * self.disk.position
        self.disk_lifetime = 1 | units.Myr
        self.alpha = 1e-3

        self.dt = pre_dt

        self.eta = 0.1

        self.C1 = 4e-3
        self.C2 = 2e-3

    def cal_Rm(self, time):
        return (10|units.au) * np.exp(2*time/5/self.disk_lifetime)
    
    def set_time_step(self, eta, tau_I, model_time_i, end_time):
        dt_hill = np.log(1+min(Rhills(self.planets.dynamical_mass,self.star.mass,self.planets.semimajor_axis)/self.planets.semimajor_axis))*tau_I
        return min(abs(dt_hill), abs(end_time-model_time_i))

    def evolve_model (self, end_time):
        model_time_i = self.model_time
        Rdisk = self.disk.position
        Sigmag = self.disk.surface_gas
        Hdisk = self.disk.scale_height

        while model_time_i < end_time:
            tau_a = np.zeros(len(self.planets)) | units.kyr
            for i in range(len(self.planets)):
                ap = self.planets[i].semimajor_axis
                
                ipi = 0
                for j in range(len(Rdisk)-1):                
                    if (Rdisk[j]<=ap) & (Rdisk[j+1]>ap):
                        ipi=j
                
                pg = (Sigmag[ipi+1]-Sigmag[ipi])/(Rdisk[ipi+1]-Rdisk[ipi])*(Rdisk[ipi+1]+Rdisk[ipi])/(Sigmag[ipi+1]+Sigmag[ipi])

                sigmag_p = (Sigmag[ipi]-Sigmag[ipi+1])*(Rdisk[ipi]-ap)/(Rdisk[ipi]-Rdisk[ipi+1])+Sigmag[ipi]
                
                q_pl = self.planets[i].dynamical_mass / self.star.mass
                q_g = self.star.mass/ap**2/sigmag_p

                H_p = (Hdisk[ipi]-Hdisk[ipi+1])*(Rdisk[ipi]-ap)/(Rdisk[ipi]-Rdisk[ipi+1])+Hdisk[ipi]
                h_p = H_p/ap
                # print(H_p.value_in(units.au), Rhills(self.planets[i].dynamical_mass,self.star.mass,ap).value_in(units.au))

                if H_p>Rhills(self.planets[i].dynamical_mass,self.star.mass,ap):
                    tau_a[i] = 1/(2.728-1.082*pg)*(h_p)**2/q_pl*q_g*np.sqrt(ap**3/constants.G/self.star.mass)/2 / self.C1
                else:
                    #Type II
                    if ap<Rdisk[0]:
                        tau_a[i] = np.inf | units.kyr
                    else:
                        Rm = self.cal_Rm(model_time_i)
                        for j in range(len(Rdisk)-1):                
                            if (Rdisk[j]<=Rm) & (Rdisk[j+1]>Rm):
                                im=j
                        omega_p_m = (ap/Rm)**(3/2)
                        adoti = ap * 3*np.sign((ap-Rm).value_in(units.au))*self.alpha*(Sigmag[im]*Rm**2/self.planets[i].dynamical_mass)*(Hdisk[im]/ap)**2 * omega_p_m * np.sqrt(constants.G*self.star.mass/Rm**3)
                        tau_a[i] = -ap/adoti / self.C2

            dt = self.set_time_step(self.eta, min(tau_a), model_time_i, end_time)
            model_time_i += dt

            for i in range(len(self.planets)):
                ap = self.planets[i].semimajor_axis
                a_dot = -ap/tau_a[i]
                self.planets[i].semimajor_axis += a_dot * dt

            if dt == 0|units.s:
                break
        
        self.model_time = end_time


#---------------------------------------------

def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    typeI_migration = TypeIMigration()

    # Add codes
    system.add_code(typeI_migration)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix = timestep

    return system, typeI_migration # core_accretion, typeI_migration


def run_single_pps (disk, planets, star_mass, dt, end_time, dt_plot):

    system, _ = setup_single_pps(dt)

    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass

    N_plot_steps = int(end_time/dt_plot)
    t = np.zeros((N_plot_steps, len(planets))) | units.Myr
    a = np.zeros((N_plot_steps, len(planets))) | units.au
    M = np.zeros((N_plot_steps, len(planets))) | units.MEarth

    fig = plt.figure(0,figsize=(10,8))

    for i in range(N_plot_steps):

        system.codes[0].evolve_model( (i+1) * dt_plot )

        print ("Time(/Myr)", system.codes[0].model_time.value_in(units.Myr), "End Time(/Myr)", end_time.value_in(units.Myr))

        t[i] = system.codes[0].model_time
        a[i] = system.codes[0].planets.semimajor_axis
        M[i] = system.codes[0].planets.dynamical_mass
    
    ax = plt.subplot(2,1,1)
    for i in range(len(planets)):
        ax.plot(a[:,i].value_in(units.au), t[:,i].value_in(units.kyr))

    ax.set_ylabel('$time [kyr]$')
    ax.set_xlabel('$a [au]$')
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlim(1e-2, 1e3)

    ax = plt.subplot(2,1,2)
    ax.plot(system.codes[0].disk.position.value_in(units.AU), 
                    system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2))

    ax.set_xlabel('$a [au]$')
    ax.set_ylabel(r'$\Sigma_g$[$g/cm^{2}$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 1e3)
    return system


if __name__ == '__main__':

    M = [1e-3] | units.MEarth
    a = [20.] | units.AU

    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0 |units.g,
        semimajor_axis = a
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 100 | units.kyr
    end_time = 4000. | units.kyr
    dt_plot = end_time/100
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)

    sigmag_0 = 2400 | units.g/units.cm**2
    fg = 5
    pg0 = -1
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU
    fDG, FeH, pT, star_mass = 0.0149, 0, -0.5, 1|units.MSun
    mu = 2.4

    disk.position = Rdisk0(1e-3|units.au, 1e2|units.au, pre_ndisk)
    disk.surface_gas = sigma_g0(sigmag_0, fg, pg0, disk.position, Rdisk_in, Rdisk_out)

    T = temperature (disk.position, pT, star_mass)
    disk.surface_solid = sigma_d0(disk.surface_gas, fDG, FeH, T)
    disk.scale_height = scale_height(sound_speed(T, mu), star_mass, disk.position)

    system = run_single_pps(disk, planets, star_mass, dt, end_time, dt_plot)

    plt.show()
