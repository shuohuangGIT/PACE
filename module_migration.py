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
        self.disk.alpha = 1e-4

        self.dt = pre_dt
        self.TypeI = False
        self.eta = 0.1

    def set_time_step(self, tau_I, model_time_i, end_time):
        if np.isinf(tau_I.value_in(units.kyr)):
            dt = end_time-model_time_i
        else:
            dt_hill = np.log(1+min(Rhills(self.planets.dynamical_mass,self.star.mass,self.planets.semimajor_axis)/self.planets.semimajor_axis))*tau_I
            dt_min = self.eta* tau_I
            dt = min(abs(dt_hill), abs(end_time-model_time_i), abs(dt_min))
        return dt
    
    def evolve_model (self, end_time):
        model_time_i = self.model_time
        Rdisk = self.disk.position
        Sigmag = self.disk.surface_gas
        Hdisk = self.disk.scale_height
        if (min(self.planets.semimajor_axis)>Rdisk[0]) and (max(self.planets.semimajor_axis)<Rdisk[-1]):
            while model_time_i < end_time:
                tau_a = np.zeros(len(self.planets)) | units.kyr
                for i in range(len(self.planets)):
                    ap = self.planets[i].semimajor_axis
                    
                    if (ap<=Rdisk[0]) or (ap>=Rdisk[-1]):
                        tau_a[i] = np.inf | units.kyr
                        print("WARNING: Planet(s) is out of the grids")
                    else:
                        ipi = np.nonzero(Rdisk<=ap)[0][-1]
                        alpha = self.disk.alpha[ipi]

                        # calculate slope
                        xi = -np.log(Hdisk[int(ipi)+1]/Hdisk[int(ipi)])/np.log(Rdisk[int(ipi)+1]/Rdisk[int(ipi)])*2+3
                        beta = - np.log(Sigmag[int(ipi)+1]/Sigmag[int(ipi)])/np.log(Rdisk[int(ipi)+1]/Rdisk[int(ipi)])

                        sigmag_p = (Sigmag[ipi]-Sigmag[ipi+1])*(Rdisk[ipi]-ap)/(Rdisk[ipi]-Rdisk[ipi+1])+Sigmag[ipi]
                        # sigmag_p = Sigmag[ipi]

                        q_pl = self.planets[i].dynamical_mass / self.star.mass
                        q_g = self.star.mass/ap**2/sigmag_p

                        # H_p = Hdisk[ipi]
                        H_p = (Hdisk[ipi]-Hdisk[ipi+1])*(Rdisk[ipi]-ap)/(Rdisk[ipi]-Rdisk[ipi+1])+Hdisk[ipi]
                        h_p = H_p/ap
                        # print(H_p.value_in(units.au), Rhills(self.planets[i].dynamical_mass,self.star.mass,ap).value_in(units.au))
                        if self.TypeI==True:
                            K=0
                        else:
                            K = (q_pl)**2*(h_p)**-5/alpha
                        tau_a[i] = 1/(1.36+0.62*beta+0.43*xi)*(h_p)**2/q_pl*q_g*np.sqrt(ap**3/constants.G/self.star.mass)/2*(1+0.04*K)

                dt = self.set_time_step(min(tau_a), model_time_i, end_time)
                model_time_i += dt

                for i in range(len(self.planets)):
                    ap = self.planets[i].semimajor_axis
                    a_dot = -ap/tau_a[i]
                    self.planets[i].semimajor_axis += a_dot * dt

                if dt == 0|units.s:
                    break
        else:
            pass
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

    M = [1e0] | units.MEarth
    a = [25.] | units.AU

    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0 |units.g,
        semimajor_axis = a
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 4 | units.kyr
    end_time = 2000 | units.kyr
    dt_plot = end_time/1000
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)

    sigmag_0 = 2400 | units.g/units.cm**2
    fg = 5
    beta = 15/14
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU
    fDG, FeH, xi, star_mass = 0.01, 0, 3/7, 1|units.MSun
    mu = 2.4
    alpha = 1e-4

    disk.position = Rdisk0(1e-3|units.au, 1e2|units.au, pre_ndisk)
    disk.surface_gas = fg*sigmag_0 * (disk.position.value_in(units.au))**(-beta)

    T = temperature (disk.position, xi, star_mass)
    disk.surface_solid = disk.surface_gas *fDG
    disk.scale_height = scale_height(sound_speed(T, mu), star_mass, disk.position)
    disk.alpha = 1e-4

    system = run_single_pps(disk, planets, star_mass, dt, end_time, dt_plot)

    plt.show()
