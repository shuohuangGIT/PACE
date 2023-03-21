import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid
from venice_src.venice import Venice

from extra_funcs import *
from amuse.community.secularmultiple.interface import SecularMultiple

class SecularEvolution_migration:

    def __init__ (self):
        # need to initialize disk, planet, star, disk lifetime
        self.planets = Particles()
        self.planets.add_calculated_attribute('dynamical_mass', dynamical_mass)
        self.star = Particle(mass=1|units.MSun)

        self.model_time = 0 | units.Myr
        self.dt = dt

        self.code = SecularMultiple()
        self.code.model_time = 0 | units.Myr

    def cal_tauA(self):
        tau_A = 10 | units.kyr
        # tau_A = np.inf | units.kyr
        return tau_A
    
    def cal_tauE(self):
        tau_E = 10 | units.kyr
        # tau_E = np.inf | units.kyr
        return tau_E
    
    def cal_tauI(self):
        return np.inf | units.kyr
    
    def cal_tauAOP(self):
        return np.inf | units.kyr
    
    def cal_tauLOAN(self):
        return np.inf | units.kyr
    
    def evolve_model (self, end_time):
        model_time_i = self.model_time
        Nbodies = len(self.planets)+1
        
        while model_time_i < end_time:
            dt = min(self.dt, abs(end_time - model_time_i))
            model_time_i += dt
                
            self.code.evolve_model(model_time_i)
        
            # condidering damping of orbital elements, linear superposition
            A_dot = -self.planets.semimajor_axis/self.cal_tauA()
            E_dot = -self.planets.eccentricity/self.cal_tauE()
            I_dot = -self.planets.inclination/self.cal_tauI()
            AOP_dot = -self.planets.argument_of_pericenter/self.cal_tauAOP()
            LOAN_dot = -self.planets.longitude_of_ascending_node/self.cal_tauLOAN()

            self.planets.semimajor_axis = self.code.particles[Nbodies:].semimajor_axis + A_dot*dt
            self.planets.eccentricity = self.code.particles[Nbodies:].eccentricity + E_dot*dt
            self.planets.inclination = self.code.particles[Nbodies:].inclination + I_dot*dt
            self.planets.argument_of_pericenter = self.code.particles[Nbodies:].argument_of_pericenter + AOP_dot*dt
            self.planets.longitude_of_ascending_node = self.code.particles[Nbodies:].longitude_of_ascending_node + LOAN_dot*dt
            
            for planet_i in range(Nbodies-1):
                self.code.set_orbital_elements(Nbodies+planet_i, self.planets[planet_i].semimajor_axis, self.planets[planet_i].eccentricity,
                                           self.planets[planet_i].inclination, self.planets[planet_i].argument_of_pericenter, 
                                           self.planets[planet_i].longitude_of_ascending_node)
            # xxx
        self.model_time = end_time
#---------------------------------------------

def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    # core_accretion = CoreAccretion()
    secularEvolution = SecularEvolution_migration()

    # Add codes
    # system.add_code(core_accretion)
    system.add_code(secularEvolution)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix = timestep

    return system, secularEvolution # core_accretion, secularEvolution


def run_single_pps (disk, planets, star_mass, dt, end_time, dt_plot):

    system, _ = setup_single_pps(dt)

    system.codes[0].planets.add_particles(planets)
    system.codes[0].star.mass = star_mass

    particles, binaries = initialize_planetary_system(len(planets)+1, star_mass, planets.mass, planets.semimajor_axis, planets.eccentricity, planets.inclination, planets.argument_of_pericenter, planets.longitude_of_ascending_node)
    # print(particles[particles.is_binary])

    N_plot_steps = int(end_time/dt_plot)
    ### set up some arrays for plotting ###
    N_binaries = len(planets)
    print_smas_AU = [[] for x in range(N_binaries)]
    print_rps_AU = [[] for x in range(N_binaries)]
    print_parent_is_deg = [[] for x in range(N_binaries)]
    print_times_Myr = []
    
    system.codes[0].code.particles.add_particles(particles)
    channel_from_particles_to_code = particles.new_channel_to(system.codes[0].code.particles)
    channel_from_code_to_particles = system.codes[0].code.particles.new_channel_to(particles)
    channel_from_particles_to_code.copy()

    time = 0.0|units.yr
    while time<=end_time:
        time += dt_plot

        system.codes[0].evolve_model(time)
        channel_from_code_to_particles.copy()

        print('='*50)
        print(binaries.mass)
        print('t/Myr',time.value_in(units.Myr))
        print('e',binaries.eccentricity)
        print('a',system.codes[0].planets.semimajor_axis)
        print('i/deg', np.rad2deg(binaries.inclination))
        print('AP/deg', \
            np.rad2deg(binaries.argument_of_pericenter))  
        print('LAN/deg', \
            np.rad2deg(binaries.longitude_of_ascending_node))
        
        ### write to output arrays ###
        print_times_Myr.append(time.value_in(units.Myr))
        for index_binary in range(N_binaries):
            print_smas_AU[index_binary].append( binaries[index_binary].semimajor_axis.value_in(units.AU) )
            print_rps_AU[index_binary].append( binaries[index_binary].semimajor_axis.value_in(units.AU)*(1.0 - binaries[index_binary].eccentricity) )
            print_parent_is_deg[index_binary].append( np.rad2deg(binaries[index_binary].inclination_relative_to_parent) )

    a = [1., 4., 8.] | units.AU
    inclination = [0., 0.1, 0.1]
    longitude_of_ascending_node = [0.001, 0.001, 0.001]

    ### compute the `canonical' maximum eccentricity/periapsis distance that applies in the quadrupole-order test-particle limit if the `outer' binary is replaced by a point mass ###
    print(inclination[0],inclination[2],longitude_of_ascending_node[0],longitude_of_ascending_node[2])
    i_AC_init = compute_mutual_inclination(inclination[0],inclination[2],longitude_of_ascending_node[0],longitude_of_ascending_node[2])
    i_BC_init = compute_mutual_inclination(inclination[1],inclination[2],longitude_of_ascending_node[1],longitude_of_ascending_node[2])
    
    canonical_rp_min_A_AU = (a[0]*(1.0 - np.sqrt( 1.0 - (5.0/3.0)*np.cos(i_AC_init)**2 ) )).value_in(units.AU)
    canonical_rp_min_B_AU = (a[1]*(1.0 - np.sqrt( 1.0 - (5.0/3.0)*np.cos(i_BC_init)**2 ) )).value_in(units.AU)

    data = print_times_Myr,print_smas_AU,print_rps_AU,print_parent_is_deg,canonical_rp_min_A_AU,canonical_rp_min_B_AU

    plot_function(data)
    return system


if __name__ == '__main__':

    M = [1,1,1] | units.MEarth
    M[1] = 1|units.MJupiter
    M[2] = 0.1|units.MJupiter
    a = [1., 4., 8.] | units.AU
    a = a
    eccentricity = [0.1, 0.1, 0.3]
    inclination = np.deg2rad([0, 0.1, 0.1])
    argument_of_pericenter = np.deg2rad([10, 30, 60])
    longitude_of_ascending_node = np.deg2rad([0.001, 0.001, 0.001])

    planets = Particles(len(M),
        mass=M,
        eccentricity = eccentricity,
        envelope_mass = np.zeros(len(M)) | units.g,
        semimajor_axis = a, 
        inclination = inclination,
        argument_of_pericenter = argument_of_pericenter,
        longitude_of_ascending_node = longitude_of_ascending_node
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 0.1 | units.kyr
    end_time = 0.01 | units.Myr
    dt_plot = end_time/400.
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)

    sigmag_0 = 2400 | units.g/units.cm**2
    fg = 5
    pg0 = -1
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU
    fDG, FeH, pT, star_mass = 0.0149, 0, -0.5, 1|units.MSun
    mu = 2.4

    disk.position = Rdisk0(Rdisk_in, Rdisk_out, pre_ndisk)
    disk.surface_gas = sigma_g0(sigmag_0, fg, pg0, disk.position, Rdisk_in, Rdisk_out)
    
    T = temperature (disk.position, pT, star_mass)
    disk.surface_solid = sigma_d0(disk.surface_gas, fDG, FeH, T)
    disk.scale_height = scale_height(sound_speed(T, mu), star_mass, disk.position)

    # plt.plot(disk.position.value_in(units.au), disk.surface_solid.value_in(units.g/units.cm**2))
    # plt.xscale('log')
    # plt.yscale('log')
    system = run_single_pps(disk, planets, star_mass, dt, end_time, dt_plot)
