import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.datamodel import Particles, Particle

from venice import Venice

class PebbleAccretion:

    def __init__ (self):

        # Model data
        self.planets = Particles()
        self.star = Particle(mass=0.08|units.MSun)

        self.model_time = 0. | units.Myr

        # Model parameters
        self.Mg_dot = 1e-10 | units.MSun/units.yr
        self.disk_mass = 0.1 * self.star.mass
        self.disk_radius = 100. | units.AU
        self.disk_lifetime = 3. | units.Myr
        self.metallicity = 0.02
        self.kappa = 10
        self.alpha = 1e-3
        self.St = 0.05

        # Integration parameters
        self.eta = 1e-2


    def evolve_model (self, end_time):

        for i in range(len(self.planets)):

            model_time_i = self.model_time

            while model_time_i < end_time:

                # Pebble accretion cutoff criteria
                if self.planets[i].mass > self.pebble_isolation_mass(
                        self.planets[i].semimajor_axis) or \
                        model_time_i > self.pebble_cutoff_time:
                    break

                q_pl = self.planets[i].mass / self.star.mass

                # Accretion of dry pebbles within the iceline
                if self.planets[i].semimajor_axis < self.a_iceline:

                    epsilon_3D = 0.07 * q_pl/1e-5

                    Mp_dot = epsilon_3D * self.fpg(model_time_i) * 0.5**(5./3.)

                # Accretion of icy pebbles outside the iceline
                else:

                    epsilon_2D = 184.6 * q_pl**(2./3.)

                    Mp_dot = epsilon_2D * self.fpg(model_time_i) * self.Mg_dot

                # This model uses forward Eulerian integration with adaptive
                # timesteps, with dt = x/(dx/dt)
                dt = min(abs(self.eta * self.planets[i].mass / Mp_dot),
                    end_time - model_time_i)

                self.planets[i].mass += Mp_dot * dt

                model_time_i += dt

        self.model_time = end_time


    # Quantities to be computed during evolution
    def fpg (self, time):

        return 2./3. * self.disk_mass * self.metallicity**(5./3.) / \
            (self.Mg_dot*self.disk_radius*self.kappa**(2./3.)) * \
            (constants.G*self.star.mass/max(time, 1.|units.kyr))**(1./3.)


    def pebble_isolation_mass (self, semimajor_axis):

        h = 0.05 * semimajor_axis.value_in(units.AU)**0.25

        res = h**3 * np.sqrt(37.3*self.alpha + 0.01)
        res *= 1. + 0.2 * (np.sqrt(self.alpha*(4. + 1./self.St**2))/h)**0.7
        res *= self.star.mass

        return res


    # Derived properties, these are recomputed each time in case any of these values
    # change. Not in this implementation, but maybe we want to add evolution!
    @property
    def a_iceline (self):
        return 0.077 * (self.star.mass/(0.08|units.MSun))**2 | units.AU


    @property
    def pebble_cutoff_time (self):
        return self.kappa/self.metallicity * \
            (self.disk_radius**3/(constants.G*self.star.mass))**0.5


class TypeIMigration:

    def __init__ (self):

        self.planets = Particles()
        self.star = Particle(mass=0.08|units.MSun)

        self.model_time = 0. | units.Myr

        self.cmig = 1.
        self.p = 1
        self.Mg_dot = 1e-10 | units.MSun/units.yr
        self.disk_mass = 0.1 * self.star.mass
        self.disk_radius = 100. | units.AU
        self.disk_lifetime = 3. | units.Myr

        self.eta = 1e-2


    def evolve_model (self, end_time):

        for i in range(len(self.planets)):

            q_pl = self.planets[i].mass / self.star.mass

            model_time_i = self.model_time

            while model_time_i < end_time:

                a = self.planets[i].semimajor_axis

                if a < self.disk_inner_edge or model_time_i > self.disk_lifetime:
                    break

                h = 0.05 * a.value_in(units.AU)**0.25
                vK = (constants.G*self.star.mass/a)**0.5

                a_dot = -2.8 * q_pl * self.cmig * self.sigma_g(a, model_time_i) * \
                    a**2 * vK / (self.star.mass * h**2)

                dt = min(abs(self.eta * self.planets[i].semimajor_axis / a_dot),
                    end_time - model_time_i)

                self.planets[i].semimajor_axis += a_dot * dt

                model_time_i += dt

        self.model_time = end_time


    def sigma_g (self, a, time):

        return (2. - self.p) * self.disk_mass / (2.*np.pi*self.disk_radius**2) * \
            (a/self.disk_radius)**-self.p * np.exp(-time/self.disk_lifetime)


    @property
    def disk_inner_edge (self):
        return 0.0102 * ((0.08|units.MSun)/self.star.mass)**(1./7.) * \
            ((1e-10 | units.MSun/units.yr)/self.Mg_dot)**(2./7.) | units.AU


def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    pebble_accretion = PebbleAccretion()
    typeI_migration = TypeIMigration()

    # Add codes
    system.add_code(pebble_accretion)
    system.add_code(typeI_migration)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix[0,1] = timestep

    # Add channels
    # PebbleAccretion informs TypeIMigration of updated mass
    system.add_channel(0, 1, from_attributes=['mass'], to_attributes=['mass'],
        from_set_name='planets', to_set_name='planets')
    # TypeIMigration informs PebbleAccretion of updated semimajor axis
    system.add_channel(1, 0, from_attributes=['semimajor_axis'], 
        to_attributes=['semimajor_axis'],
        from_set_name='planets', to_set_name='planets')

    return system, pebble_accretion, typeI_migration


def run_single_pps (planets, dt, end_time, dt_plot):

    system, _, _ = setup_single_pps(dt)

    system.codes[0].planets.add_particles(planets)
    system.codes[1].planets.add_particles(planets)


    N_plot_steps = int(end_time/dt_plot)
    a = np.zeros((N_plot_steps, len(planets))) | units.AU
    M = np.zeros((N_plot_steps, len(planets))) | units.MEarth

    for i in range(N_plot_steps):

        system.evolve_model( (i+1) * dt_plot )

        print (system.model_time.value_in(units.Myr), end_time.value_in(units.Myr))

        a[i] = system.codes[0].planets.semimajor_axis
        M[i] = system.codes[0].planets.mass


    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(planets)):
        ax.plot(a[:,i].value_in(units.AU), M[:,i].value_in(units.MEarth))

    ax.set_xlabel('a [au]')
    ax.set_ylabel('M [M$_\\oplus$]')

    ax.set_xscale('log')
    ax.set_yscale('log')


    return system


if __name__ == '__main__':

    M = [1e-4, 1e-4, 1e-4] | units.MEarth
    a = [1., 3., 10.] | units.AU

    planets = Particles(len(M),
        mass=M,
        semimajor_axis = a
    )

    dt = 1. | units.kyr
    end_time = 3000. | units.kyr
    dt_plot = 10. | units.kyr

    system = run_single_pps(planets, dt, end_time, dt_plot)

    plt.show()
