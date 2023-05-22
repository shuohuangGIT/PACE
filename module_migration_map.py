import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid
from venice_src.venice import Venice

from extra_funcs import *
from migration_map_paadekooper import cal_tau_I, cal_temperature

class TypeIMigration:

    def __init__ (self):
        # need to initialize disk, planet, star, disk lifetime
        self.planets = Particles()
        self.planets.add_calculated_attribute('dynamical_mass', dynamical_mass)
        self.star = Particle(mass=1|units.MSun)
        self.star_teff = 5775 | units.K
        self.model_time = 0. | units.Myr

        self.disk = new_regular_grid(([int(pre_ndisk)]),[1])
        self.disk.surface_gas = 100 | units.g/units.cm**2
        self.disk.surface_solid = 10 | units.g/units.cm**2
        self.disk.temperature = 10 | units.K
        self.disk.scale_height = 0.03 * self.disk.position
        self.disk_lifetime = 1 | units.Myr
        self.alpha = 2e-3
        self.gamma = 7/5

        self.dt = pre_dt

        self.eta = 0.1
        self.C2 = 2e-3

    def cal_Rm(self, time):
        return (10|units.au) * np.exp(2*time/5/self.disk_lifetime)
    
    def set_time_step(self, tau_I, model_time_i, end_time):
        dt_hill = np.log(1+min(Rhills(self.planets.dynamical_mass,self.star.mass,self.planets.semimajor_axis)/self.planets.semimajor_axis))*tau_I
        dt_min = self.eta* tau_I
        dt = min(abs(dt_hill), abs(end_time-model_time_i), abs(dt_min))
        return dt

    def access_migration_map(self, rp, Mp):
        Ms, gamma, sigmag, sigmad, tempd, rgrid = self.star.mass, self.gamma, self.disk.surface_gas, self.disk.surface_solid, self.disk.temperature, self.disk.position
        M_planet, M_star, sigma_g, sigma_d, r_grid = Mp.value_in(units.g), Ms.value_in(units.g), sigmag.value_in(units.g/units.cm**2), sigmad.value_in(units.g/units.cm**2), rgrid.value_in(units.cm)
        rpj = rp.value_in(units.cm)
        temp_d = tempd.value_in(units.K)
        alpha = self.alpha
        Z, Mig_ratej = cal_tau_I(np.array([rpj]), M_planet, M_star, gamma, sigma_g, sigma_d, temp_d, r_grid, alpha)

        return Z, Mig_ratej|units.yr**-1

    def evolve_model (self, end_time):
        model_time_i = self.model_time
        Rdisk = self.disk.position
        dtgr = self.disk.surface_solid/self.disk.surface_gas
        temp_d = np.array(cal_temperature(self.disk.position.value_in(units.cm),self.star.mass.value_in(units.g),self.star.radius.value_in(units.cm),
                                          self.star_teff.value_in(units.K), self.alpha, self.disk.surface_gas.value_in(units.g/units.cm**2), dtgr)) |units.K
        self.disk.temperature = temp_d.reshape((pre_ndisk,1))
        while model_time_i < end_time:
            tau_a = np.zeros(len(self.planets)) | units.kyr
            for i in range(len(self.planets)):
                ap = self.planets[i].semimajor_axis
                #Type I & Type II
                if (ap<=Rdisk[0]) or (ap>=Rdisk[-1]):
                    tau_a[i] = np.inf | units.kyr
                    print("WARNING: Planet(s) is out of the grids")
                else:
                    _, rate = self.access_migration_map(ap, self.planets[i].dynamical_mass)
                    tau_a[i] = -rate**-1

            dt = self.set_time_step(min(abs(tau_a)), model_time_i, end_time)
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


def run_single_pps (ax1, disk, planets, star_mass, star_radius, dt, end_time, dt_plot):

    system, _ = setup_single_pps(dt)

    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass
    system.codes[0].star.radius = star_radius
    system.codes[0].C1 = 1
    system.codes[0].C2 = 1

    N_plot_steps = int(end_time/dt_plot)+1
    t = np.zeros((N_plot_steps, len(planets))) | units.Myr
    a = np.zeros((N_plot_steps, len(planets))) | units.au
    M = np.zeros((N_plot_steps, len(planets))) | units.MEarth

    for i in range(N_plot_steps):
        system.codes[0].evolve_model( (i) * dt_plot )
        print ("Time(/Myr)", system.codes[0].model_time.value_in(units.Myr), "End Time(/Myr)", end_time.value_in(units.Myr), 
               "Planet sma/au", system.codes[0].planets.semimajor_axis.value_in(units.AU))

        t[i] = system.codes[0].model_time
        a[i] = system.codes[0].planets.semimajor_axis
        M[i] = system.codes[0].planets.dynamical_mass

    for i in range(len(planets)):
        ax1.plot(a[:,i].value_in(units.au), M[:,i].value_in(units.MEarth), 'ko-', linewidth=2, alpha=0.7)

    # ax1.set_ylabel('$time [kyr]$')
    # ax1.set_xlabel('$a [au]$')
    # ax1.set_xscale('log')
    # # ax1.set_yscale('log')
    # ax1.set_xlim(1e-2, 1e2)

    ax2 = plt.subplot(2,1,2)
    ax2.plot(system.codes[0].disk.position.value_in(units.AU), 
                    system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2))

    ax2.set_xlabel('$a [au]$')
    ax2.set_ylabel(r'$\Sigma_g$[$g/cm^{2}$]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1,1e4)
    ax2.set_xlim(1e-2, 1e2)
    return system


if __name__ == '__main__':

    M = [1, 2, 4, 10] | units.MEarth
    a = [10., 10., 10, 10] | units.AU

    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0 |units.g,
        semimajor_axis = a
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 4 | units.kyr
    end_time = 10000 | units.kyr
    dt_plot = end_time/100
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)

    #star
    M_star=1|units.MSun
    R_star=1|units.RSun
    Teff = 5770|units.K

    #disk
    sigmag_0 = 40 | units.g/units.cm**2
    fg = 5
    pg0 = -1
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU
    fDG, FeH, pT = 0.0149*0.1, 0, -0.5
    mu = 2.4
    alpha = 2e-3
    gamma = 7/5

    disk.position = Rdisk0(Rdisk_in, 10|units.au, pre_ndisk)
    disk.surface_gas = sigma_g0(sigmag_0, fg, pg0, disk.position, Rdisk_in, Rdisk_out)
    disk.surface_solid = disk.surface_gas*0.0196

    dtgr = disk.surface_solid/disk.surface_gas
    temp_d = np.array(cal_temperature(disk.position.value_in(units.cm),M_star.value_in(units.g),R_star.value_in(units.cm),Teff.value_in(units.K),alpha, disk.surface_gas.value_in(units.g/units.cm**2), dtgr)) |units.K
    disk.temperature = temp_d.reshape((pre_ndisk,1))

    disk.scale_height = scale_height(sound_speed(disk.temperature, mu), M_star, disk.position)

    from test_migration_map import access_migration_map
    rp = (disk.position[:-1]+disk.position[1:])/2

    mp = 10**np.linspace(-1,2,200) | units.MEarth
    X,Y = np.meshgrid(rp.value_in(units.au),mp.value_in(units.MEarth))
    Z=[]
    
    Mig_rate = []
    from tqdm import *

    for i, M_planet in enumerate(tqdm(mp)):
        Zi = []
        Mig_ratei = []
        for j, rpj in enumerate(rp):
            # print(rpj, M_planet, M_star, gamma, disk.surface_gas, disk.surface_solid, temp_d, disk.position, alpha)
            Zj, Mig_ratej = access_migration_map(rpj[0], M_planet, M_star, gamma, disk.surface_gas, disk.surface_solid, temp_d, disk.position, alpha)

            Zi.append(Zj[0])
            Mig_ratei.append(Mig_ratej[0].value_in(units.yr**-1))
        Z.append(Zi)
        Mig_rate.append(Mig_ratei)

    Z=np.array(Z)
    Mig_rate=np.array(Mig_rate)

    fig = plt.figure(0,figsize=(10,8))
    ax = plt.subplot(2,1,1)

    import matplotlib.colors as colors
    levels = np.linspace(-1e-5,1e-5,200)
    # cnt = ax.contourf(X, Y, Z, levels=levels,extend='both', cmap='RdBu_r')
    lnrwidth = 1e-8
    shadeopts = {'cmap': 'RdBu_r', 'shading': 'gouraud'}
    colormap = 'RdBu_r'
    gain = 1e-5
    pcm = ax.pcolormesh(X, Y, Mig_rate,
                        norm=colors.AsinhNorm(linear_width=lnrwidth,
                                                vmin=-gain, vmax=gain),
                        **shadeopts)
    plt.yscale('log')
    plt.xscale('log')
    # plt.yticks([1,3,10,30],[1,3,10,30])
    # plt.ylim(0.1,1e3)
    plt.xlabel(r'$r[AU]$')
    plt.ylabel(r'$M_p[M_\oplus]$')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ticks = np.array([-1e-5,-1e-6,-1e-7,-1e-8,1e-8,1e-7,1e-6,1e-5])
    cbar = plt.colorbar(pcm, cax=cax, ticks=ticks,label=r'$\dot{a}/a$')

    system = run_single_pps(ax, disk, planets, M_star, R_star, dt, end_time, dt_plot)

    plt.savefig("migration_map.png",dpi=500)
