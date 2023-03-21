import numpy as np
import matplotlib.pyplot as plt

from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid

from venice_src.venice import Venice

import sys

from extra_funcs import *

class CoreGasAccretion:

    def __init__(self):
        # Model data
        # add envelope attribute
        self.planets = Particles()
        self.planets.add_calculated_attribute('dynamical_mass', dynamical_mass)
        self.star = Particle(mass=1|units.MSun)
        self.model_time = 0. | units.Myr
        
        # setup disk
        self.disk = new_regular_grid(([int(pre_ndisk)]),[1])
        self.disk.surface_gas = 0 | units.g/units.cm**2
        self.disk.surface_solid = 0 | units.g/units.cm**2
        self.disk.scale_height = 0.03 * self.disk.position
        self.alpha = 1e-3
        self.flub = 0.9 # disk accretion fraction

        # Model parameters
        self.CMdotc = 1
        self.BL = 6 #feeding zone size
        self.rhoe = 1 | units.g/units.cm**3
        self.mpla = 1e18 | units.g

        # Integration parameters
        self.eta = 0.1

        # planet atmosphere
        self.kappa = 1e-2 | units.g/units.cm**2
        self.pKH = 10.4
        self.qKH = -1.5
    
    def feedzone(self):
        Sigmadmean = np.zeros(len(self.planets)) | units.g/units.cm**2
        Mfeed = np.zeros(len(self.planets)) | units.g
        imax = np.zeros(len(self.planets))
        imin = np.zeros(len(self.planets))
        ip = np.zeros(len(self.planets))
        
        Rdisk = self.disk.position

        Sigmad = self.disk.surface_solid
        fliss = self.BL

        for i in range(len(self.planets)):
            ap = self.planets[i].semimajor_axis
            RH = Rhills(self.planets[i].dynamical_mass,self.star.mass,ap)
            Rmin=ap-fliss*RH
            Rmax=ap+fliss*RH

            Rmin=max(Rmin, Rdisk[1])
            Rmax=min(Rmax, Rdisk[-2])
            
            Mfeedi = 0. | units.g

            for j in range(len(Rdisk)-1):
                if (Rdisk[j]<=Rmin) & (Rdisk[j+1]>Rmin):
                    imini=j

                if (Rdisk[j]<Rmax) & (Rdisk[j+1]>=Rmax):
                    imaxi=j
                
                if (Rdisk[j]<=ap) & (Rdisk[j+1]>ap):
                    ipi=j

            # improved from mo
            if imaxi == imini:
                    imaxi = imini+1

            for j in range(imini, imaxi):
                Mfeedi += np.pi* (Rdisk[j+1]**2-Rdisk[j]**2)*(Sigmad[j+1]+Sigmad[j])/2

            Sigmadmeani=Mfeedi/(np.pi*(Rdisk[imaxi]**2-Rdisk[imini]**2))
            
            Sigmadmean[i] = Sigmadmeani
            Mfeed[i] = Mfeedi
            imax[i] = imaxi
            imin[i] = imini
            ip[i] = ipi
        return Sigmadmean, Mfeed, imax, imin, ip

    def Mdotcore(self):
        Sigmag = self.disk.surface_gas  ####!!! need evolve gas disk
        mdotc = np.zeros(len(self.planets)) | units.MEarth/units.kyr
        Sigmadmean, Mfeed, imax, imin, ip = self.feedzone()
        Mstar = self.star.mass
        sigmagmin = 1|units.g/units.cm**2

        for i in range(len(self.planets)):
            ap = self.planets[i].semimajor_axis
            mp = self.planets[i].dynamical_mass
            mc = self.planets[i].core_mass
            Sigmag_p = Sigmag[int(ip[i])]
            if (Sigmadmean[i]<1e-20|units.g/units.cm**2):
                mdotc[i] = 0 | units.g/units.s
            else:
                Sigmag_p0 = 0|units.g/units.cm**2

                tau_cgas = (1.2e5|units.yr) *(Sigmadmean[i]/(10|units.g/units.cm**2))**(-1)*(ap/(1|units.AU))**(0.5)*(mp/(1|units.MEarth))**(1/3)*(Mstar/(1|units.MSun))**(-1/6)*((Sigmag_p/(2400|units.g/units.cm**2))**(-2/5)*(ap/(1|units.AU))**(2/20)*(self.mpla/(1e18|units.g))**(2/15))

                tau_cnog = (1e7|units.yr)*(Sigmadmean[i]/(10|units.g/units.cm**2))**(-1)*(mp/(1|units.MEarth))**(1/3) *(self.rhoe/(1|units.g/units.cm**3))**(2/3) *(Mstar/(1|units.MSun))**0.5*(ap/(1|units.AU))**1.5
                # smooth??
                tau_cnog *= (1+ Sigmag_p/sigmagmin)
                #consider scattering
                Rp = (3.*mp/(4.*self.rhoe))**(1/3)
                Vsurf=np.sqrt(constants.G*mp/Rp)
                Vesc=np.sqrt(2*constants.G*Mstar/ap)
                tau_cnog *= 1+(Vsurf/Vesc)**4

                tau_c = min(tau_cgas, tau_cnog)

                mdotc[i] = self.CMdotc*mc/tau_c # CMdotc is an artificial number =1
        return mdotc, Sigmadmean, Mfeed, imax, imin, ip

    def Mdotgas(self, Mdotcore,ip):
        if len(Mdotcore) != len(self.planets):
            print('dimension is not consistent!')
            sys.exit(0)
        mdotg = np.zeros(len(self.planets)) | units.g/units.s
        mass_p = self.planets.dynamical_mass
        ap = self.planets.semimajor_axis
        kappa = self.kappa
        pKH = self.pKH
        qKH = self.qKH
        Sigmag = self.disk.surface_gas
        Mstar = self.star.mass
        Hdisk = self.disk.scale_height
        me = self.planets.envelope_mass
        alpha = self.alpha
        flub = self.flub
        for i in range(len(self.planets)):
            M_crit = (10|units.MEarth)*(Mdotcore[i]/(1e-6|units.MEarth/units.yr))**(1/4) * (kappa/(1|units.g/units.cm**2))**(1/4)            
            if mass_p[i] < M_crit:  
                mdotg[i] = 0 | units.g/units.s
            else:
                tau_KH = 10**pKH |units.yr *(mass_p[i]/(1|units.MEarth))**qKH *(kappa/(1|units.g/units.cm**2))
                mdotg[i] = mass_p[i]/tau_KH
            omega = np.sqrt(constants.G*Mstar/ap[i]**3)
            mdotbondi = Sigmag[int(ip[i])]/Hdisk[int(ip[i])]*omega*(Rhills(mass_p[i],Mstar,ap[i])/3)**3
            nudisk = alpha*Hdisk[int(ip[i])]**2*omega
            Mdotdisk = flub*3*np.pi*nudisk*Sigmag[int(ip[i])]
            mdotg[i] = min(mdotbondi,mdotg[i],Mdotdisk)

            #Truncation at the gas isolation mass ---------------1
            # Meiso=(4.*np.pi*2.*ap[i]**2*Sigmag[int(ip[i])])**1.5/(3*Mstar)**0.5
            # if me[i]>Meiso:
            #     mdotg[i] = 0 | units.g/units.s

            # ------------------------------2
            # if 1.5*Hdisk[int[i]]>Rhills(mass_p[i],Mstar,ap[i]):
            #     mdotg[i]=0

            # ------------------------------3
            f_va04 = 1.668* (mass_p[i]/(1.5|units.MJupiter))**(1/3)*np.exp(-mass_p[i]/(1.5|units.MJupiter))+0.04
            mdot_eva04 = f_va04*Mdotdisk/flub
            mdotg[i] = min(mdotg[i],mdot_eva04)
            

            

        return mdotg

    def set_time_scale(self, Mdotcore, Mdotenvelop, model_time_i, end_time):
        dt_core = min(self.planets.core_mass/Mdotcore)
        dt_env = min(self.planets.envelope_mass/Mdotenvelop)
        dt = abs(min(dt_core, dt_env, model_time_i- end_time))
        return dt
    
    def evolve_model(self, end_time):
        model_time_i = self.model_time
        Rdisk = self.disk.position
        Sigmad = self.disk.surface_solid

        while model_time_i < end_time: # improved from pps.py
            Mdotcore, Sigmadmean, Mfeed, imax, imin, ip = self.Mdotcore()
            Mdotenvelop = self.Mdotgas(Mdotcore, ip)
            dt = self.set_time_scale(Mdotcore, Mdotenvelop, model_time_i, end_time)
            model_time_i += dt

            for i in range(len(self.planets)):
                                    
                if Mfeed[i]>dt* Mdotcore[i]:
                    Mfeed[i] -= dt* Mdotcore[i]
                    
                else:
                    Mdotcore[i] = Mfeed[i]/dt
                    Mfeed[i]= 0. | units.g

                Sigmadmean[i]=Mfeed[i]/(np.pi*(Rdisk[int(imax[i])]**2-Rdisk[int(imin[i])]**2))
                
                # necessary??
                # if (Sigmadmean[i]<1e-20|units.g/units.cm**2):
                #     Sigmadmean[i]=0.|units.g/units.cm**2

                for j in range(int(imin[i]), int(imax[i])+1):
                    Sigmad[j] = Sigmadmean[i].value_in(units.g/units.cm**2) | units.g/units.cm**2

                self.planets[i].core_mass += Mdotcore[i] * dt
                self.planets[i].envelope_mass += Mdotenvelop[i] * dt
        
        self.model_time = end_time # improved from pps.py
        self.disk.surface_solid = Sigmad

    def diskd_mass(self):
        Mddisk = 0|units.g
        Rdisk = self.disk.position
        Sigmad = self.disk.surface_solid
        for i in range(len(Rdisk)-1):
            Mddisk+=np.pi*(Rdisk[i+1]**2-Rdisk[i]**2)*0.5*(Sigmad[i+1]+Sigmad[i])
        return Mddisk

    def diskg_mass(self):
        Mgdisk = 0|units.g
        Rdisk = self.disk.position
        Sigmag = self.disk.surface_gas
        for i in range(len(Rdisk)-1):
            Mddisk+=np.pi*(Rdisk[i+1]**2-Rdisk[i]**2)*0.5*(Sigmag[i+1]+Sigmag[i])
        return Mgdisk
    

#--------------------------------
def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    core_gas_accretion = CoreGasAccretion()

    # Add codes
    system.add_code(core_gas_accretion)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix = timestep

    return system, core_gas_accretion #, typeI_migration


def run_single_pps (disk, planets, star_mass, dt, end_time, dt_plot):
    system, _ = setup_single_pps(dt)
    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass

    N_plot_steps = int(end_time/dt_plot)
    t = np.zeros((N_plot_steps, len(planets))) | units.Myr
    a = np.zeros((N_plot_steps, len(planets))) | units.au
    M = np.zeros((N_plot_steps, len(planets))) | units.MEarth
    Me = np.zeros((N_plot_steps, len(planets))) | units.MEarth

    fig = plt.figure(0,figsize=(5,6))
    ax = plt.subplot(2,1,1)
    ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2),color = (1-0,0,0))

    for i in range(N_plot_steps):

        system.codes[0].evolve_model( (i+1) * dt_plot )

        print (system.codes[0].model_time.value_in(units.Myr), end_time.value_in(units.Myr))

        t[i] = system.codes[0].model_time
        a[i] = system.codes[0].planets.semimajor_axis
        M[i] = system.codes[0].planets.dynamical_mass
        Me[i] = system.codes[0].planets.envelope_mass
        color = (1-(i+1)/N_plot_steps,0,(i+1)/N_plot_steps)
        ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2),color = color, label = '%.1f kyr'%system.codes[0].model_time.value_in(units.kyr))
        print('total', np.sum(system.codes[0].planets.dynamical_mass)/(1|units.MEarth),'Mearth')
        print('envelope', np.sum(system.codes[0].planets.envelope_mass)/(1|units.MEarth),'Mearth')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01,1e2)
    ax.set_ylim(1,1e5)
    ax.set_xlim(0.01,50)
    ax.set_ylim(1,250)
    ax.set_xlabel('R [Au]')
    ax.set_ylabel(r'$\Sigma_d$[$g/cm^{2}$]')
    plt.legend(loc = 'upper right')
    ax = plt.subplot(2,1,2)
    for i in range(len(planets)):
        ax.plot(t[:,i].value_in(units.kyr), M[:,i].value_in(units.MEarth),label='total mass')
        ax.plot(t[:,i].value_in(units.kyr), Me[:,i].value_in(units.MEarth), label='atmosphere')
    ax.set_xlabel('Time [kyr]')
    ax.set_ylabel('M [$M_\oplus$]')

    ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.legend(loc='upper left')

    return system


if __name__ == '__main__':

    M = [1e-3] | units.MEarth
    a = [20.] | units.AU

    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0|units.g,
        semimajor_axis = a
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 100 | units.kyr
    end_time = 4000. | units.kyr
    dt_plot = end_time/10
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
    plt.show()
