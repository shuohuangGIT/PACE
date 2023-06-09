import numpy as np
import matplotlib.pyplot as plt
import sys

from amuse.units import units, constants
from amuse.datamodel import Particles, Particle, new_regular_grid
from venice_src.venice import Venice

from extra_funcs import *

class PebbleGasAccretion:

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
        self.disk.alpha = 1e-4
        self.st = 0.01 # stokes number

        # Model parameters
        self.BL = 1 #feeding zone size =6 for planetesimal accretion
        # self.miso1 = 25 | units.MEarth
        # Integration parameters
        self.eta = 0.1
        
        # planet atmosphere
        self.kappa = 0.005 | units.m**2/units.kg
    
    def feedzone(self):
        Sigmadmean = np.zeros(len(self.planets)) | units.g/units.cm**2
        Mfeed = np.zeros(len(self.planets)) | units.g
        imax = np.zeros(len(self.planets))
        imin = np.zeros(len(self.planets))
        ip = np.zeros(len(self.planets))
        RHs = np.zeros(len(self.planets)) | units.au
        Rdisk = self.disk.position

        Sigmad = self.disk.surface_solid
        fliss = self.BL

        for i in range(len(self.planets)):
            ap = self.planets[i].semimajor_axis
            RH = Rhills(self.planets[i].core_mass,self.star.mass,ap)
            RHs[i] = RH
            Rmin=ap-fliss*RH
            Rmax=ap+fliss*RH

            Mfeedi = 0. | units.g

            if (Rmin<Rdisk[-1]) and (Rmin>Rdisk[0]):
                imini = np.nonzero(Rdisk<=Rmin)[0][-1]
            elif (Rmin<=Rdisk[0]):
                imini = 0
            else:
                imini = -2

            if (Rmax<Rdisk[-1]) and (Rmax>Rdisk[0]):
                imaxi = np.nonzero(Rdisk<=Rmax)[0][-1]
            elif (Rmax<=Rdisk[0]):
                imaxi = 0
            else:
                imaxi = -2

            ipi   = np.nonzero(Rdisk<=ap)[0][-1]
            
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
        return Sigmadmean, Mfeed, imax, imin, ip, RHs

    def Mdotcore(self):
        mdotc = np.zeros(len(self.planets)) | units.MEarth/units.kyr
        Sigmadmean, Mfeed, imax, imin, ip, RHs = self.feedzone()
        Mstar = self.star.mass
        misos = np.zeros(len(self.planets)) | units.MEarth
        Hdisk = self.disk.scale_height
        Rdisk = self.disk.position
        sigmad = self.disk.surface_solid
        sigmag = self.disk.surface_gas

        for i in range(len(self.planets)):
            ap = self.planets[i].semimajor_axis
            mc = self.planets[i].core_mass
            
            #calculate slope
            xi = - np.log(Hdisk[int(ip[i])+1]/Hdisk[int(ip[i])])/np.log(Rdisk[int(ip[i])+1]/Rdisk[int(ip[i])])*2+3
            beta = - np.log(sigmag[int(ip[i])+1]/sigmag[int(ip[i])])/np.log(Rdisk[int(ip[i])+1]/Rdisk[int(ip[i])])

            H = (Hdisk[int(ip[i])]-Hdisk[int(ip[i])+1])*(Rdisk[int(ip[i])]-ap)/(Rdisk[int(ip[i])]-Rdisk[int(ip[i])+1])+Hdisk[int(ip[i])]
            miso = (25|units.MEarth)*(0.34*(-3/np.log10(self.disk.alpha[int(ip[i])]))**4+0.66)*(1-(1.5*(1-xi)-beta+2.5)/6)*(H/ap/0.05)**3
            misos[i] = miso
            sigmadp = (sigmad[int(ip[i])]-sigmad[int(ip[i])+1])*(Rdisk[int(ip[i])]-ap)/(Rdisk[int(ip[i])]-Rdisk[int(ip[i])+1])+sigmad[int(ip[i])]
            if (mc>=miso) or (self.planets[i].isohist==True):
                mdotc[i] = 0 | units.g/units.s
                self.planets[i].isohist = True
                # if (sigmadp<1e-20|units.g/units.cm**2):
                #     mdotc[i] = 0 | units.g/units.s
                # else:
                #     omegap = np.sqrt(constants.G*Mstar/ap**3)
                #     mdotc[i] = 2*(self.st/0.1)**(2/3)* sigmadp*RHs[i]**2* omegap  # pebble accretion
            else:
                if (sigmadp<1e-20|units.g/units.cm**2):
                    mdotc[i] = 0 | units.g/units.s
                else:
                    omegap = np.sqrt(constants.G*Mstar/ap**3)
                    mdotc[i] = 2*(self.st/0.1)**(2/3)* sigmadp*RHs[i]**2* omegap  # pebble accretion
        return mdotc, Sigmadmean, Mfeed, imax, imin, ip, misos

    def Mdotgas(self, misos, ip):
        if len(misos) != len(self.planets):
            print('dimension is not consistent!')
            sys.exit(0)
        mdotg = np.zeros(len(self.planets)) | units.g/units.s
        mass_p = self.planets.dynamical_mass
        ap = self.planets.semimajor_axis
        kappa = self.kappa
        Sigmag = self.disk.surface_gas
        Mstar = self.star.mass
        Hdisk = self.disk.scale_height
        Rdisk = self.disk.position
        for i in range(len(self.planets)):
            M_crit = misos[i]
            alpha = self.disk.alpha[int(ip[i])]
            if mass_p[i] < M_crit:  
                mdotg[i] = 0 | units.g/units.s
            else:
                mdotg[i] = (10**-5|units.MEarth/units.yr)*(mass_p[i].value_in(units.MEarth)/10)**4*(kappa/(0.1|units.m**2/units.kg))**(-1) # Johansen2019; Ikoma2000

            omega = np.sqrt(constants.G*Mstar/ap[i]**3)
            H = (Hdisk[int(ip[i])]-Hdisk[int(ip[i])+1])*(Rdisk[int(ip[i])]-ap[i])/(Rdisk[int(ip[i])]-Rdisk[int(ip[i])+1])+Hdisk[int(ip[i])]

            h=H/ap[i]
            q = mass_p[i]/Mstar
            K = (q)**2*(h)**-5/alpha

            mdotbondi = 0.29/np.pi/3*(h)**(-4)*(q)**(4/3)/(1+0.04*K)/alpha # Johansen2019; Tanigawa & Tanaka2016
            ff = min(mdotbondi,1) # min(mgdot, mgdot_bondi)
            
            # mdotbondi = Sigmag[int(ip[i])]/Hdisk[int(ip[i])]*omega*(Rhills(mass_p[i],Mstar,ap[i])/3)**3 # mordasini 2015
            nudisk = alpha*(h*ap[i])**2*omega
            Mdotdisk = 3*np.pi*nudisk*Sigmag[int(ip[i])]
            # Mdotdisk = 10**-8 | units.MSun/units.yr
            # print(Mdotdisk.value_in(units.MSun/units.yr), ff, mdotg[i].value_in(units.MSun/units.yr))
            mdotg[i] = min(mdotg[i],ff*Mdotdisk)

        return mdotg

    def set_time_scale(self, Mdotcore, Mdotenvelop, model_time_i, end_time):
        dt_core = 0.01*min(self.planets.core_mass/Mdotcore)
        dt_env = 0.01*min(self.planets.envelope_mass/Mdotenvelop)
        dt = abs(min(dt_core, dt_env, model_time_i- end_time))
        return dt
    
    def evolve_model(self, end_time):
        model_time_i = self.model_time
        Rdisk = self.disk.position
        Sigmad = self.disk.surface_solid

        if (min(self.planets.semimajor_axis)>Rdisk[0]) and (max(self.planets.semimajor_axis)<Rdisk[-1]):
            while model_time_i < end_time: # improved from pps.py
            
                Mdotcore, Sigmadmean, Mfeed, imax, imin, ip, misos = self.Mdotcore()
                Mdotenvelop = self.Mdotgas(misos, ip)
                dt = self.set_time_scale(Mdotcore, Mdotenvelop, model_time_i, end_time)
                model_time_i += dt

                for i in range(len(self.planets)):
                                        
                    if Mfeed[i]>dt* Mdotcore[i]:
                        Mfeed[i] -= dt* Mdotcore[i]
                        
                    else:
                        Mdotcore[i] = Mfeed[i]/dt
                        Mfeed[i]= 0. | units.g

                    # Sigmadmean[i]=Mfeed[i]/(np.pi*(Rdisk[int(imax[i])]**2-Rdisk[int(imin[i])]**2))

                    # for j in range(int(imin[i]), int(imax[i])+1):
                    #     Sigmad[j] = Sigmadmean[i].value_in(units.g/units.cm**2) | units.g/units.cm**2

                    self.planets[i].core_mass += Mdotcore[i] * dt
                    self.planets[i].envelope_mass += Mdotenvelop[i] * dt # gas accretion
        else:
            pass

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
    
from module_migration import *
#--------------------------------
def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    core_gas_accretion = PebbleGasAccretion()
    typeI_migration = TypeIMigration()

    # Add codes
    system.add_code(core_gas_accretion)
    system.add_code(typeI_migration)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix[0,1] = timestep

    system.add_channel(0,1, from_attributes = ['core_mass'], 
                to_attributes = ['core_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
                
    system.add_channel(0,1, from_attributes = ['envelope_mass'], 
                to_attributes = ['envelope_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
    system.add_channel(0,1, from_attributes = ['surface_solid'], 
                to_attributes = ['surface_solid'], 
                from_set_name = 'disk', to_set_name = 'disk')
    system.add_channel(1,0, from_attributes=['semimajor_axis'], 
                to_attributes=['semimajor_axis'],
                from_set_name='planets', to_set_name='planets')
    
    return system, core_gas_accretion, typeI_migration


def run_single_pps (disk, stokes_number, xi, beta, planets, Miso1, star_mass, dt, end_time, dt_plot):
    system, _, _ = setup_single_pps(dt)
    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass
    system.codes[0].st  = stokes_number

    system.codes[1].planets.add_particles(planets)
    system.codes[1].disk = disk
    system.codes[1].star.mass = star_mass
    system.codes[1].xi = xi
    system.codes[1].beta = beta
    system.codes[1].TypeI = True

    print(Miso1.value_in(units.MEarth))

    N_plot_steps = int(end_time/dt_plot)
    t = np.zeros((N_plot_steps+1, len(planets))) | units.Myr
    a = np.zeros((N_plot_steps+1, len(planets))) | units.au
    M = np.zeros((N_plot_steps+1, len(planets))) | units.MEarth
    Me = np.zeros((N_plot_steps+1, len(planets))) | units.MEarth

    fig = plt.figure(0,figsize=(5,6))
    ax = plt.subplot(2,1,1)
    ax.plot(system.codes[0].disk.position.value_in(units.AU), system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2),color = (1-0,0,0))

    for i in range(N_plot_steps+1):

        system.evolve_model( (i) * dt_plot )

        print (system.codes[0].model_time.value_in(units.Myr), end_time.value_in(units.Myr))

        t[i] = system.codes[0].model_time
        a[i] = system.codes[0].planets.semimajor_axis
        M[i] = system.codes[0].planets.dynamical_mass
        Me[i] = system.codes[0].planets.envelope_mass
        cm1 = plt.cm.get_cmap('bwr_r')

        color = cm1(t[i]/end_time)
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


    ax = plt.subplot(2,1,2)
    for i in range(len(planets)):
        ax.plot(a[:,i].value_in(units.au), M[:,i].value_in(units.MEarth),label='simulated total mass')
        ax.plot(a[:,i].value_in(units.au), (M[:,i]-Me[:,i]).value_in(units.MEarth), label='simulated core mass')
    ax.set_xlabel('sma [au]')
    ax.set_ylabel('M [$M_\oplus$]')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(0.1,100)
    ax.set_ylim(0.01,1000)  

    return system,ax


if __name__ == '__main__':

    M0 = 0.01
    M = [M0] | units.MEarth
    a = [25.] | units.AU

    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0|units.g,
        semimajor_axis = a,
        isohist = False
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 0.01 | units.kyr
    end_time = 200. | units.kyr
    dt_plot = end_time/1000
    disk = new_regular_grid(([pre_ndisk]), [1]|units.au)

    sigmag_0 = 2400 | units.g/units.cm**2
    fg = 1
    beta = 15/14
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU
    fDG, FeH, xi, star_mass = 0.02, 0, 3/7, 1|units.MSun
    mu = 2.3
    stokes_number = 0.01
    alpha = 1e-4

    disk.position = Rdisk0(Rdisk_in, Rdisk_out, pre_ndisk)
    disk.surface_gas = fg*sigmag_0 * (disk.position.value_in(units.au))**(-beta)

    T1 = 117|units.K
    T = T1*(disk.position.value_in(units.au))**(-xi)
    disk.surface_solid = disk.surface_gas *fDG
    disk.scale_height = scale_height(sound_speed(T, mu), star_mass, disk.position)
    disk.alpha=alpha
    # analytical
    
    st = stokes_number
    
    r0 = a[0]
    cs1 = np.sqrt(constants.kB*T1/mu/constants.u)
    print(cs1)
    kmig = (1.36+0.62*beta+0.43*xi)*2

    # pebble isolation mass
    vk1 = np.sqrt(constants.G*(1|units.MSun)/(1|units.au)) # keplerian velocity at r0
    Miso1 = (25|units.MEarth)*(0.34*(-3/np.log10(alpha))**4+0.66)*(1-(1.5*(1-xi)-beta+2.5)/6)*(cs1/vk1/0.05)**3

    system,ax = run_single_pps(disk, stokes_number, xi, beta, planets, Miso1, star_mass, dt, end_time, dt_plot)

    riso = 10**np.linspace(-1,2,1000) | units.au
    Miso = Miso1*(riso.value_in(units.au))**(1.5*(1-xi))
    
    ax.plot(riso.value_in(units.au),Miso.value_in(units.MEarth), 'k--',label='pebble isolation')
    M = 10**np.linspace(-2, 2.5, 1000) # units.MEarth
    Mmax43 = fDG * (4/3)*2*(st/0.1)**(2/3)*star_mass*(3*star_mass)**(-2/3)/kmig/constants.G*cs1**2*(1|units.au)**xi *r0**(1-xi)/(1-xi)+ (M0|units.MEarth)**(4/3)
    
    r = r0*(1-(M**(4/3)-M0**(4/3))/(Mmax43.value_in(units.MEarth**(4/3))-M0**(4/3)))**(1/(1-xi))
    r = r.value_in(units.au)

    ax.plot(r, M, 'k-.', label = 'Theoretical')
    ax.legend(loc = 'upper right')

    plt.show()
