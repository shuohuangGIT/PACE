import numpy as np
from amuse.units import units, constants
from amuse.datamodel import Particles
import matplotlib.pyplot as plt

pre_dt = 0.1 | units.kyr # timescale for integration
pre_ndisk = 500

def Rhills(Mp,Mstar,ap):
    return ap*(Mp/3/Mstar)**(1/3)

def Rdisk0(Rdisk_in, Rdisk_out, ndisk):
    # return np.linspace(4,9,ndisk) | units.au
    return 10**np.linspace(np.log10(Rdisk_in/(1|units.au)), np.log10(Rdisk_out/(1|units.au)), ndisk) | units.au

def temperature (Rdisk, pT, star_mass):
    # r in unit au; M in unit solar mass
    return (280 | units.K) * (Rdisk/(1|units.au))**pT *(star_mass/(1|units.MSun))

def sound_speed(temperature, mu):
    return np.sqrt(constants.kB*temperature/mu/constants.atomic_mass_unit_hyphen_kilogram_relationship)

def scale_height(cs, Mstar, ap):
    omega = np.sqrt(constants.G*Mstar/ap**3)
    return cs/omega

# initial gas surface density
def sigma_g0(fg, pg0, Rdisk, Rdisk_in, Rdisk_out):
    # typos from Mordasini??
    sigmag_0 = 2400 | units.g/units.cm**2
    sigmag = sigmag_0 * fg* (Rdisk/(1|units.au))**pg0 * np.exp(-(Rdisk/Rdisk_out)**(2-pg0))*(1-np.minimum(np.sqrt(Rdisk_in/Rdisk),1)) #* (np.array((Rdisk/Rdisk_out)**(2-pg0))<10)
    sigmag0 = np.maximum(sigmag.value_in(units.g/units.cm**2), 1e-300)|units.g/units.cm**2
    return sigmag0

# initial dust surface density
def sigma_d0(sigma_g, fDG, FeH, temperature):
    judge = (temperature<(170|units.K))
    eta_ice = judge*0.75 + 0.25
    return  fDG * 10**FeH * eta_ice* sigma_g

def dynamical_mass(core_mass, envelope_mass):
    return core_mass + envelope_mass

def initialize_planetary_system(N_bodies, star_mass, planet_mass, semimajor_axis, eccentricity, inclination, argument_of_pericenter, longitude_of_ascending_node):

    masses = np.zeros(N_bodies) | units.MEarth
    masses[0] = star_mass
    masses[1:] = planet_mass

    N_binaries = N_bodies-1
    particles = Particles(N_bodies+N_binaries)
    for index in range(N_bodies):
        particle = particles[index]
        particle.mass = masses[index]
        particle.is_binary = False
        particle.radius = 1.0 | units.RSun
        particle.child1 = None
        particle.child2 = None

    for index in range(N_binaries):
        particle = particles[index+N_bodies]
        particle.is_binary = True
        # particle.mass = masses[index+1]
        particle.semimajor_axis = semimajor_axis[index]
        particle.eccentricity = eccentricity[index]
        particle.inclination = inclination[index]
        particle.argument_of_pericenter = argument_of_pericenter[index]
        particle.longitude_of_ascending_node = longitude_of_ascending_node[index]
        
        ### specify the `2+2' hierarchy; this is easy to change to the `3+1' hierarchy
        if index==0:
            particle.child1 = particles[0]
            particle.child2 = particles[1]
        elif index==1:
            particle.child1 = particles[2]
            particle.child2 = particles[3]
        elif index==2:
            particle.child1 = particles[4]
            particle.child2 = particles[5]        
    binaries = particles[particles.is_binary]

    return particles, binaries

def compute_mutual_inclination(INCL_k,INCL_l,LAN_k,LAN_l):
    cos_INCL_rel = np.cos(INCL_k)*np.cos(INCL_l) + np.sin(INCL_k)*np.sin(INCL_l)*np.cos(LAN_k-LAN_l)
    return np.arccos(cos_INCL_rel)

def plot_function(data):
    print_times_Myr,print_smas_AU,print_rps_AU,print_parent_is_deg,canonical_rp_min_A_AU,canonical_rp_min_B_AU = data

    N_binaries = len(print_smas_AU)

    plt.rc('text',usetex=True)
    plt.rc('legend',fancybox=True)
            
    linewidth=4
    dlinewidth=2
    fig=plt.figure(figsize=(10,9))
    plot1=fig.add_subplot(2,1,1,yscale="log")
    plot2=fig.add_subplot(2,1,2)

    #from distinct_colours import get_distinct

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    labels = ["$A$","$B$","$C$"]
    labels_i = ["$i_{AC}$","$i_{BC}$",None]
    for index_binary in range(N_binaries):
        label = labels[index_binary]
        label_i = labels_i[index_binary]        
        color = colors[index_binary]
        
        plot1.plot(print_times_Myr,print_smas_AU[index_binary],color=color,linestyle='dashed',linewidth=dlinewidth)
        plot1.plot(print_times_Myr,print_rps_AU[index_binary],color=color,linewidth=linewidth,label=label)
        
        plot2.plot(print_times_Myr,print_parent_is_deg[index_binary],color=color,linewidth=linewidth,label=label_i)

    plot1.axhline(y = canonical_rp_min_A_AU, color= colors[0],linestyle='dotted',linewidth=dlinewidth)
    plot1.axhline(y = canonical_rp_min_B_AU, color= colors[1],linestyle='dotted',linewidth=dlinewidth)

    handles,labels = plot1.get_legend_handles_labels()
    plot1.legend(handles,labels,loc="upper right",fontsize=12)

    handles,labels = plot2.get_legend_handles_labels()
    plot2.legend(handles,labels,loc="lower right",fontsize=12)

    plot1.set_xlabel("t [Myr]",fontsize=18)
    plot2.set_xlabel("t [Myr]",fontsize=18)

    plot1.set_ylabel("$a_i [\mathrm{AU}]$",fontsize=18)
    plot2.set_ylabel("$i_{kl} [\mathrm{deg}]$",fontsize=18)

    plot1.set_xlim(0.0,print_times_Myr[-1])
    plot2.set_xlim(0.0,print_times_Myr[-1])

    plot1.tick_params(axis='both', which ='major', labelsize = 18)
    plot2.tick_params(axis='both', which ='major', labelsize = 18)

    fig.savefig("figure.pdf")

    plt.show()

# N_bodies = 3
# star_mass = 1 | units.MSun
# planet_mass = [1,2] | units.MEarth
# semimajor_axis = [1,2] | units.au
# eccentricity = [0,0]
# inclination = [0,0]
# argument_of_pericenter = [0,0]
# longitude_of_ascending_node = [0,0]

# particles, binaries = initialize_planetary_system(N_bodies, star_mass, planet_mass, semimajor_axis, eccentricity, inclination, argument_of_pericenter, longitude_of_ascending_node)
# print(particles)