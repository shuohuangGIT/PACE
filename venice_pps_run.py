import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units
from amuse.datamodel import Particles, new_regular_grid
from venice_src.venice import Venice

from extra_funcs import *
from module_coreaccretion import *
from module_gasevolution import *
from module_migration import *

def setup_single_pps (timestep, verbose=False):

    # Initiate Venice
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    core_gas_accretion = CoreGasAccretion()
    disk_gas_evolution = DiskGasEvolution()
    typeI_migration = TypeIMigration()

    # Add codes
    system.add_code(core_gas_accretion)
    system.add_code(disk_gas_evolution)
    system.add_code(typeI_migration)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix[0,1] = timestep
    system.timestep_matrix[0,2] = timestep/3
    system.timestep_matrix[1,2] = timestep

    system.add_channel(0,2, from_attributes = ['core_mass'], 
                to_attributes = ['core_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
                
    system.add_channel(0,2, from_attributes = ['envelope_mass'], 
                to_attributes = ['envelope_mass'],
                from_set_name = 'planets', to_set_name = 'planets')
    system.add_channel(0,2, from_attributes = ['surface_solid'], 
                to_attributes = ['surface_solid'], 
                from_set_name = 'disk', to_set_name = 'disk')

    system.add_channel(1,0, from_attributes = ['surface_gas'], 
                to_attributes = ['surface_gas'], 
                from_set_name = 'disk', to_set_name = 'disk')
    system.add_channel(1,2, from_attributes = ['surface_gas'], 
                to_attributes = ['surface_gas'], 
                from_set_name = 'disk', to_set_name = 'disk')

    system.add_channel(2,0, from_attributes=['semimajor_axis'], 
                to_attributes=['semimajor_axis'],
                from_set_name='planets', to_set_name='planets')

    return system, core_gas_accretion, disk_gas_evolution, typeI_migration


def run_single_pps (disk, planets, star_mass, dt, end_time, dt_plot):
    system,_,_,_ = setup_single_pps(dt)
    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass

    system.codes[1].disk = disk

    system.codes[2].planets.add_particles(planets)
    system.codes[2].disk = disk
    system.codes[2].star.mass = star_mass

    N_plot_steps = int(end_time/dt_plot)
    print(N_plot_steps)
    t = np.zeros((N_plot_steps+1, len(planets))) | units.Myr
    a = np.zeros((N_plot_steps+1, len(planets))) | units.au
    Mc = np.zeros((N_plot_steps+1, len(planets))) | units.MEarth
    Me = np.zeros((N_plot_steps+1, len(planets))) | units.MEarth

    fig = plt.figure(0,figsize=(10,8))
    # dict_disk = {}
    # dict_disk['position(au)'] = system.codes[0].disk.position.value_in(units.AU)
    gas = []
    solid = []
    disk_time = []
    for i in range(N_plot_steps+1):

        system.evolve_model( (i) * dt_plot )

        print ('real time:', system.codes[0].model_time.value_in(units.Myr), 
                'expect time:', ((i) * dt_plot).value_in(units.Myr), 
                'end time:', end_time.value_in(units.Myr), '(Myr)')

        t[i] = system.codes[0].model_time
        a[i] = system.codes[0].planets.semimajor_axis
        Mc[i] = system.codes[0].planets.core_mass
        Me[i] = system.codes[0].planets.envelope_mass
        
        # control the snapshots of disk
        N_snapshot = 10
        if i%(int(N_plot_steps/N_snapshot)) == 0:
            color = (1-(i)/N_plot_steps,0,(i)/N_plot_steps)
            disk_time.append(system.codes[0].model_time.value_in(units.kyr))

            label_s = '%.1f (kyr,solid)'%system.codes[0].model_time.value_in(units.kyr)
            label_g = '%.1f (kyr,gas)'%system.codes[0].model_time.value_in(units.kyr)
            
            ax = plt.subplot(2,2,1)
            solid.append(system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2))
            # dict_disk[label_s] = list(_flatten((system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2)).tolist()))
            ax.plot(system.codes[0].disk.position.value_in(units.AU), 
                    system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2),
                    color = color, 
                    label = label_s)
            
            ax = plt.subplot(2,2,2)
            gas.append(system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2))
            # dict_disk[label_g] = list(_flatten((system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2)).tolist()))
            ax.plot(system.codes[0].disk.position.value_in(units.AU), 
                    system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2),
                    color = color, 
                    label = label_g)
        
        print('core:', (system.codes[0].planets.core_mass)/(1|units.MEarth),'envelope:', (system.codes[0].planets.envelope_mass)/(1|units.MEarth),r'($M_\oplus$)')
        print('-------------')
    # print(dict_disk)
    # dataframe = pd.DataFrame(data=dict_disk)
    # dataframe.to_csv('disk.csv', index=True, sep = ',')
    np.savez('disk_data.npz', time=disk_time, position = system.codes[0].disk.position.value_in(units.AU), 
                gas=gas, solid=solid)

    ax = plt.subplot(2,2,1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01,1e2)
    ax.set_ylim(1,250)
    ax.set_xlabel('a [au]')
    ax.set_ylabel(r'$\Sigma_d$[$g/cm^{2}$]')
    plt.legend(loc = 'upper right')

    ax = plt.subplot(2,2,2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01,1e2)
    ax.set_ylim(1,1e5)
    ax.set_xlabel('a [au]')
    ax.set_ylabel(r'$\Sigma_g$[$g/cm^{2}$]')
    plt.legend(loc = 'upper right')

    ax = plt.subplot(2,2,3)
    for i in range(len(planets)):
        ax.plot(t[:,i].value_in(units.kyr), Mc[:,i].value_in(units.MEarth),label='core mass')
        ax.plot(t[:,i].value_in(units.kyr), Me[:,i].value_in(units.MEarth), label='atmosphere')
    ax.set_xlabel('Time [kyr]')
    ax.set_ylabel('M [$M_\oplus$]')

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.legend(loc='upper left')

    ax = plt.subplot(2,2,4)
    for i in range(len(planets)):
        ax.plot(t[:,i].value_in(units.kyr), a[:,i].value_in(units.au))

    ax.set_xlabel('Time [kyr]')
    ax.set_ylabel('a [au]')

    #savedata
    np.savez('planet_data.npz', time=t.value_in(units.kyr), Mc=Mc.value_in(units.MEarth), 
                Me=Me.value_in(units.MEarth), a = a.value_in(units.au))
    return system


if __name__ == '__main__':

    M = [1e-5, 1e-4] | units.MEarth
    a = [2.7, 7.5] | units.AU

    planets = Particles(len(M),
        core_mass=M,
        envelope_mass = 0|units.g,
        semimajor_axis = a
    )
    planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

    dt = 1 | units.kyr # timestep of the matrix
    end_time = 10000. | units.kyr
    dt_plot = end_time/400
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
