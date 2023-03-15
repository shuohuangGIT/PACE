import matplotlib.pyplot as plt
import numpy as np

load_data_disk = np.load('disk_data.npz',allow_pickle=True)
load_data_planet = np.load('planet_data.npz',allow_pickle=True)
load_data_disk.keys()
load_data_planet.keys()

t = load_data_planet['time']
Mc = load_data_planet['Mc']
Me = load_data_planet['Me']
a = load_data_planet['a']

position = load_data_disk['position']
time = load_data_disk['time']
gas = load_data_disk['gas']
solid = load_data_disk['solid']

# print(Mc[0])

fig = plt.figure(0,figsize=(10,8))
ax = plt.subplot(2,2,3)
ax.set_xscale('log')
ax.set_yscale('log')
for i in range(Mc.shape[1]):
    ax.plot(t[:,i], Mc[:,i],label=r'$M_\mathrm{c,%i}$'%(i+1))
    ax.plot(t[:,i], Me[:,i], label=r'$M_\mathrm{gas,%i}$'%(i+1))
ax.set_xlabel('Time [kyr]')
ax.set_ylabel('M [$M_\oplus$]')

plt.legend(loc='upper left')

ax = plt.subplot(2,2,4)
for i in range(Mc.shape[1]):
    ax.plot(t[:,i], a[:,i])

ax.set_xlabel('Time [kyr]')
ax.set_ylabel('a [au]')

for i in range(gas.shape[0]):

    print ('real time:', time[i], 'end time:', time[-1], '(Myr)')

    color = (1-(i)/(gas.shape[0]-1),0,(i)/(gas.shape[0]-1))

    label_s = '%.1f kyr'%time[i]
    label_g = '%.1f kyr'%time[i]
    
    ax = plt.subplot(2,2,1)
    ax.plot(position, 
            solid[i,:],
            color = color, 
            label = label_s)
    
    ax = plt.subplot(2,2,2)
    ax.plot(position, 
            gas[i,:],
            color = color, 
            label = label_g)
    
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

plt.show()
