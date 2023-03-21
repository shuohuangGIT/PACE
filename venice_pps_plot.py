import matplotlib.pyplot as plt
import numpy as np
fontsize = 14

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

# line cyclers adapted to colourblind people
# from cycler import cycler
# line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
#                  cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
# marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
#                  cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
#                  cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
# plt.rc("axes", prop_cycle=line_cycler)
default_color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
fig = plt.figure(0,figsize=(25,8))
plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize)

ax = plt.subplot(2,2,3)
ax.set_xscale('log')
ax.set_yscale('log')
for i in range(Mc.shape[1]):
    ax.plot(t[:,i]/1e3, Mc[:,i], linestyle='-',label=r'$M_\mathrm{core,%i}$'%(i+1), color=default_color[i])
    ax.plot(t[:,i]/1e3, Me[:,i], linestyle='--',label=r'$M_\mathrm{envelope,%i}$'%(i+1), color=default_color[i])
ax.set_xlabel('Time [Myr]', fontsize=fontsize)
ax.set_ylabel('M [$M_\oplus$]', fontsize=fontsize)

plt.legend(loc='upper left', fontsize=fontsize)
cm1 = plt.cm.get_cmap('bwr')

for i in range(gas.shape[0]):

    print ('real time:', time[i], 'end time:', time[-1], '(Myr)')

    color = cm1(time[i]/1e4)

    label_s = '%.1f Myr'%(time[i]/1e3)
    label_g = '%.1f Myr'%(time[i]/1e3)
    
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

ax = plt.subplot(2,2,4)
for i in range(Mc.shape[1]):
    ax.plot(t[:,i]/1e3, a[:,i],label=r'$a_{%i}$'%(i+1))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time [Myr]', fontsize=fontsize)
ax.set_ylabel('a [au]', fontsize=fontsize)
plt.legend(loc = 'lower left', fontsize = fontsize)
    
ax = plt.subplot(2,2,1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.01,1e2)
ax.set_ylim(1,250)
ax.set_xlabel('a [au]',fontsize=fontsize)
ax.set_ylabel(r'$\Sigma_d$[$g/cm^{2}$]',fontsize=fontsize)

# plt.legend(loc = 'upper right')

ax2 = ax.twinx()
ax2.set_ylabel(r'$M_\mathrm{p}[M_\oplus]$', color='b', fontsize=fontsize)
for i in range(Mc.shape[1]):   
    color = t[:,i]/1e3
    sc2 = ax2.scatter(a[:,i], Mc[:,i]+Me[:,i], marker='x', cmap=cm1, s=20, c = color)#, linewidths=0.5,edgecolors='grey')


ax=plt.axes((0.91,0.57,0.015,0.25))
cbar=plt.colorbar(sc2,cax=ax)
cbar.set_label(r"Time[Myr]", fontsize=fontsize)

ax2.set_ylim([5e-6,1e8])
ax2.set_xscale('log')
ax2.set_yscale('log')


ax = plt.subplot(2,2,2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.01,1e2)
ax.set_ylim(1,1e5)
ax.set_xlabel('a [au]',fontsize=fontsize)
ax.set_ylabel(r'$\Sigma_g$[$g/cm^{2}$]',fontsize=fontsize)
# plt.legend(loc = 'upper right')
plt.savefig('pps.pdf')
plt.show()
