import numpy as np
import os
import matplotlib.pyplot as plt

datafile = 'planet_evo/'
N = 7000
fontsize = 15

a=[]
M=[]
P=[]
q=[]

for i in range(0,N):
    filename = datafile +'%06d_'%i+'planet.npz'
    if (os.path.exists(filename)):
        print(i)
        load_data_disk = np.load(filename, allow_pickle=True)
        load_data_planet = np.load(filename, allow_pickle=True)
        load_data_disk.keys()
        load_data_planet.keys()
        Mci = load_data_planet['Mc']
        Mei = load_data_planet['Me']
        ai = load_data_planet['a']
        star_massi = load_data_planet['star_mass']
        M.append(Mci[-1]+Mei[-1])
        a.append(ai[-1])
        P.append(ai[-1]**(3/2)*365.24*star_massi**-0.5)
        q.append((Mci[-1]+Mei[-1])/star_massi*3e-6)

a=np.array(a)
M=np.array(M)
P=np.array(P)
q=np.array(q)

fig = plt.figure(0,figsize=(8,8))
plt.plot(P,q,'o',color='grey')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-1,10**5)
plt.ylim(10**-6,10**-2)
plt.xlabel('period[day]', fontsize = fontsize)
plt.ylabel(r'Mass ratio', fontsize = fontsize)
plt.savefig('pps_P_q.png', dpi=500)
plt.close()

fig = plt.figure(1,figsize=(8,8))
plt.plot(a,M,'o',color='grey')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-2.5,10**4)
plt.ylim(10**-1,10**4)
plt.xlabel('sma[au]', fontsize = fontsize)
plt.ylabel(r'Mass[$M_\oplus$]', fontsize = fontsize)
plt.savefig('pps_a_M.png', dpi=500)
plt.close()

fig = plt.figure(2,figsize=(8,8))
plt.plot(P,M,'o',color='grey')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-1,10**6)
plt.ylim(10**-2,10**4)
plt.xlabel('Period[day]', fontsize = fontsize)
plt.ylabel(r'Mass[$M_\oplus$]', fontsize = fontsize)
plt.savefig('pps_P_M.png', dpi=500)
plt.close()