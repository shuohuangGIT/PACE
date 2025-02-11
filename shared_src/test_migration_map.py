from amuse.units import units
import numpy as np
from migration_map_paadekooper import cal_tau_I, cal_temperature

def access_migration_map(rp, Mp, Ms, gamma, sigmag, sigmad, tempd, rgrid, alpha):
    M_planet, M_star, sigma_g, sigma_d, r_grid = Mp.value_in(units.g), Ms.value_in(units.g), sigmag.value_in(units.g/units.cm**2), sigmad.value_in(units.g/units.cm**2), rgrid.value_in(units.cm)
    rpj = rp.value_in(units.cm)
    temp_d = tempd.value_in(units.K)
    Z, Mig_ratej = cal_tau_I(np.array([rpj]), M_planet, M_star, gamma, sigma_g, sigma_d, temp_d, r_grid, alpha)

    return Z, Mig_ratej|units.yr**-1


if __name__ == '__main__':
    # star and planet:
    import numpy as np
    # import params as ps
    import matplotlib.pyplot as plt
    from amuse.units import units
    from extra_funcs import *

    M_star=1|units.MSun
    R_star=1|units.RSun
    Teff = 5770|units.K

    # surface density model:
    gamma=7/5
    alpha=2e-3

    sigmag_0 =  40 | units.g/units.cm**2
    fg = 5
    pg0 = -1
    Rdisk_in = 0.03 | units.AU
    Rdisk_out = 30 | units.AU

    mu = 2.4

    r_grid  = Rdisk0(0.1|units.AU, 15|units.AU, 200)

    sigma_g = sigma_g0(sigmag_0, fg, pg0, r_grid, Rdisk_in, Rdisk_out)
    # from migration_map_paadekooper import cal_surf_density
    # r_in, beta_g, r_0, sigma0, R_disk = 0.091, 0.9, 5., 200, 30
    # sigma_g = cal_surf_density(r_grid.value_in(units.cm), r_in, beta_g, r_0, sigma0, R_disk) | units.g/units.cm**2
    sigma_d = sigma_g*0.01 #*(1-0.99*(r_grid<(10|units.au))*(r_grid>(0.01|units.au)))
    
    # plt.plot(r_grid.value_in(units.AU), sigma_g.value_in(units.g/units.cm**2))
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    rp  = (r_grid[:-1]+r_grid[1:])/2
    
    mp = 10**np.linspace(-1,4,200) | units.MEarth

    X,Y = np.meshgrid(rp.value_in(units.au),mp.value_in(units.MEarth))
    Z=[]
    
    Mig_rate = []
    from tqdm import *
    dtgr = sigma_d/sigma_g

    temp_d = cal_temperature(r_grid.value_in(units.cm),M_star.value_in(units.g),R_star.value_in(units.cm),Teff.value_in(units.K),alpha, sigma_g.value_in(units.g/units.cm**2), dtgr) *(1|units.K)
    
    for i, M_planet in enumerate(tqdm(mp)):
        Zi = []
        Mig_ratei = []
        for j, rpj in enumerate(rp):
            # print(rpj, M_planet, M_star, gamma, sigma_g, sigma_d, temp_d, r_grid, alpha)
            Zj, Mig_ratej = access_migration_map(rpj, M_planet, M_star, gamma, sigma_g, sigma_d, temp_d, r_grid, alpha)

            Zi.append(Zj[0])
            Mig_ratei.append(Mig_ratej[0].value_in(units.yr**-1))
        Z.append(Zi)
        Mig_rate.append(Mig_ratei)

    Z=np.array(Z)
    Mig_rate=np.array(Mig_rate)

    # plot
    fig, ax = plt.subplots(figsize=(8,4),dpi=100)
    import matplotlib.colors as colors
    levels = np.linspace(-3,3,200)
    # cnt = ax.contourf(X, Y, Z, levels=levels,extend='both', cmap='RdBu_r')
    lnrwidth = 0.001
    shadeopts = {'cmap': 'RdBu_r', 'shading': 'gouraud'}
    colormap = 'RdBu_r'
    gain = 3
    pcm = ax.pcolormesh(X, Y, Z,
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
    ticks = np.array([-1,-0.1,-0.01,-0.001,0.001,0.01,0.1,1])
    cbar = plt.colorbar(pcm, cax=cax, ticks=ticks,label=r'$\Gamma_I/\Gamma_0$')
    plt.savefig('torque_map.png',dpi=300)


    fig, ax = plt.subplots(figsize=(8,4),dpi=100)
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

    plt.savefig('migration_rate_map.png',dpi=300)