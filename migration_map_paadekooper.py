import numpy as np
import params as ps
import scipy as sci
import matplotlib.pyplot as plt

Z_sol = 0.0196/100

def torque_L(beta,alpha,gamma):
    return (-2.5-1.7*beta+0.1*alpha)/gamma

def torque_hs_baro(alpha,gamma):
    return (1.1*(3/2-alpha))/gamma

def torque_hs_ent(beta,alpha,gamma):
    xi=beta-(gamma-1)*alpha
    return (7.9*xi/gamma)/gamma

def torque_c_lin_baro(alpha,gamma):
    return (0.7*(3/2-alpha))/gamma

def torque_c_lin_ent(beta,alpha,gamma):
    xi=beta-(gamma-1)*alpha
    return ((2.2-1.4/gamma)*xi)/gamma

def Fp(p):
    Fp = 1/(1+(p/1.3)**2)
    return Fp
    # return 8*sci.special.iv(4/3,p)/(3*p*sci.special.iv(1/3,p)+9/2*p**2*sci.special.iv(4/3,p))

def Gp(p):
    if (p<np.sqrt(8/45/np.pi)):
        Gp = 16/25*(45*np.pi/8)**(3/4)*p**(3/2)
    else:
        Gp = (1-9/25*(8/45/np.pi)**(4/3)*p**(-8/3))
    return Gp

def Kp(p):
    if (p<np.sqrt(28/45/np.pi)):
        Kp = 16/25*(45*np.pi/28)**(3/4)*p**(3/2)
    else:
        Kp = (1-9/25*(28/45/np.pi)**(4/3)*p**(-8/3))
    return Kp

def torque_cbaro(beta,alpha,gamma,p_vis):
    return torque_hs_baro(alpha,gamma)*Fp(p_vis)*Gp(p_vis)+(1-Kp(p_vis))*torque_c_lin_baro(alpha,gamma)

def torque_cent(beta,alpha,gamma,p_vis,p_therm):
    torque_c_ent = torque_hs_ent(beta,alpha,gamma)*Fp(p_vis)*Fp(p_therm)*np.sqrt(Gp(p_vis)*Gp(p_therm))+np.sqrt((1-Kp(p_vis))*(1-Kp(p_therm)))*torque_c_lin_ent(beta,alpha,gamma)
    return torque_c_ent
    
def torque_cent2(beta,alpha,gamma,p_vis):
    p_therm=p_vis
    torque_c_ent = torque_hs_ent(beta,alpha,gamma)*np.sqrt(Fp(p_vis)*Fp(p_therm))*np.sqrt(Gp(p_vis)*Gp(p_therm))+np.sqrt((1-Kp(p_vis))*(1-Kp(p_therm)))*torque_c_lin_ent(beta,alpha,gamma)
    return torque_c_ent

def torque_tot(beta,alpha,gamma,p_vis,p_therm,K):
    torque_c = torque_cbaro(beta,alpha,gamma,p_vis)+torque_cent(beta,alpha,gamma,p_vis,p_therm)
    torque_tot = (torque_L(beta,alpha,gamma)+torque_c*np.exp(-K/20))/(1+0.04*K)
    return torque_tot

def cal_surf_density(r_cm,r_in=0.091, beta_g=0.9, r_0=5.2, sigma_g0=100, R_disk=30):
    r = r_cm /ps.au
    return sigma_g0*(r/r_0)**(-beta_g)*np.exp(-(r/R_disk)**(2-beta_g))*(1-np.sqrt(r_in/r))

def cal_kappa_gas1(rho,T,Z):
    #simple version 
    T=np.array(T)
    inner = 160
    outer = 200
    smooth=50
    smoothi=(np.tanh((T-inner)/smooth)+1)/2
    smootho=(np.tanh((T-outer)/smooth)+1)/2
    kappa = 2e-4*T**2*(1-smoothi)+0.1*T**0.5*smootho #+2e16/T**7*(1-smootho)*smoothi
    # kappa_0=kappa/0.0196*np.maximum(Z,1e-6)
    kappa_0=kappa
    return kappa_0

def cal_soundspeed(T):
    mu = 2.24 # gas mean molecular weight
    return np.sqrt(ps.k_SB*T/mu/(ps.mProton+ps.mElectron))

def cal_omega(M_star, r):
    return np.sqrt(ps.G*M_star/r**3)

def cal_nu(T,alpha,M_star,r):
    mu = 2.24
    return alpha* ps.k_SB*T/mu/(ps.mProton+ps.mElectron) /cal_omega(M_star,r)

def fun_tem(M_star, R_star, T_star, r, alpha, T_prec, sigma, Z):
    
    omega = cal_omega(M_star, r)
    
    dT = 1e4
    T = T_star**4*2/3/np.pi*(R_star/r)**3           
    while dT > T_prec:
        T_old=T
        H     = cal_soundspeed(T)/omega
        rho   = sigma/np.sqrt(2*np.pi)/H

        kappa = cal_kappa_gas1(rho, T, Z)
        nu_vis= cal_nu(T, alpha, M_star, r)

        tau_R = kappa*sigma
        tau_P = 2.4*tau_R
        E_dot = 9/4*sigma*nu_vis*omega**2

        T_cd  = 10 # unit: K
        T_s4   = T_star**4*(2/3/np.pi*(R_star/r)**3+0.5*(R_star/r)**2* H/r*(9/7-1))+T_cd**4
        
        T = ((0.5*(3/8*tau_R+1/2/tau_P)*E_dot+ps.sigma_SB*T_s4)/ps.sigma_SB)**(1/4)
        if np.isinf(T):
            print(tau_R, tau_P, E_dot)
        elif np.isnan(T):
            print(tau_R, tau_P, E_dot)
        dT = abs(T_old-T)

    return T

def cal_temperature(rs, M_star, R_star, T_star, alpha, sigma, Z):
    # rs in unit cm
    rs = np.array(rs)
    T = np.zeros(len(rs))
    for i, r in enumerate(rs):
        result = fun_tem(M_star, R_star, T_star, r, alpha, 0.1, sigma[i], Z[i])
        T[i] = result
    return T

def cal_chi(gamma, temperature, kappa, sigma, omega):
    # gamma: gas
    if sigma<1e-100:
        chi = [np.inf]
    else:
        chi = 4*gamma*(gamma-1)*ps.sigma_SB*temperature**4*2*np.pi/3/kappa/sigma**2/omega**2
    return  chi #paardecooper != izidoro

def cal_xs(gamma_eff, M_star, M_planet,h):
    return 1.1/gamma_eff**(1/4)*np.sqrt(M_planet/M_star/h)

def cal_pvis(r,omega,xs,nu):
    return 2/3*np.sqrt(r**2*omega/2/np.pi/nu *xs**3)

def cal_ptherm(chi, r, M_star, M_planet, h, gamma_eff):
    omega = cal_omega(M_star, r)
    xs = cal_xs(gamma_eff, M_star, M_planet,h)
    return np.sqrt(r**2*omega/2/np.pi/chi*xs**3)

def fun_gamma_eff_izid(*args):
    gamma, chi, r, M_star, _, h = args
    omega = cal_omega(M_star, r)
    Q = 2*chi/3/h**3/omega/r**2
    # warning comes from too large or too small Chi
    if (chi>1e100):
        gamma_eff = 1
        # print(gamma_eff, chi)
    elif (chi==0):
        gamma_eff = 5/3
    else:
        gamma_eff = 2*Q*gamma/(gamma*Q+0.5*np.sqrt(2*np.sqrt((gamma**2*Q**2+1)**2-16*Q**2*(gamma-1))+2*gamma**2*Q**2-2))
    return gamma_eff


def cal_gamma_eff_izid(gamma, chi, rs, M_star, M_planet, h):
    # rs in unit cm
    rs = np.array(rs)
    chi=np.array(chi)
    h=np.array(h)
    gamma_eff = np.ones(len(rs))
    for i, r in enumerate(rs):
        # result = fsolve(fun_gamma_eff_izid, 2, args=(gamma, chi[i], r, M_star, M_planet, h[i]), xtol=1e-4)
        # gamma_eff[i] = result[0]
        result = fun_gamma_eff_izid(gamma, chi[i], r, M_star, M_planet, h[i])
        gamma_eff[i] = result
    
    return gamma_eff

def cal_tau_I(r, M_planet, M_star, gamma, sigma_g, sigma_d, temp, r_grid, alpha):
    ip   = np.nonzero(r_grid<=r)[0][-1]
    
    sigma_g_p = (sigma_g[ip]*(r_grid[ip+1]-r)+sigma_g[ip+1]*(r-r_grid[ip]))/(r_grid[ip+1]-r_grid[ip])
    sigma_d_p = (sigma_d[ip]*(r_grid[ip+1]-r)+sigma_d[ip+1]*(r-r_grid[ip]))/(r_grid[ip+1]-r_grid[ip])

    T = (temp[ip]*(r_grid[ip+1]-r)+temp[ip+1]*(r-r_grid[ip]))/(r_grid[ip+1]-r_grid[ip])
    T0 = temp[ip]
    T1 = temp[ip+1]

    if sigma_g_p==0:
        Z=0.01
    else:
        Z = sigma_d_p/sigma_g_p

    # surface density and temperature slope:    
    beta_slope  = -(np.log(T1/T0))/(np.log(r_grid[ip+1]/r_grid[ip]))
    if beta_slope == None:
        beta_slope=1e-4

    alpha_slope = -(np.log(sigma_g[ip+1]/sigma_g[ip]))/(np.log(r_grid[ip+1]/r_grid[ip]))
    if alpha_slope == None:
        alpha_slope=1e-4
    
    omega=cal_omega(ps.mSun,r)
    H = cal_soundspeed(T)/omega
    rho = sigma_g_p/np.sqrt(2*np.pi)/H
    kappa=cal_kappa_gas1(rho, T, Z)

    nu = cal_nu(T,2e-3,ps.mSun,r)
    chi = cal_chi(gamma, T, kappa, sigma_g_p, omega)
    
    h =  H/r
    gamma_eff = cal_gamma_eff_izid(gamma, chi, r, M_star, M_planet, h)
    xs = cal_xs(gamma_eff, M_star, M_planet, h)

    p_vis = cal_pvis(r,omega,xs,nu)
    p_therm = cal_ptherm(chi, r, M_star, M_planet, h, gamma_eff)
    
    K = (M_planet/M_star)**2*(h)**-5*alpha
 
    torque = torque_tot(beta_slope,alpha_slope,gamma_eff,p_vis,p_therm, K)

    mig_rate = (torque*(M_planet/M_star/h)**2*sigma_g_p*(r)**2*omega/M_planet*ps.yr  *2)
    return torque, mig_rate


if __name__ == '__main__':
    # star and planet:
    M_star=ps.mSun
    R_star=ps.rSun
    Teff = 5770

    # surface density model:
    r_in=0.091
    beta_g=0.9
    r_0=5.
    sigma_g0=200
    R_disk=30
    gamma=7/5
    alpha=2e-3

    r_grid  = 10**np.linspace(np.log10(r_in),2,200)*ps.au
    sigma_g = cal_surf_density(r_grid,r_in, beta_g, r_0, sigma_g0, R_disk)
    sigma_d = 0.01* sigma_g # simple

    rp  = (r_grid[:-1]+r_grid[1:])/2
    mp = 10**np.linspace(-1,2,200)*ps.mEarth

    X,Y = np.meshgrid(rp/ps.au,mp/ps.mEarth)
    Z=[]
    Mig_rate = []
    
    dtgr = sigma_d/sigma_g
    temp = cal_temperature(r_grid,M_star,R_star,Teff,alpha, sigma_g, dtgr)
    from tqdm import *
    for i, M_planet in enumerate(tqdm(mp)):
        Zi = []
        Mig_ratei = []
        for j, rpj in enumerate(rp):
            Zj, Mig_ratej = cal_tau_I(np.array([rpj]), M_planet, M_star, gamma, sigma_g, sigma_d, temp, r_grid, alpha)

            Zi.append(Zj[0])
            Mig_ratei.append(Mig_ratej[0])
        Z.append(Zi)
        Mig_rate.append(Mig_ratei)

    Z=np.array(Z)
    Mig_rate=np.array(Mig_rate)
    print(Z.shape,Mig_rate.shape)

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
    # print(levels)
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
