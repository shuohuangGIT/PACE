import numpy as np
import sys

#Feel free to use, please cite
# - Liu & Ormel (2018) [Paper I]
# - Ormel & Liu (2018) [Paper II]
#

def epsilon (mode='', times_eta=False, **pars):
    """
    returns the pebble accretion efficiency "\epsilon" defined in:

        Mdot = \epsilon * 2\pi r v_dr \Sigma 
             = \epsilon * (pebble mass flux)

    where:
        -\Sigma :surface density of pebbles
        - v_dr  :pebble drift velocity
        - r     :location (semi-major axis) of planet

    MODE is a string that should include one of:
        [2d,3d,]   :2d (planar) limit, 3d limit; or mix (default)
        [f,]        :inclusion of fset modulation factor 
        [set,bal,]  :settling only; ballistic only; or mix (default)
                    :(mix always triggers calculation of fset factor)

    examples, mode =
        2dset   :2d (planar), settling regime; Eq. A4, Paper I
        3dsetf  :3d regime, including modulation factor; Eq. 21 Paper II
        set     :interpolation b/w 2d and 3d limits, Eq. 38 Paper II
        2dbalf  :2d, ballistic regime; Eq. A.15 Paper I
        3dbalf  :3d, ballistic regime; Eq. A.16 Paper I
        ''      :all effects

    PARS is a list of keyword arguments. Valid keywords are
        qp      :planet-to-star mass ratio
        tau     :dimensionless stoping time (tstop*\Omega)
        alphaz  :turbulence diffusivity parameterization in z-direction
        taucorr :dimensionless turbulence correlation time (tcorr*\Omega)
        hgas    :gas aspect ratio
        hP      :pebble aspect ratio (optional)
        eta     :gas dimensionless pressure gradient
        sigvec  :turbulence rms velocity components
        nvec    :relative strengths (x,y,z) turbulent velocity components
        Rp      :planet radius over orbital radius (for ballistic regime)
        ETC

    in case of turbulence, please specify the rms-velocities explicitly
    via sigvec OR based on alphaz using nvec method. If none is given then
    nvec = (1,1,1) is assumed.

    NOTES   :keywords name should be identical to the above list!
            :keywords order is immaterial   
            :python raises an error when a required keywords is not present
            :expressions valid only for tau<1


    HISTORY
        [18.08.14]: initial upload
        [21.09.08]: some corrections from Tommy Chi Ho Lau accounted for
        [22.01.18]: support for the "R" and "P" nondimensionalization forms
                    (see end of this file)
    """

    #these should always be present
    tau = pars['tau']
    qp = pars['qp']

    #xi factor by Youdin & Lithwick (usually taking ~unity)
    xi = xi_YL07 (**pars)

    #cases when we need to calculate fset modulation factor
    if mode.count('f'):
        doCalcfset = True
    else:
        doCalcfset = False

    #when we mix ballistic/settling regimes, always calculate fset
    if mode.count('set')==0 and mode.count('bal')==0:
        doCalcfset = True


    #Obtain the pebble scaleheight (needed for 3D)
    if mode.count('2d')==0:

        #obtain the pebble scaleheight, if not an argument
        #use Youdin & Lithwick expression
        if 'hP' in pars:
            hP = pars['hP']
        else:
            h0 = hp_Dubrulle (**pars)
            hP = h0 /np.sqrt(xi)  #Youdin & Lithwick correction term

        #find the effective scaleheight in case of inclined planets
        if 'ip' in pars:
            ip = pars['ip']
            heff = heff_app(hP, ip)
        else:
            ip = 0
            heff = hP

    #calculate delVy (usually, but not always, needed below)
    if mode.count('3d')==0 or mode.count('set')==0 or doCalcfset:
        #circular velocity (Paper I)
        vcir = v_circ(**pars)

        if 'ep' in pars:
            ep = pars['ep']
        else:
            ep = 0.0

        delVy = np.maximum(vcir, ae*ep) 
    else:
        delVy = 0.

    #in case eta*epsilon is asked for
    #simply put \eta=1 in the following
    if times_eta: 
        pars['eta'] = 1.0


    #the 3D limit, settling regime
    if mode.count('2d')==0 and mode.count('bal')==0:
        eps3D = eps_3D (heff=heff, **pars)

        #velocity in z direction
        delVz = ai *ip
    else:
        delVz = 0.
        eps3D = np.inf


    #the 2D limiting expression
    if mode.count('3d')==0 and mode.count('bal')==0:
        eps2D = eps_2D (delVy=delVy, **pars)
    else:
        eps2D = np.inf

    #the non-turbulent component of the velocity
    delVvec = [0., delVy, delVz]


    #the turbulent component of the velocity (can be zero)
    if doCalcfset or mode.count('set')==0:
        #the turbulent components of the velocity
        #get it from input parameters (for anisotropic turbulence)
        if 'sigvec' in pars:
            sigvec = pars['sigvec']

        #or relative to alpha-z
        elif 'alphaz' in pars:

            #relative weights specified or not (isotropy)
            if 'nvec' in pars:
                nvec = np.array(pars['nvec'])
            else:
                nvec = np.ones(3)

            #nvec is defined wrt the vertical velocity component
            sigvec = [nvec[k] *sig_turb(**pars) for k in range(3)]

        #no turbulence
        else:
            sigvec = np.zeros((3))

        #pebble rms velocity different from gas
        if 'taucorr' in pars:
            taucorr = pars['taucorr']
        else:
            taucorr = 1.0

        #particle rms velocities 
        #[21.09.08] added xi-correction term
        sigPvec = [sigvec[k] *np.sqrt(taucorr/(taucorr+tau)) /(np.sqrt(xi)) for k in range(3)]
    else:
        sigPvec = np.zeros((3))

    #calculate the fset reduction factor 
    if doCalcfset:
        vast = v_ast(**pars)
        fset = f_set (delVvec, sigPvec, vast)
    else:
        fset = 1.

    if mode.count('bal')==0:
        #interpolation formula (Eq. 39 OL18)
        eps2Dset = eps2D *fset
        eps3Dset = eps3D *fset**2

        if mode.count('2d'):
            epsset = eps2Dset
        elif mode.count('3d'):
            epsset = eps3Dset
        else:
            epsset = eps_23 (eps2Dset, eps3Dset)

    else:
        epsset = 0.


    #The ballistic regime (Paper I)
    if mode.count('set')==0:
        #obtain absolute velocity, including turbulence motions
        delV2 = [delVvec[k]**2 +sigPvec[k]**2 for k in range(3)]
        delV = np.sqrt(delV2[0] +delV2[1] +delV2[2])

        if 'Rp' not in pars:
            print('[OL18.py]error: No input physical radius "Rp" is given, yet ballistic effects are')
            print('                requested. Either provide Rp" or include "set" in the mode string.')
            sys.exit()

        
        #if settling is also calculated, reduce eps2Dbal
        #(this mimics aerodynamic deflection)
        if mode.count('3d')==0:
            eps2Dbal = eps_2D_bal (delV=delV, **pars)
            if doCalcfset: eps2Dbal *= (1-fset)

        #[22.09.28]I removed "hP=heff" from expression below
        #[23.07.26]in 3D heff is determined above and should be added here...
        #           (i.e., dont understand previous comment)
        if mode.count('2d')==0:
            eps3Dbal = eps_3D_bal (delV=delV, hP=heff, **pars)
            #[21.09.08]:corrected this expression
            if doCalcfset: eps3Dbal *= (1-fset**2)


        if mode.count('2d'):
            epsbal = eps2Dbal
        elif mode.count('3d'):
            epsbal = eps3Dbal
        else:
            #we assume the same mixing expresssion as in the settling case
            epsbal = eps_23 (eps2Dbal, eps3Dbal)

    else:
        epsbal = 0.

    #add the settling and balastic expressions 
    #Note: eq.41 of OL18 is wrong. We have already accounted for fset factors!
    epsgen = epsset +epsbal
    return epsgen


#all fit constants
A2 =    0.322
A3 =    0.393
ash =   0.515
acir =  5.66
aturb = 0.332
ae =    0.764
ai =    0.677
aset =  0.5

#to avoid 0/0
tiny = np.finfo(np.float64).tiny

def v_circ (tau, qp, eta=0, **dumargs):
    """
    circular velocity
    Eqs. A9 of Paper I
    """
    vhw = eta        
    vsh = ash*(qp*tau)**(1./3)

    qc = eta**3/tau
    vcir = vsh +vhw *qc/(qc +acir*qp)
    return vcir


def xi_YL07 (tau, taucorr=1.0, **dumargs):
    """
    YL07 correction factor xi
    """
    xi = 1 + tau*taucorr**2 /(tau+taucorr)  #correction term
    return xi


def hp_Dubrulle (tau, hgas, alphaz, **dumargs):
    """
    This is the widely-used expression for the particle scaleight
    first derived (to my recollection) in Dubrulle et al. (1995)
    """
    return np.sqrt(alphaz/(alphaz+tau)) *hgas #the Dubrulle expression


def heff_app (hP, ip):
    """
    Approximation to the effective scaleheight in case
    of planet inclination (ip<>0)

    Eq. 26 of OL18
    """

    #avoid 0/0
    arg = hP**2 +0.5*np.pi*ip**2 *(1 -np.exp(-0.5*ip/(hP+tiny) ))
    return np.sqrt(arg)


def sig_turb (alphaz, hgas, taucorr=1.0, **dumargs):
    """
    the rms turbulent velocity. 
    Note its dependence on the correlation time taucorr, which 
        is usually taken unity
    """
    sigturb = alphaz**0.5 *hgas /np.sqrt(taucorr)

    return sigturb


def v_ast (tau, qp, **dumargs):
    """
    Characteristic velocity for pebble accretion 
    """
    return (qp/tau)**(1./3)


def f_set_i (delVi, sigi, vast):
    """
    Calculates fset for a single direction
    Eq. 35 in OL18
    """
    fset = np.exp(-aset *delVi**2/(vast**2 +aturb*sigi**2))\
            *vast/np.sqrt(vast**2 +aturb*sigi**2)

    return fset


def f_set (delVvec, sigPvec, vast):
    """
    Calculates fset for all direction
    Eq. 35 in OL18
    """
    fset = 1.
    for k in range(3):
        fset *= f_set_i (delVvec[k], sigPvec[k], vast)

    return fset


def eps_3D (tau, qp, eta, heff, **dumargs): 
    """
    the 3D PA efficiency, uncorrected for f_set
    Eq. 39b of OL18
    """

    eps3D = A3*qp /(eta *(heff+tiny))
    return eps3D


def eps_2D (tau, qp, eta, delVy, **dumargs): 
    """
    the 2D PA efficiency, uncorrected for f_set
    Eq. 39a of OL18
    """

    eps2D = A2 *np.sqrt(qp/tau /eta**2 *delVy)
    return eps2D


def eps_2D_bal (tau, qp, eta, delV, Rp, **dumargs):
    """
    Equation (A.15) of Paper I, uncorrected for fset
    """
    eps2Dbal = Rp/(2*np.pi*tau*eta) *np.sqrt(2*qp/Rp +delV**2)
    return eps2Dbal


def eps_3D_bal (tau, qp, eta, delV, Rp, hP, **dumargs):
    """
    Equation (A.15) of Paper I, uncorrected for fset
    """
    eps3Dbal = 1./(4*np.sqrt(2*np.pi)*tau*eta*hP) *\
                (2*qp*Rp/delV +Rp**2 *delV)
    return eps3Dbal


def eps_23 (eps2D, eps3D):
    """
    The mixing expressions
    Essentially, eps23 = min(eps2D, eps3D)
    Eq. 40 of OL18 
    """
    eps23 = (eps2D**-2 +eps3D**-2)**-0.5
    return eps23


def rate_P (mode='', **pars):
    """
    this provides the dimensionaless pebble accretion rate "P"
    as defined in Hill units:

        Mdot = P rHill**2 Omega Sigma
             = P (qp/3)**(2/3) r**2 Omega Sigma

    where:
        - Omega :orbital frequency at location planet     
        - Sigma :surface density of pebbles
        - r     :location (semi-major axis) of planet
        - rHill :Hill radius
    """
    qp = pars['qp']
    return rate_R(mode, **pars) *(3/qp)**(2/3)


def rate_R (mode='', **pars):
    """
    this provides the dimensionless pebble accretion rate "R" 
    as defined:

        Mdot =  R r**2 \Omega \Sigma

    where:
        - R     :dimensionless accretion rate
        - Mdot  :pebble accretion rate on planet
        - Sigma :surface density of pebbles
        - r     :location (semi-major axis) of planet
        - Omega :orbital frequency at location planet     

    it can be shown that: k = 4pi tau * (eta * epsilon)
    to avoid singularities when eta-->0 (epsilon -> infinity)
    the product eta*epsilon is obtained directly
    """
    epseta = epsilon (mode, times_eta=True, **pars)
    tau = pars['tau']
    return 4*np.pi *tau *epseta


