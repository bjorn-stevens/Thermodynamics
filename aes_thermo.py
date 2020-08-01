# -*- coding: utf-8 -*-
"""
Created on Tuesday, May 22, 2019
Author: Bjorn Stevens (bjorn.stevens@mpimet.mpg.de)
"""
#
import numpy as np
from scipy import interpolate, optimize

gravity = 9.8076
Rstar   = 8.31446261815324
P0      = 100000.  # Standard Pressure [Pa]
T0      = 273.15   # Standard Temperature [K]
#
# Based on Park et al (2004) Meteorlogia, O2 levels are declining as CO2 levels rise, but at a tiny arte.
#
x_ar  = 9.332e-3
x_o2  = 0.20944
x_n2  = 0.78083
x_co2 = 0.415e-3
#
# Based on Chase (1998) J Phys Chem Ref Data
#
m_ar  = 39.948
m_o2  = 15.9994 * 2
m_n2  = 14.0067 * 2
# Based on Chase (1998) J Phys Chem Ref Data
#
gravity = 9.8076
Rstar   = 8.31446261815324
P0      = 100000.  # Standard Pressure [Pa]
T0      = 273.15   # Standard Temperature [K]
#
# Based on Park et al (2004) Meteorlogia, O2 levels are declining as CO2 levels rise, but at a tiny arte.
#
x_ar  = 9.332e-3
x_o2  = 0.20944
x_n2  = 0.78083
x_co2 = 0.415e-3
#
# Based on Chase (1998) J Phys Chem Ref Data
#
m_ar  = 39.948
m_o2  = 15.994  * 2
m_n2  = 14.0067 * 2
# Based on Chase (1998) J Phys Chem Ref Data
#
Rstar   = 8.31446261815324
P0      = 100000.  # Standard Pressure [Pa]
T0      = 273.15   # Standard Temperature [K]
#
# Based on Park et al (2004) Meteorlogia, O2 levels are declining as CO2 levels rise, but at a tiny arte.
#
x_ar  = 9.332e-3
x_o2  = 0.20944
x_n2  = 0.78083
x_co2 = 0.415e-3
#
# Based on Chase (1998) J Phys Chem Ref Data
#
m_ar  = 39.948
m_o2  = 15.994  * 2
m_n2  = 14.0067 * 2
m_co2 = 44.011
m_h2o = 18.01528

cp_ar  = 20.786  # 298.15K
cp_o2  = 29.376  # 298.15K or 29.126 @ 200K
cp_n2  = 29.124  # 298.15K or 29.107 @ 200K
cp_co2 = 37.129  # 298.15K or 32.359 @ 200K
cp_h2o = 33.349 + (33.590 - 33.349)/98.15 * (T0-200) # Interpolated to T0 from Chase values (but not used)

s0_ar  = 154.845  # 298.15K
s0_o2  = 205.147  # 298.15K
s0_n2  = 191.609  # 298.15K
s0_co2 = 213.795  # 298.15K
s0_h2o = 188.854  # 298.15

md    = x_ar*m_ar + x_o2*m_o2 + x_n2*m_n2 + x_co2*m_co2 # molar mass of dry air
q_ar  = x_ar *m_ar /md
q_o2  = x_o2 *m_o2 /md
q_n2  = x_n2 *m_n2 /md
q_co2 = x_co2*m_co2/md

Rd  = (Rstar/md)*(x_ar+x_o2+x_n2+x_co2) * 1000.  #J/kg/K
cpd = (   1./md)*(x_ar*cp_ar + x_o2*cp_o2 + x_n2*cp_n2 + x_co2*cp_co2) *1000.  #J/kg/K
sd00= (   1./md)*(x_ar*s0_ar + x_o2*s0_o2 + x_n2*s0_n2 + x_co2*s0_co2) * 1000.  + cpd * np.log(T0/298.15)  # Dry air entropy at P0, T0

es_default = 'analytic-liq'

cpv     = 1865.01   # IAPWS97 at 273.15 , for this we could use the Chase values, but they are closer to 1861
cl      = 4179.57   # IAPWS97 at 305 and P=0.1 MPa (chosen to give a good fit for es over ice)
ci      = 1905.43   # IAPWS97 at 247.065 and P=0.1 MPa (chosen to give a good fit for es over ice)
#
# cl and ci, especially ci, varies considerably with temperature.  Consider that
# cl = 4273 J/kg/K at 263 K decreases sharply to 4220 J/kg/K by 273 K and ever more slowly to 4179 J/kg/K at 313 K with most variation at lower temperatures
# ci = 1450 J/kg/K at 183 K and increases progressively to a value of 2132 J/kg/K at 278K
#
# At standard temperature and pressure they hav the values
#    cl      = 4219.32   # ''
#    ci      = 2096.70   # ''

lv0     = 2500.93e3 # IAPWS97 at 273.15
lf0     =  333.42e3 # ''
Rv      = (Rstar/m_h2o) *1000.  #J/kg/K
sv00    = (s0_h2o/m_h2o)*1000.  + cpv * np.log(T0/298.15)

eps1     = Rd/Rv
eps2     = Rv/Rd -1.

PvC     = 22.064e6 # Critical pressure [Pa] of water vapor
TvC     = 647.096  # Critical temperature [K] of water vapor
TvT     = 273.16   # Triple point temperature [K] of water
PvT     = 611.655
lvT     = lv0 + (cpv-cl)*(TvT-T0)
lfT     = lf0 + (cpv-ci)*(TvT-T0)
lsT     = lvT + lfT

def flatten_input(x):

    x = np.asarray(x).flatten()
    scalar_input = False
    if x.ndim == 0:
        x = x[None]  # Makes x 1D
        scalar_input = True

    return x, scalar_input

def es(T,es_formula=es_default):
    """ Returns the saturation vapor pressure of water over liquid or ice, or the minimum of the two,
    depending on the specificaiton of the state variable.  The calculation follows Wagner and Pruss (2002)
    fits (es[li]f) for saturation over planar liquid, and Wagner et al., 2011 for saturation over ice.  The choice
    choice of formulation was based on a comparision of many many formulae, among them those by Sonntag, Hardy,
    Romps, Murphy and Koop, and others (e.g., Bolton) just over liquid. The Wagner and Pruss and Wagner
    formulations were found to be the most accurate as cmpared to the IAPWS standard for warm temperatures,
    and the Wagner et al 2011 form is the IAPWS standard for ice.  Additionally an 'analytic' expression es[li]a
    for computations that require consisntency with assumption of cp's being constant can be selected.  The analytic
    expressions become identical to Romps in the case when the specific heats are adjusted to his suggested values.
    >>> es([273.16,290.])
    [611.65706974 1919.87719485]
    """

    def esif(T):
        a1 = -0.212144006e+2
        a2 =  0.273203819e+2
        a3 = -0.610598130e+1
        b1 =  0.333333333e-2
        b2 =  0.120666667e+1
        b3 =  0.170333333e+1
        theta = T/TvT
        return PvT * np.exp((a1*theta**b1 + a2 * theta**b2 + a3 * theta**b3)/theta)

    def eslf(T):
        vt = 1.-T/TvC
        return PvC * np.exp(TvC/T * (-7.85951783*vt + 1.84408259*vt**1.5 - 11.7866497*vt**3 + 22.6807411*vt**3.5 - 15.9618719*vt**4 + 1.80122502*vt**7.5))

    def esla(T):
        c1 = (cpv-cl)/Rv
        c2 = lvT/(Rv*TvT) - c1
        return PvT * np.exp(c2*(1.-TvT/x)) * (x/TvT)**c1

    def esia(T):
        c1 = (cpv-ci)/Rv
        c2 = lsT/(Rv*TvT) - c1
        return PvT * np.exp(c2*(1.-TvT/x)) * (x/TvT)**c1

    x,  scalar_input = flatten_input(T)

    if (es_formula == 'liq'):
        es = eslf(x)
    if (es_formula == 'ice'):
        es = esif(x)
    if (es_formula == 'mxd'):
        es = np.minimum(esif(x),eslf(x))
    if (es_formula == 'analytic-liq'):
        es = esla(x)
    if (es_formula == 'analytic-ice'):
        es = esia(x)
    if (es_formula == 'analytic-mxd'):
        es = np.minimum(esia(x),esla(x))

    if scalar_input:
        return np.squeeze(es)
    return es

def phase_change_enthalpy(Tx,fusion=False):
    """ Returns the enthlapy [J/g] of vaporization (default) of water vapor or
    (if fusion=True) the fusion anthalpy.  Input temperature can be in degC or Kelvin
    >>> phase_change_enthalpy(273.15)
    2500.8e3
    """

    TK, scalar_input = flatten_input(Tx)
    if (fusion):
        el = lf0 + (cl-ci)*(TK-T0)
    else:
        el = lv0 + (cpv-cl)*(TK-T0)

    if scalar_input:
        return np.squeeze(el)
    return el

def pp2sm(pv,p):
    """ Calculates specific mass from the partial and total pressure
    assuming both have same units and no condensate is present.  Returns value
    in units of kg/kg. checked 15.06.20
    >>> pp2sm(es(273.16),60000.)
    0.00636529
    """

    pv,  scalar_input1 = flatten_input(pv) # don't specify pascal as this will wrongly corrected
    p ,  scalar_input2 = flatten_input(p )
    scalar_input = scalar_input1 and scalar_input2

    x   = eps1*pv/(p-pv)
    sm  = x/(1+x)
    if scalar_input:
        return np.squeeze(sm)
    return sm

def pp2mr(pv,p):
    """ Calculates mixing ratio from the partial and total pressure
    assuming both have same unitsa nd no condensate is present. Returns value
    in units of kg/kg. Checked 20.03.20
    """

    pv,  scalar_input1 = flatten_input(pv) # don't specify pascal as this will wrongly corrected
    p ,  scalar_input2 = flatten_input(p )
    scalar_input = scalar_input1 and scalar_input2

    mr = eps1*pv/(p-pv)
    if scalar_input:
        return np.squeeze(mr)
    return mr

def mr2pp(mr,p):
    """ Calculates partial pressure from mixing ratio and pressure, if mixing ratio
    units are greater than 1 they are normalized by 1000.
    checked 20.03.20
    """

    mr,  scalar_input1 = flatten_input(mr)
    p ,  scalar_input2 = flatten_input(p )
    scalar_input = scalar_input1 and scalar_input2

    ret = mr*p/(eps1+mr)
    if scalar_input:
        return np.squeeze(ret)
    return ret

def get_pseudo_theta_e(T,P,qt,es_formula=es_default):
    """ Calculates pseudo equivalent potential temperature. following Bolton
    checked 31.07.20
    """

    TK,  scalar_input1 = flatten_input(T)
    PPa, scalar_input2 = flatten_input(P)
    qt,  scalar_input3 = flatten_input(qt)
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    rs = pp2mr(es(TK,es_formula),PPa)
    rv = qt/(1.-qt)
    rv = np.minimum(rv,rs)
    pv = mr2pp(rv,PPa)

    Tl      = 55.0 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
    theta_e = TK*(P0/PPa)**(0.2854*(1.0 - 0.28*rv)) * np.exp((3376./Tl - 2.54)*rv*(1+0.81*rv))

    if scalar_input:
        return np.squeeze(theta_e)
    return(theta_e)

def get_theta_e(T,P,qt,es_formula=es_default):
    """ Calculates equivalent potential temperature corresponding to Eq. 2.42 in the Clouds
    and Climate book.
    checked 19.03.20
    """

    TK,  scalar_input1 = flatten_input(T)
    PPa, scalar_input2 = flatten_input(P)
    qt,  scalar_input3 = flatten_input(qt)
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * eps1 * (1.0 - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    Re = (1.0-qt)*Rd
    R  = Re + qv*Rv
    pv = qv * (Rv/R) *PPa
    RH = pv/ps
    lv = phase_change_enthalpy(TK)
    cpe= cpd + qt*(cl-cpd)
    omega_e = RH**(-qv*Rv/cpe) * (R/Re)**(Re/cpe)
    theta_e = TK*(P0/PPa)**(Re/cpe)*omega_e*np.exp(qv*lv/(cpe*TK))

    if scalar_input:
        return np.squeeze(theta_e)
    return(theta_e)

def get_theta_l(T,P,qt,es_formula=es_default):
#   """ Calculates liquid-water potential temperature.  Following Stevens and Siebesma
#   Eq. 2.44-2.45 in the Clouds and Climate book
#   """

    TK,  scalar_input1 = flatten_input(T)
    PPa, scalar_input2 = flatten_input(P)
    qt,  scalar_input3 = flatten_input(qt)
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * eps1 * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    R  = Rd*(1-qt) + qv*Rv
    Rl = Rd + qt*(Rv - Rd)
    cpl= cpd + qt*(cpv-cpd)
    lv = phase_change_enthalpy(TK)

    omega_l = (R/Rl)**(Rl/cpl) * (qt/(qv+1.e-15))**(qt*Rv/cpl)
    theta_l = (TK*(P0/PPa)**(Rl/cpl)) *omega_l*np.exp(-ql*lv/(cpl*TK))

    if scalar_input:
        return np.squeeze(theta_l)
    return(theta_l)

def get_theta_s(T,P,qt,es_formula=es_default):
#   """ Calculates entropy potential temperature. This follows the formulation of Pascal
#   Marquet and ensures that parcels with different theta-s have a different entropy
#   """

    TK,  scalar_input1 = flatten_input(T)
    PPa, scalar_input2 = flatten_input(P)
    qt,  scalar_input3 = flatten_input(qt)
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    kappa = Rd/cpd
    e0    = es(T0,es_formula)
    Lmbd  = ((sv00 - Rv*np.log(e0/P0)) - (sd00 - Rd*np.log(1-e0/P0)))/cpd
    lmbd  = cpv/cpd - 1.
    eta   = 1/eps1
    delta = eps2
    gamma = kappa/eps1
    r0    = e0/(P0-e0)/eta

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * eps1 * (1. - qt)
    qv = np.minimum(qt,qs)
    ql = qt-qv

    lv = phase_change_enthalpy(TK)

    R  = Rd + qv*(Rv - Rd)
    pv = qv * (Rv/R) *PPa
    RH = pv/ps
    rv = qv/(1-qv)

    x1 = 1
    x1 = (T/T0)**(lmbd*qt) * (P0/PPa)**(kappa*delta*qt) * (rv/r0)**(-gamma*qt) * RH**(gamma*ql)
    x2 = (1.+eta*rv)**(kappa*(1+delta*qt)) * (1+eta*r0)**(-kappa*delta*qt)
    theta_s = (TK*(P0/PPa)**(kappa)) * np.exp(-ql*lv/(cpd*TK)) * np.exp(qt*Lmbd) * x1 * x2

    if scalar_input:
        return np.squeeze(theta_s)
    return(theta_s)

def get_theta_rho(T,P,qt,es_formula=es_default):
#   """ Calculates theta_rho as theta_l * (1+Rd/Rv qv - qt)
#   """

    TK,  scalar_input1 = flatten_input(T)
    PPa, scalar_input2 = flatten_input(P)
    qt,  scalar_input3 = flatten_input(qt)
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    theta_l = get_theta_l(TK,PPa,qt,es_formula)

    ps = es(TK,es_formula)
    qs = (ps/(PPa-ps)) * (Rd/Rv) * (1. - qt)
    qv = np.minimum(qt,qs)
    theta_rho = theta_l * (1.+ qv/eps1 - qt)

    if scalar_input:
        return np.squeeze(theta_rho)
    return(theta_rho)

def T_from_Te(Te,P,qt,es_formula=es_default):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Te(350.,1000.,17)
	304.4761977
    """

    def zero(T,Te,P,qt):
        return  np.abs(Te-get_theta_e(T,P,qt,es_formula))
    return optimize.fsolve(zero,   200., args=(Te,P,qt), xtol=1.e-10)

def T_from_Tl(Tl,P,qt,es_formula=es_default):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Tl(282.75436951,90000,20.e-3)
	290.00
    """
    def zero(T,Tl,P,qt):
        return  np.abs(Tl-get_theta_l(T,P,qt,es_formula))
    return optimize.fsolve(zero,   200., args=(Tl,P,qt), xtol=1.e-10)

def T_from_Ts(Ts,P,qt,es_formula=es_default):
    """ Given theta_e solves implicitly for the temperature at some other pressure,
    so that theta_e(T,P,qt) = Te
	>>> T_from_Tl(282.75436951,90000,20.e-3)
	290.00
    """
    def zero(T,Ts,P,qt):
        return  np.abs(Ts-get_theta_s(T,P,qt,es_formula))
    return optimize.fsolve(zero,   200., args=(Ts,P,qt), xtol=1.e-10)

def P_from_Te(Te,T,qt,es_formula=es_default):
    """ Given Te solves implicitly for the pressure at some temperature and qt
    so that theta_e(T,P,qt) = Te
	>>> P_from_Te(350.,305.,17)
	100464.71590478
    """
    def zero(P,Te,T,qt):
        return np.abs(Te-get_theta_e(T,P,qt,es_formula))
    return optimize.fsolve(zero, 90000., args=(Te,T,qt), xtol=1.e-10)

def P_from_Tl(Tl,T,qt,es_formula=es_default):
    """ Given Tl solves implicitly for the pressure at some temperature and qt
    so that theta_l(T,P,qt) = Tl
	>>> T_from_Tl(282.75436951,290,20.e-3)
	90000
    """
    def zero(P,Tl,T,qt):
        return np.abs(Tl-get_theta_l(T,P,qt,es_formula))
    return optimize.fsolve(zero, 90000., args=(Tl,T,qt), xtol=1.e-10)

def get_Plcl(T,P,qt,es_formula=es_default,iterate=False):
    """ Returns the pressure [Pa] of the LCL.  The routine gives as a default the
    LCL using the Bolton formula.  If iterate is true uses a nested optimization to
    estimate at what pressure, Px and temperature, Tx, qt = qs(Tx,Px), subject to
    theta_e(Tx,Px,qt) = theta_e(T,P,qt).  This works for saturated air.
	>>> Plcl(300.,1020.,17)
	96007.495
    """

    def delta_qs(P,Te,qt,es_formula=es_default):
        TK = T_from_Te(Te,P,qt)
        ps = es(TK,es_formula)
        qs = (1./(P/ps-1.)) * eps1 * (1. - qt)
        return np.abs(qs/qt-1.)

    TK,  scalar_input1 = flatten_input(T)
    PPa, scalar_input2 = flatten_input(P)
    qt,  scalar_input3 = flatten_input(qt)
    scalar_input = scalar_input1 and scalar_input2 and scalar_input3

    if (iterate):
        Te   = get_theta_e(TK,PPa,qt,es_formula)
        if scalar_input:
            Plcl = optimize.fsolve(delta_qs, 80000., args=(Te,qt), xtol=1.e-10)
            return np.squeeze(Plcl)
        else:
            if (scalar_input3):
                qx =np.empty(np.shape(Te)); qx.fill(np.squeeze(qt)); qt = qx
            elif len(Te) != len(qt):
                print('Error in get_Plcl: badly shaped input')

        Plcl = np.zeros(np.shape(Te))
        for i,x in enumerate(Te):
            Plcl[i] = optimize.fsolve(delta_qs, 80000., args=(x,qt[i]), xtol=1.e-10)
    else: # Bolton
        cp = cpd + qt*(cpv-cpd)
        R  = Rd  + qt*(Rv-Rd)
        pv = mr2pp(qt/(1.-qt),PPa)
        Tl = 55 + 2840./(3.5*np.log(TK) - np.log(pv/100.) - 4.805)
        Plcl = PPa * (Tl/TK)**(cp/R)

    return Plcl

def get_Zlcl(Plcl,T,P,qt,Z,):
    """ Returns the height of the LCL assuming temperature changes following a
    dry adiabat with vertical displacements from the height where the ambient
    temperature is measured.
	>>> Zlcl(300.,1020.,17)
	96007.495
    """
    cp = cpd + qt*(cpv-cpd)
    R  = Rd  + qt*(Rv-Rd)

    return T*(1. - (Plcl/P)**(R/cp)) * cp/gravity + Z
