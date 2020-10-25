import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
import math
import cmath
import pickle
from IPython.display import Image
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp2d
from qutip import *
# qutip.settings.has_mkl = False



# In[ ]:




# # spectrum $S_\Gamma$
# \begin{equation}
#     S_{\Gamma}^{(1)}(\omega_1) = \frac{\Gamma}{2\pi}\langle a_1^\dagger a_1 \rangle
# \end{equation}

# In[ ]:


# def calc_spectra(rho00, lv, kappa_d, wd1, a, Nall):    
def calc_S_Gamma(rho00, lv, kappa_d, wd1, a, Nall):    
#     rho00 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)

    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd1)).data, operator_to_vector(1.j*a*rho00))
    rho10 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho01 = rho10.trans().conj()
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*(a*rho01-rho10*a.dag())))
    rho11 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    return (kappa_d/(2.*np.pi))*rho11.tr().real;


def calc_S_Gamma_alpha(rho00, lv, kappa_d, wd1, a, Nall, alpha):    
#     rho00 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)

    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd1)).data, operator_to_vector(1.j*(a+alpha)*rho00))
    rho10 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho01 = rho10.trans().conj()
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*((a+alpha)*rho01-rho10*(a.dag()+np.conj(alpha)))))
    rho11 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    return (kappa_d/(2.*np.pi))*rho11.tr().real;


# ## frequency-resolved correlation ($\tau=0$)
# \begin{equation}
# G_{\Gamma,\Gamma}^{(2)}(\omega_1,\omega_2) = \left(\frac{\Gamma}{2\pi}\right)^2\langle a_1^\dagger a_2^\dagger a_1 a_2 \rangle
# \end{equation}

# In[ ]:


# def calc_correlation(rho0000, lv, kappa_d, wd1, wd2, a, Nall):    
def calc_G_Gamma12(rho0000, lv, kappa_d, wd1, wd2, a, Nall):    
#     rho0000 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)

    # 25a
#     ww = np.linalg.inv(lv-(kappa_d/2.+1j*wd1)) * operator_to_vector(1.j*a*rho0000)
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd1)).data, operator_to_vector(1.j*a*rho0000))
    rho1000 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0010 = rho1000.trans().conj()

    # 25b
#     ww = np.linalg.inv(lv-(kappa_d/2.+1j*wd2)) * operator_to_vector(1.j*a*rho0000)
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd2)).data, operator_to_vector(1.j*a*rho0000))
    rho0100 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0001 = rho0100.trans().conj()

    # 25c
#     ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho0010-rho1000*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*(a*rho0010-rho1000*a.dag())))
    rho1010 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    # 25d
#     ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho0001-rho0100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*(a*rho0001-rho0100*a.dag())))
    rho0101 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    # 25e
#     ww = np.linalg.inv(lv-(kappa_d+1j*(wd1+wd2))) * operator_to_vector(1.j*a*(rho0100+rho1000))
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d+1j*(wd1+wd2))).data, operator_to_vector(1.j*a*(rho0100+rho1000)))
    rho1100 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0011 = rho1100.trans().conj()

    # 25f
#     ww = np.linalg.inv(lv-(kappa_d+1j*(wd1-wd2))) * operator_to_vector(1.j*(a*rho0001-rho1000*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d+1j*(wd1-wd2))).data, operator_to_vector(1.j*(a*rho0001-rho1000*a.dag())))
    rho1001 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0110 = rho1001.trans().conj()

    # 25g
#     ww = np.linalg.inv(lv-(1.5*kappa_d+1j*wd1)) * operator_to_vector(1.j*(a*rho0101+a*rho1001-rho1100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(1.5*kappa_d+1j*wd1)).data, operator_to_vector(1.j*(a*rho0101+a*rho1001-rho1100*a.dag())))
    rho1101 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0111 = rho1101.trans().conj()

    # 25h
#     ww = np.linalg.inv(lv-(1.5*kappa_d+1j*wd2)) * operator_to_vector(1.j*(a*rho0110+a*rho1010-rho1100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(1.5*kappa_d+1j*wd2)).data, operator_to_vector(1.j*(a*rho0110+a*rho1010-rho1100*a.dag())))
    rho1110 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho1011 = rho1110.trans().conj()

    # 25i
#     ww = np.linalg.inv(lv-(2.*kappa_d)) * operator_to_vector(1.j*(a*rho0111+a*rho1011-rho1101*a.dag()-rho1110*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(2.*kappa_d)).data, operator_to_vector(1.j*(a*rho0111+a*rho1011-rho1101*a.dag()-rho1110*a.dag())))
    rho1111 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    return (kappa_d/(2.*np.pi))**2*rho1111.tr().real;

# def calc_correlation(rho0000, lv, kappa_d, wd1, wd2, a, Nall):    
def calc_G_Gamma12_twooperators(rho0000, lv, kappa_d, wd1, wd2, a1, a2, Nall):    
#     rho0000 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)

    # 25a
#     ww = np.linalg.inv(lv-(kappa_d/2.+1j*wd1)) * operator_to_vector(1.j*a*rho0000)
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd1)).data, operator_to_vector(1.j*a1*rho0000))
    rho1000 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0010 = rho1000.trans().conj()

    # 25b
#     ww = np.linalg.inv(lv-(kappa_d/2.+1j*wd2)) * operator_to_vector(1.j*a*rho0000)
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd2)).data, operator_to_vector(1.j*a2*rho0000))
    rho0100 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0001 = rho0100.trans().conj()

    # 25c
#     ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho0010-rho1000*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*(a1*rho0010-rho1000*a1.dag())))
    rho1010 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    # 25d
#     ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho0001-rho0100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*(a2*rho0001-rho0100*a2.dag())))
    rho0101 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    # 25e with corrected sign (see 24f)
#     ww = np.linalg.inv(lv-(kappa_d+1j*(wd1+wd2))) * operator_to_vector(1.j*a*(rho0100+rho1000))
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d+1j*(wd1+wd2))).data, operator_to_vector(1.j*a1*rho0100+1.j*a2*rho1000))
    rho1100 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0011 = rho1100.trans().conj()

    # 25f
#     ww = np.linalg.inv(lv-(kappa_d+1j*(wd1-wd2))) * operator_to_vector(1.j*(a*rho0001-rho1000*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d+1j*(wd1-wd2))).data, operator_to_vector(1.j*(a1*rho0001-rho1000*a2.dag())))
    rho1001 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0110 = rho1001.trans().conj()

    # 25g
#     ww = np.linalg.inv(lv-(1.5*kappa_d+1j*wd1)) * operator_to_vector(1.j*(a*rho0101+a*rho1001-rho1100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(1.5*kappa_d+1j*wd1)).data, operator_to_vector(1.j*(a1*rho0101+a2*rho1001-rho1100*a2.dag())))
    rho1101 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0111 = rho1101.trans().conj()

    # 25h
#     ww = np.linalg.inv(lv-(1.5*kappa_d+1j*wd2)) * operator_to_vector(1.j*(a*rho0110+a*rho1010-rho1100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(1.5*kappa_d+1j*wd2)).data, operator_to_vector(1.j*(a1*rho0110+a2*rho1010-rho1100*a1.dag())))
    rho1110 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho1011 = rho1110.trans().conj()

    # 25i
#     ww = np.linalg.inv(lv-(2.*kappa_d)) * operator_to_vector(1.j*(a*rho0111+a*rho1011-rho1101*a.dag()-rho1110*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(2.*kappa_d)).data, operator_to_vector(1.j*(a1*rho0111+a2*rho1011-rho1101*a1.dag()-rho1110*a2.dag())))
    rho1111 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    return (kappa_d/(2.*np.pi))**2*rho1111.tr().real;


# def calc_correlation(rho0000, lv, kappa_d, wd1, wd2, a, Nall):    
def calc_G_Gamma12_alpha(rho0000, lv, kappa_d, wd1, wd2, a, Nall, alpha):    
#     rho0000 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)

    # 25a
#     ww = np.linalg.inv(lv-(kappa_d/2.+1j*wd1)) * operator_to_vector(1.j*a*rho0000)
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd1)).data, operator_to_vector(1.j*(a+alpha)*rho0000))
    rho1000 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0010 = rho1000.trans().conj()

    # 25b
#     ww = np.linalg.inv(lv-(kappa_d/2.+1j*wd2)) * operator_to_vector(1.j*a*rho0000)
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d/2.+1j*wd2)).data, operator_to_vector(1.j*(a+alpha)*rho0000))
    rho0100 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0001 = rho0100.trans().conj()

    # 25c
#     ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho0010-rho1000*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*((a+alpha)*rho0010-rho1000*(a.dag()+np.conj(alpha)))))
    rho1010 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    # 25d
#     ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho0001-rho0100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*((a+alpha)*rho0001-rho0100*(a.dag()+np.conj(alpha)))))
    rho0101 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    # 25e
#     ww = np.linalg.inv(lv-(kappa_d+1j*(wd1+wd2))) * operator_to_vector(1.j*a*(rho0100+rho1000))
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d+1j*(wd1+wd2))).data, operator_to_vector(1.j*(a+alpha)*(rho0100+rho1000)))
    rho1100 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0011 = rho1100.trans().conj()

    # 25f
#     ww = np.linalg.inv(lv-(kappa_d+1j*(wd1-wd2))) * operator_to_vector(1.j*(a*rho0001-rho1000*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(kappa_d+1j*(wd1-wd2))).data, operator_to_vector(1.j*((a+alpha)*rho0001-rho1000*(a.dag()+np.conj(alpha)))))
    rho1001 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0110 = rho1001.trans().conj()

    # 25g
#     ww = np.linalg.inv(lv-(1.5*kappa_d+1j*wd1)) * operator_to_vector(1.j*(a*rho0101+a*rho1001-rho1100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(1.5*kappa_d+1j*wd1)).data, operator_to_vector(1.j*((a+alpha)*rho0101+(a+alpha)*rho1001-rho1100*(a.dag()+np.conj(alpha)))))
    rho1101 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho0111 = rho1101.trans().conj()

    # 25h
#     ww = np.linalg.inv(lv-(1.5*kappa_d+1j*wd2)) * operator_to_vector(1.j*(a*rho0110+a*rho1010-rho1100*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(1.5*kappa_d+1j*wd2)).data, operator_to_vector(1.j*((a+alpha)*rho0110+(a+alpha)*rho1010-rho1100*(a.dag()+np.conj(alpha)))))
    rho1110 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()
    rho1011 = rho1110.trans().conj()

    # 25i
#     ww = np.linalg.inv(lv-(2.*kappa_d)) * operator_to_vector(1.j*(a*rho0111+a*rho1011-rho1101*a.dag()-rho1110*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-(2.*kappa_d)).data, operator_to_vector(1.j*((a+alpha)*rho0111+(a+alpha)*rho1011-rho1101*(a.dag()+np.conj(alpha))-rho1110*(a.dag()+np.conj(alpha)))))
    rho1111 = Qobj(ww.reshape((Nall,Nall)),dims=rho0000.dims).trans()

    return (kappa_d/(2.*np.pi))**2*rho1111.tr().real;


def calc_g_Gamma12(rho0000, lv, kappa_d, wd1, wd2, a, Nall):
    return calc_G_Gamma12(rho0000, lv, kappa_d, wd1, wd2, a, Nall) \
    /calc_S_Gamma(rho0000, lv, kappa_d, wd1, a, Nall)/calc_S_Gamma(rho0000, lv, kappa_d, wd2, a, Nall)


def calc_g_Gamma12_twooperators(rho0000, lv, kappa_d, wd1, wd2, a1, a2, Nall):
    return calc_G_Gamma12_twooperators(rho0000, lv, kappa_d, wd1, wd2, a1, a2, Nall) \
    /calc_S_Gamma(rho0000, lv, kappa_d, wd1, a1, Nall)/calc_S_Gamma(rho0000, lv, kappa_d, wd2, a2, Nall)


def calc_g_Gamma12_alpha(rho0000, lv, kappa_d, wd1, wd2, a, Nall, alpha):
    return calc_G_Gamma12_alpha(rho0000, lv, kappa_d, wd1, wd2, a, Nall, alpha) \
    /calc_S_Gamma_alpha(rho0000, lv, kappa_d, wd1, a, Nall, alpha)/calc_S_Gamma_alpha(rho0000, lv, kappa_d, wd2, a, Nall, alpha)

# ## frequency-resolved correlation ($\tau\neq 0$)
# \begin{equation}
#     G_{\Gamma,\Gamma}^{(2)}(\omega_1,\omega_2; \tau) = \left(\frac{\Gamma}{2\pi}\right)^2\langle a_1^\dagger(0) a_2^\dagger(\tau) a_2(\tau) a_1(0) \rangle
# \end{equation}
# then (in the steady state):
# \begin{equation}
#     g_{\Gamma,\Gamma}^{(2)}(\omega_1,\omega_2; \tau) = \frac{G_{\Gamma,\Gamma}^{(2)}(\omega_1,\omega_2; \tau)}{S_\Gamma(\omega_1)S_\Gamma(\omega_2)}
# \end{equation}

# In[ ]:


# def correlation_ss_gtt(H, tlist, c_ops, a_op, b_op, c_op, d_op, rho0=None):
def calc_G_Gamma12_tau(H, tlist, c_ops, a_op, b_op, c_op, d_op, rho0=None):
    if rho0 == None:
        rho0 = steadystate(H, c_ops)
    return mesolve(H, d_op * rho0 * a_op, tlist, c_ops, [b_op * c_op]).expect[0]

# call as: calc_G_Gamma12_tau(H, taulist, c_ops, d1.dag(), d2.dag(), d2, d1, rho0=rho_ss)


# # CSI violation
# \begin{equation}\label{CSI}
#     R_\Gamma(\omega_1,\omega_2) = \frac{\left[g_{\Gamma}^{(2)}(\omega_1,\omega_2)\right]^{2}}{g_{\Gamma,11}^{(2)}(\omega_1,\omega_1) g_{\Gamma,22}^{(2)}(\omega_2,\omega_2)}
# \end{equation}

# In[ ]:


def calc_CSI(rho00, lv, kappa_d, wd1, wd2, a, Nall):    
#     rho00 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)
    dd1 = calc_g11(rho00, lv, kappa_d, wd1, a, Nall)
    dd2 = calc_g11(rho00, lv, kappa_d, wd2, a, Nall)
    cr = calc_g_Gamma12(rho00, lv, kappa_d, wd1, wd2, a, Nall)**2/(dd1*dd2)
    return cr;


# ## autocorrelation
# \begin{equation}
# g_{\Gamma,11}^{(2)}(\omega_1,\omega_1) = \frac{\langle (a_1^\dagger)^2 a_1^2 \rangle}{\langle a_1^\dagger a_1 \rangle^2} = \frac{G_{\Gamma,11}^{(2)}(\omega_1,\omega_1)}{S_\Gamma(\omega_1)^2}
# \end{equation}

# In[ ]:


def calc_g11(rho00, lv, kappa_d, wd1, a, Nall):    
    return calc_G_Gamma11(rho00, lv, kappa_d, wd1, a, Nall)/(calc_S_Gamma(rho00, lv, kappa_d, wd1, a, Nall)**2);

# def calc_g_Gamma11(rho00, lv, kappa_d, wd1, a, Nall):    
#     return calc_G_Gamma11(rho00, lv, kappa_d, wd1, a, Nall)/(calc_S_Gamma(rho00, lv, kappa_d, wd1, a, Nall)**2);


# \begin{equation}
# G_{\Gamma,11}^{(2)}(\omega_1,\omega_1)=\left(\frac{\Gamma}{2\pi}\right)^2\langle (a_1^\dagger)^2 a_1^2 \rangle
# \end{equation}

# In[ ]:


# def calc_S11(rho00, lv, kappa_d, wd1, a, Nall):    
def calc_G_Gamma11(rho00, lv, kappa_d, wd1, a, Nall):    
#     rho00 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)

#     ww = np.linalg.inv(lv-.5*kappa_d-1j*wd1) * operator_to_vector(1.j*a*rho00)
    ww = scipy.sparse.linalg.spsolve((lv-.5*kappa_d-1j*wd1).data, operator_to_vector(1.j*a*rho00))
    rho10 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho01 = rho10.trans().conj()

#     ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho01-rho10*a.dag()))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d).data, operator_to_vector(1.j*(a*rho01-rho10*a.dag())))
    rho11 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()

#     ww = np.linalg.inv(lv-kappa_d+2j*wd1) * operator_to_vector(-1.j*rho01*a.dag()*np.sqrt(2))
    ww = scipy.sparse.linalg.spsolve((lv-kappa_d+2j*wd1).data, operator_to_vector(-1.j*rho01*a.dag()*np.sqrt(2)))
    rho02 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho20 = rho02.trans().conj()
    
#     ww = np.linalg.inv(lv-1.5*kappa_d+1j*wd1) * operator_to_vector(1.j*(-rho11*a.dag()*np.sqrt(2)+a*rho02))
    ww = scipy.sparse.linalg.spsolve((lv-1.5*kappa_d+1j*wd1).data, operator_to_vector(1.j*(-rho11*a.dag()*np.sqrt(2)+a*rho02)))
    rho12 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho21 = rho12.trans().conj()

#     ww = np.linalg.inv(lv-2*kappa_d) * operator_to_vector(1.j*np.sqrt(2)*(-rho21*a.dag()+a*rho12))
    ww = scipy.sparse.linalg.spsolve((lv-2*kappa_d), operator_to_vector(1.j*np.sqrt(2)*(-rho21*a.dag()+a*rho12)))
    rho22 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
# this calculates the population of the 2nd excited state; multiply by 2 to get number of photons
#     return 2*rho22.tr().real;
    return (kappa_d/(2.*np.pi))**2*2*rho22.tr().real;


# In[ ]:


# def calc_S11_depr(rho00, lv, kappa_d, wd1, a, Nall):    
def calc_S11_depr(rho00, lv, kappa_d, wd1, a, Nall):    
#     rho00 = steadystate(H, c_ops, method = 'direct')
#     lv = liouvillian(H, c_ops)

    ww = np.linalg.inv(lv-.5*kappa_d-1j*wd1) * operator_to_vector(1.j*a*rho00)
    rho10 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho01 = rho10.trans().conj()
    ww = np.linalg.inv(lv-kappa_d) * operator_to_vector(1.j*(a*rho01-rho10*a.dag()))
    rho11 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    ww = np.linalg.inv(lv-kappa_d+2j*wd1) * operator_to_vector(-1.j*rho01*a.dag()*np.sqrt(2))
    rho02 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho20 = rho02.trans().conj()
    ww = np.linalg.inv(lv-1.5*kappa_d+1j*wd1) * operator_to_vector(1.j*(-rho11*a.dag()*np.sqrt(2)+a*rho02))
    rho12 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
    rho21 = rho12.trans().conj()
    ww = np.linalg.inv(lv-2*kappa_d) * operator_to_vector(1.j*np.sqrt(2)*(-rho21*a.dag()+a*rho12))
    rho22 = Qobj(ww.reshape((Nall,Nall)),dims=rho00.dims).trans()
# this calculates the population of the 2nd excited state; multiply by 2 to get number of photons
#     return 2*rho22.tr().real;
    return (kappa_d/(2.*np.pi))**2*2*rho22.tr().real;

