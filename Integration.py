# coding: utf-8

# In[24]:


import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate



def f(x):
    return np.cos(x)

# trapezoidal rule
def trapz(f, a, b, n):
    # n evenly spaced
    g = 0
    if b > a:
        h = (b-a)/float(n)
    else:
        h = (a-b)/float(n)
    for i in range (0, n):
        k = 0.5 * h * ( f(a + i*h) + f(a + (i+1)*h) )
        g = g + k
    return g

# Gaussian quadrature
def gauss(f,a,b):
    return integrate.quad(f,a,b)


#Monte Carlo integration
def MC(f, a, b, n):
    x = np.random.uniform(a, b, n)
    fvals= f(x)
    #print(fvals)
    # estimate
    I = ((b-a)/n)*np.sum(fvals)
    return I


# main function
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s -N [number of intervals for integration] -a [lower bound] -b [upper bound]" % sys.argv[0])
        print
        sys.exit(1)
    
    # define maximum interval (default)
    a = -1
    b = 1
    N = 1000

    if '-N' in sys.argv:
        p = sys.argv.index('-N')
        N = int(sys.argv[p + 1])
        if N > 0:
            Nmax = N
               
    if '-a' in sys.argv:
        p = sys.argv.index('-a')
        a0 = int(sys.argv[p+1])
        if a0 > -100:
            a = a0
    
    if '-b' in sys.argv:
        p = sys.argv.index('-b')
        b0 = int(sys.argv[p+1])
        if b0 > a:
            b = b0
    
    
    # do the integration for every n and plot
    tz = []
    tz_err = []
    gs = []
    gs_err = []
    mc =[]
    mc_err = []
    true = []
    for n in range(1,N):
        x = np.linspace(a,b,n)
    
        # true value of the integral
        true_v = f(b) - f(a)
        
        # integration using trapezoidal rule and calculate the error in the trapez approximation
        tz_v = trapz(f, a, b, n)
        t_err = tz_v - true_v
        
        # integration using Gaussian quadrature
        gs_v, g_err = gauss(f, a, b)
        
        
        # MC integration and calculate the mc_error in the Monte carlo approximation
        mc_v = MC(f, a, b, n)
        m_err = mc_v - true_v
        
        
        #append into list for plotting
        true.append(true_v)
        tz.append(tz_v)
        gs.append(gs_v)
        mc.append(mc_v)
        mc_err.append(m_err)
        gs_err.append(g_err)
        tz_err.append(t_err)
        
    tz = np.array(tz)
    gs = np.array(gs)
    mc = np.array(mc)
    
    gs_err = np.array(gs_err)
    tz_err = np.array(tz_err)
    mc_err = np.array(mc_err)
    
    xn = np.arange(1,n+1)
    plt.figure(figsize=[12,7])

    plt.plot(xn, tz_err,  label='Using Trapezoidal rule Method ')
    plt.plot(xn, gs_err,  label='Using Gaussian quadrature Method ')
    plt.plot(xn, mc_err,  alpha=0.4, label='Using Monte Carlo Method ')    
    plt.fill_between(xn, mc_err, mc_err, color='red',alpha=0.5,label=r'Monte Carlo Error Spread')

    plt.ylabel("(True value - numerical estimation)")
    plt.xlabel("(N)")
    plt.legend()
    plt.grid()
    plt.savefig("DiffvsN_integration.pdf")

    plt.show()
    
