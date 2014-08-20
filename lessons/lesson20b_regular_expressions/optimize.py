from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

### LINEAR OPTIMIZATIONS ###

# FINDING LOCAL MINIMUM #

def f(x):
	return x**2 + 10*np.sin(x)

x = np.arange(-10, 10, 0.1)
plt.plot(x, f(x))
plt.show()

# Setting different starting points #
optimize.fmin_bfgs(f, 3)

optimize.fmin_bfgs(f, 0)

# FINDING GLOBAL MINIMUM #

grid = (-10, 10, 0.1)
xmin_global = optimize.brute(f, (grid,))
xmin_global

xmin_local = optimize.fminbound(f, 0, 10)    
xmin_local

# FINDING THE ROOT, AKA WHERE f(x) = 0 #

root = optimize.fsolve(f, 1)  # our initial guess is 1
root

plt.plot(x, f(x))
plt.show()

root2 = optimize.fsolve(f, -2.5)
root2

### CURVE FITTING TO DATA ###

# USING A TEST FUNCTION #

from scipy.optimize import curve_fit

def fitFunc(t, a, b, c):
    return a*np.exp(-b*t) + c

t = np.linspace(0,4,50)
temp = fitFunc(t, 5.0, 1.5, 0.5)
noisy = temp + 0.3*np.random.normal(size=len(temp))
fitParams, fitCovariances = curve_fit(fitFunc, t, noisy)
print ' fit coefficients:\n', fitParams
print ' Covariance matrix:\n', fitCovariances

rcParams['figure.figsize'] = 10, 6 
plt.ylabel('Temperature (C)', fontsize = 16)
plt.xlabel('time (s)', fontsize = 16)
plt.xlim(0,4.1)
# plot the data as red circles with vertical errorbars
plt.errorbar(t, noisy, fmt = 'ro', yerr = 0.2)
# now plot the best fit curve and also +- 1 sigma curves #
sigma = [np.sqrt(fitCovariances[0,0]), \
         np.sqrt(fitCovariances[1,1]), \
         np.sqrt(fitCovariances[2,2]) ]
plt.plot(t, fitFunc(t, fitParams[0], fitParams[1], fitParams[2]),\
         t, fitFunc(t, fitParams[0] + sigma[0], fitParams[1] - sigma[1], fitParams[2] + sigma[2]),\
         t, fitFunc(t, fitParams[0] - sigma[0], fitParams[1] + sigma[1], fitParams[2] - sigma[2])\
         )
plt.show()


# ANOTHER EXAMPLE, THIS TIME WITH RANDOM DATA #
def f(t, omega, phi):
    return np.cos(omega * t + phi)

# Our x and y data
x = np.linspace(0, 3, 50)
y = 1.5*np.random.normal(size=50)

# Fit the model: the parameters omega and phi can be found in the params vector #
params, params_cov = optimize.curve_fit(f, x, y)

# plot the data and the fitted curve
t = np.linspace(0, 3, 1000)

pl.figure(1)
pl.clf()
pl.plot(x, y, 'bx')
pl.plot(t, f(t, *params), 'r-')
pl.show()


### CREATING A FUNCTION TO FIT DATA ###

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
# Setting degree = 3 #
z = np.polyfit(x, y, 3)
z

p = np.poly1d(z)
p(0.5)
p(3.5)
p(10)

# Higher order equations can get crazy #

p30 = np.poly1d(np.polyfit(x, y, 30))

p30(4)
p30(5)
p30(4.5)

xp = np.linspace(-2, 6, 100)
plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2,2)
plt.show()