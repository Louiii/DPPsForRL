import sys
sys.path.append('../')
from DPP import DPP
from scipy.linalg import eigh
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt

n, sigma = 20, 0.2
N = n*n
x, y = np.meshgrid(np.linspace(1/n, 1, n), np.linspace(1/n, 1, n))
xf, yf = x.flatten('F'), y.flatten('F')
baseY = list(range(N))

stack_xf, stack_yf = np.array([xf,]*N), np.array([yf,]*N)
M = np.square( (stack_xf.T - stack_xf) ) + np.square( (stack_yf.T - stack_yf) )
K = np.exp(-M/sigma**2)


f = lambda i: np.array([ xf[i], yf[i] ])
q = lambda i, theta: np.dot( f(i), theta )

def L(theta):
	qs = np.array([ q(i, theta) for i in baseY])
	return np.multiply( K, np.outer( qs, qs ) )

theta_true, T = np.array([4.1, 0.76]), 200
dpp = DPP(*eigh(L(theta_true)))
print('...generating samples')
samples = [dpp.sample() for _ in tqdm(range(T))]

def L_At(At, theta):# L(At|theta)
	qs = np.array([ q(i, theta) for i in At])
	return np.multiply( K[np.ix_(At, At)], np.outer( qs, qs ) )

def Likelihood(A, theta):# log likelihood function for a standard DPP (not k-DPP!)
	T = len(A)
	term1 = -T*np.log(np.linalg.det( L(theta) + np.eye(N) ))
	term2 = np.sum([np.log(np.linalg.det( L_At(At, theta) )) for At in A])
	return term1 + term2

# learn theta which maximises the likelihood function.
print('...learning theta from data')
neg_likelihood = lambda theta: -Likelihood(samples, theta)
theta_estimate = minimize(neg_likelihood, np.array([2, 2]), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(theta_estimate.x)
