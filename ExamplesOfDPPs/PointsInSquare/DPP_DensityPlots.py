import numpy as np
from scipy.linalg import sqrtm, inv, eigh
from tqdm import tqdm
from misc import *

class DPP:
    def __init__(self, D, V):
        self.D = D
        self.V = V
        # self.record_items = [0,1,2,4,6,8,10,12, 14]
        self.plot_data = []

    def sample(self, k=None, rec=None):
        N = self.D.shape[0]
        # PHASE 1
        if k is None:# general dpp
            self.D /= 1 + self.D
            self.V = self.V[:,np.random.rand(N) < self.D]
            k = self.V.shape[1]

        # PHASE 2
        Y = []
        for c in tqdm(range(k-1,-1,-1)):
            # choose vector index, with prob proportional to K_ii = v_i^T v_i, lambda==1??
            P   = np.sum(np.power(self.V, 2), axis=1)
            
            if rec is not None:
                if rec=="density_fig":
                    # record the PDF of the first few points
                    if c in list(range(k-1,k-9,-1)): 
                        self.plot_data.append(np.copy(P/np.sum(P)))
                        print(c)
                if rec=="all":
                    self.plot_data.append(np.copy(P/np.sum(P)))

            r = np.random.choice(range(N), p=P/np.sum(P))
            Y.append(r)

            # Select an eigenvector to remove
            vi = np.nonzero(self.V[r])[0][0]
            v = self.V[:,vi].copy()
            self.V = np.delete(self.V, vi, axis=1)
            # Update K to condition given we've seen r
            self.V -= np.outer(v, self.V[r]/v[r])

            # Orthogonalise, Gram-Schmidt
            if c > 0:
                for a in range(c):
                    for b in range(a):
                        self.V[:, a] -= np.dot(self.V[:, a], self.V[:, b])*self.V[:, b]
                    self.V[:, a] /= np.linalg.norm(self.V[:, a])

        return Y

def makeDV(n, sigma=0.5):
    N = n*n

    # choose a grid of points for our ground set:
    x, y = np.meshgrid(np.linspace(1/n, 1, n), np.linspace(1/n, 1, n))

    # gaussian kernel, negatively correlate points close together in space
    xf, yf = x.flatten('F'), y.flatten('F')
    stack_xf, stack_yf = np.array([xf,]*N), np.array([yf,]*N)
    M = np.square( (stack_xf.T - stack_xf) ) + np.square( (stack_yf.T - stack_yf) )
    M = np.exp(-M/sigma**2)

    # decompose kernel
    D, V = eigh(M)
    return D, V, xf, yf, x, y

def i_to_xy(i):
    return xf[i], yf[i]

def f(zs):
    points = []
    for i in range(N):
        z = zs[i]
        x, y = i_to_xy(i)
        points.append( (x,y,z) )
    return points

def density_changing_fig(n, sigma=0.5):
    D, V, xf, yf, x, y = makeDV(n, sigma)
    dpp = DPP(D, V)

    # sample
    dpp_sample = dpp.sample(rec="density_fig")

    data = {i+1:{'title':'Step '+str(i), 'X':xf[dpp_sample][:i], 'Y':yf[dpp_sample][:i], 'z':np.array(dpp.plot_data[i])} for i in range(8)}
    # PLOTTING
    plot_density_8axes("density_fig.png", data, x, y, n)

def sampling_gif(n, sigma=0.5):
    D, V, xf, yf, x, y = makeDV(n, sigma)
    dpp = DPP(D, V)
    dpp_sample = dpp.sample(rec="all")

    print(len(dpp.plot_data))
    data = {i+1:{'title':'Step '+str(i), 'X':xf[dpp_sample][:i], 'Y':yf[dpp_sample][:i], 'z':np.array(dpp.plot_data[i])} for i in range(len(dpp.plot_data))}
    
    prefix = "A"
    print("Plotting...")
    for k, v in data.items():
        plot_individual(prefix+str(k), v, x, y, n, dpi=200)
    print("Making gif...")
    makeGIF("sampling", prefix, 'temp')
    print("Done.")

if __name__=="__main__":
    n, sigma = 20, 0.5
    density_changing_fig(n, sigma)
    sampling_gif(n, sigma)
    
