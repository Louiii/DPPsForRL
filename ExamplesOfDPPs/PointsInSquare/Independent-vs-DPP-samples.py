import sys
sys.path.append('../../')
from DPP import *
import matplotlib.pyplot as plt

def plot(filename, axdata):
    lims = [0, 1.02]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i, ax in enumerate([ax1, ax2]):
        ax.scatter(axdata[i][1][0],axdata[i][1][1], c=axdata[i][2], s=2)
        ax.title.set_text(axdata[i][0])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.set_size_inches(6.7, 3)
    plt.savefig(filename, dpi=600)
    plt.show()

def makeDV(n, sigma = 0.1):# kernel width
    N = n*n
    # choose a grid of points
    x, y = np.meshgrid(np.linspace(1/n, 1, n), np.linspace(1/n, 1, n))

    # gaussian kernel, negatively correlate points close together in space
    xf, yf = x.flatten('F'), y.flatten('F')
    stack_xf, stack_yf = np.array([xf,]*N), np.array([yf,]*N)

    M = np.square( (stack_xf.T - stack_xf) ) + np.square( (stack_yf.T - stack_yf) )
    M = np.exp(-M/sigma**2)

    # decompose kernel
    D, V = eigh(M)
    return D, V, xf, yf

def figure():
    n, sigma = 60, 0.1
    D, V, xf, yf = makeDV(n, sigma)
    dpp = DPP(D, V)

    # sample
    dpp_sample = dpp.sample()#k=200)
    ind_sample = np.random.choice(range(n*n), len(dpp_sample), replace=False)#np.random.randint(n*n, size=len(dpp_sample))

    # PLOTTING
    plot("plots/DPP-n-"+str(n)+".png", [("DPP", (xf[dpp_sample], yf[dpp_sample]), "r"),
                          ("Independent", (xf[ind_sample], yf[ind_sample]), "#72246C")])
if __name__=="__main__":
    figure()
