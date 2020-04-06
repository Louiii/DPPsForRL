from MLE_QualityModel import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
import itertools, collections


def plot_sample(filename, sample):
    plt.scatter(xf[sample],yf[sample],c='r',s=3)
    # plt.title('An example DPP sample with $\\theta$ = '+str(list(theta_true))+'$^\\top$')
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.xticks([])
    plt.yticks([])

    plt.savefig('plots/'+filename, dpi=600)
    plt.show()

def likelihood_heatmap(th1, th2, lkhd, filename, angle=0, vectors=None, adam=False):
    ax = plt.axes(projection='3d')
    ax.view_init(30, angle)

    if vectors is not None:
        x, y, z, dx, dy, dz = [np.array(v) for v in vectors]
        z /= 10000
        dz /= 10000
        lengths = np.array([min(1, (20/i)**2) for i in range(1, len(x)+1)])
        c = lengths/np.array(dx**2 + dy**2 + dz**2)
        dx, dy, dz = c*dx, c*dy, c*dz
        q = ax.quiver(x, y, z, dx, dy, dz, length=0.5, cmap='Reds', lw=1)
        if adam:
            lengths = np.ones(len(x))
            lengths[-1] = 0
        q.set_array(lengths)

        lkhd /= 10000

    xv, yv = np.meshgrid(th1, th2)
    ax.plot_surface(xv.T, yv.T, lkhd, cmap='viridis', alpha=0.4, edgecolor='none')

    ax.set_zlabel('Likelihood (scaled)')
    ax.set_xlabel('$\\theta_{1}$')
    ax.set_ylabel('$\\theta_{2}$')
    ax.plot([theta_true[0]], [theta_true[1]], [np.max(lkhd)], 'r.')
    plt.savefig('plots/'+filename, dpi=600)
    plt.show()
    plt.clf()



def plot_points_heatmap(Ys, filename, angle1=30, angle2=0):
    # keys are the elem of Y, values are their frequency
    counter = collections.Counter(itertools.chain(*Ys))
    colours = np.zeros(N)
    for i, count in counter.items():
        colours[i] = count
    colours /= np.max(colours)

    ax = plt.axes(projection='3d')
    ax.view_init(angle1, angle2)
    
    ax.plot_surface(x, y, colours.reshape(n, n).T, cmap='PuRd', edgecolor='none')
    # ax.set_title('Histogram of sampled points')
    ax.set_zlabel('frequency each point was sampled')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plots/'+filename, dpi=600)
    plt.show()
    
print('making likelihood meshgrid...')
n1, n2 = 20, 20
th1, th2 = np.linspace(0.5, 10, n1), np.linspace(0.2, 10, n2)

likelihood = np.zeros((n1, n2))
for i in range(n1):
	for j in range(n2):
		theta = np.array([th1[i], th2[j]])
		likelihood[i][j] = logLikelihood(samples, theta)
mn = np.min(likelihood)
likelihood -= mn


from GradientAscent import *

values = []
for (th, dth) in grad:
   f1, f2 = logLikelihood(samples, th), logLikelihood(samples, th+dth)
   values.append((th[0], th[1], f1-mn, dth[0], dth[1], f2-f1))
vectors = tuple(zip(*values))
# likelihood_heatmap(th1, th2, likelihood, 'likelihood_theta', angle=150, vectors=vectors)
likelihood_heatmap(th1, th2, likelihood, 'likelihood_theta', angle=150, vectors=vectors, adam=True)


plot_sample('sample', samples[0])

plot_points_heatmap(samples, 'histogram_heatmap_view1', angle2=-120)
plot_points_heatmap(samples, 'histogram_heatmap_view2', angle1=-10, angle2=-130)
plot_points_heatmap(samples, 'histogram_heatmap_view3', angle1=-4, angle2=-87)