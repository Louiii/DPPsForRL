import sys
sys.path.append('../../')
from DPP import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json

def my_3D_fig():
    vectors = [np.array([i,j,k])/(i**2 + j**2 + k**2)**0.5 for k in [-1,0,1] for j in [-1,0,1] for i in [-1,0,1] if [i,j,k]!=[0,0,0]]
    qs = np.ones(len(vectors))
    B  = np.array([q*v for v, q in zip(vectors, qs)]).T
    C  = np.dot(B, B.T)

    dpp = DualDPP(C, B)
    Y = dpp.sample_dual()
    [x, y, z] = zip(*np.array(vectors)[list(Y)])

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("plots/MySimpleCube_Example", dpi=100)
    plt.show()

def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize: rnd = np.random.random() * samples

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - np.power(y,2))

        phi = ((i + rnd) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x,y,z])
    return points

def my_sphere_fig():
    points  = fibonacci_sphere(100)

    vectors = [np.array(p) for p in points]
    qs = np.ones(len(vectors))
    B  = np.array([q*v for v, q in zip(vectors, qs)]).T
    C  = np.dot(B, B.T)

    dpp = DualDPP(C, B)
    Y = dpp.sample_dual()
    [x, y, z] = zip(*np.array(vectors)[list(Y)])

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', s=75, marker='.')

    [x, y, z] = zip(*[points[i] for i in range(len(points)) if i not in Y])

    ax.scatter(x, y, z, c='b', marker='.', alpha=0.2)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_title("Dual DPP sampling points\non the surface of a sphere, D=3")
    plt.savefig("plots/My3D_Sphere_Example", dpi=400)
    plt.show()

def my_bar_example(histogram=None, repeats=1000):

    def R3_to_R4(x, y, z):
        vx, vy = np.cos(x), np.cos(y)
        v = [vx*vy*np.cos(z), vx*vy*np.sin(z), vx*np.sin(y), np.sin(x)]
        return np.array(v)

    def R4_to_R3(a, b, c, d):
        x = np.arcsin(d)
        vx = np.cos(x)
        y = np.arcsin(c/vx)
        vy = np.cos(y)
        z = np.arcsin(b/(vx*vy))
        return [x, y, z]

    xs, ys, zs = np.pi*np.arange(0,12,1)/30, np.pi*np.arange(0,12,1)/30, np.pi*np.arange(0,12,1)/30
    points  = [(x, y, z) for x in xs for y in ys for z in zs]
    vectors = [R3_to_R4(x, y, z) for (x, y, z) in points]

    qs = np.ones(len(vectors))
    B  = np.array([q*v for v, q in zip(vectors, qs)]).T
    C  = np.dot(B, B.T)

    if histogram == None:
        dpp = DualDPP(C, B)
        Y = dpp.sample_dual(k=4)
        abcd = np.array(vectors)[list(Y)]
        x, y, z = zip(*[R4_to_R3(a, b, c, d) for (a,b,c,d) in abcd])

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, c='r', s=75, marker='.')

        [x, y, z] = zip(*[points[i] for i in range(len(points)) if i not in Y])

        ax.scatter(x, y, z, c='b', marker='.', alpha=0.2)

        ax.set_xlabel('$\psi$')
        ax.set_ylabel('$\\theta$')
        ax.set_zlabel('$\phi$')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title("Dual DPP sampling points, $x_i \in \mathbb{R}^3$, with\ndiversity vectors, $\phi_i \in \mathbb{R}^4$, such that $\phi_i .\phi_j \propto |x_i - x_j|$.")
        plt.savefig("plots/My3D_Bar_Example", dpi=400)
        plt.show()
    else:
        import pandas as pd
        import seaborn as sns

        if histogram=="simulate":
            from tqdm import tqdm

            def distance(v, u):
                u, v = np.array(u), np.array(v)
                return np.linalg.norm(u-v)

            def av_dist(Y):
                abcd = np.array(vectors)[list(Y)]
                x, y, z = zip(*[R4_to_R3(a, b, c, d) for (a,b,c,d) in abcd])
                dists = [distance([x[i],y[i],z[i]], [x[j],y[j],z[j]]) for i in range(len(Y)) for j in range(i+1, len(Y))]
                return np.mean(dists)
            
            def volumeR3(Y):
                abcd = np.array(vectors)[list(Y)]
                Vs = np.array([R4_to_R3(a, b, c, d) for (a,b,c,d) in abcd])
                v, Vs = Vs[0, :], Vs[1:, :]
                Vs -= v
                return np.abs(np.linalg.det(Vs))
            
            def volumeR4(Y):
                abcd = np.array(vectors)[list(Y)]
                return np.abs(np.linalg.det(abcd))
            
            print('sampling DPP...')
            dpp = DualDPP(C, B)
            dppYs, indYs = [], []
            for _ in tqdm(range(repeats)):
                dppYs.append( dpp.sample_dual(k=4) )
                indYs.append( np.random.randint(0, len(vectors), 4) )
            
            print('computing R3 distances...')
            dpp_dists = [av_dist(Y) for Y in dppYs]
            ind_dists = [av_dist(Y) for Y in indYs]
            print('computing R3 volumes...')
            dpp_volsR3 = [volumeR3(Y) for Y in dppYs]
            ind_volsR3 = [volumeR3(Y) for Y in indYs]
            print('computing R4 volumes...')
            dpp_volsR4 = [volumeR4(Y) for Y in dppYs]
            ind_volsR4 = [volumeR4(Y) for Y in indYs]
            print('exporting datasets.')
            export_dataset({'dpp': dpp_dists, 'ind':ind_dists}, "hist_dist_data")
            export_dataset({'dpp': dpp_volsR3, 'ind':ind_volsR3}, "hist_R3_vol_data")
            export_dataset({'dpp': dpp_volsR4, 'ind':ind_volsR4}, "hist_R4_vol_data")
        print('plotting.')
        fnames = ["hist_dist_data", "hist_R3_vol_data", "hist_R4_vol_data"]
        titles = ['k-DPP (k=4) vs Independent samples,\naverage distances between 3D points',
                  'k-DPP (k=4) vs Independent samples,\nvolume spanned by 3D points',
                  'k-DPP (k=4) vs Independent samples,\nvolume spanned by 4D diversity vectors']
        labels = ['dists.', 'vols.', 'vols.']
        wnames = ['R3_dists', 'R3_vols', 'R4_vols']
        limits = [[0.3, 1.4], [-0.1, 1.5], [-0.05, 0.6]]
        for fname, title, l, wname, lms in zip(fnames, titles, labels, wnames, limits):
            df = pd.DataFrame(data=load_dataset(fname))
    
            plt.figure()
            sns.distplot(df['dpp'], bins=100, color="dodgerblue", label="DPP "+l, hist_kws={'alpha':.7}, kde_kws={'linewidth':1.5})
            sns.distplot(df['ind'], bins=100, color="g", label="Indep. "+l, hist_kws={'alpha':.7}, kde_kws={'linewidth':1.5})
    
            plt.title(title, fontsize=16)
            plt.legend()
            plt.xlim(lms)
            plt.savefig('plots/'+wname+'_histogram', dpi= 400)
            plt.show()

def export_dataset(data, name):
    with open('datasets/'+name+'.json', 'w') as outfile:
        json.dump(data, outfile)

def load_dataset(name):
    json_file = open('datasets/'+name+'.json')
    json_str = json_file.read()
    return json.loads(json_str)


if __name__=="__main__":
    my_3D_fig()
    my_sphere_fig()
    my_bar_example()
    my_bar_example(histogram="simulate", repeats=50000)
    my_bar_example(histogram="", repeats=10000)



