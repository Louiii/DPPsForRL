import matplotlib.pyplot as plt
import imageio
import os
import shutil
import numpy as np

def makeGIF(filename, prefix, plotroot='temp', interval=0.3, end_pause=12):
    # Set the directory you want to start from
    rootDir = './'+plotroot
    files={}
    for dirName, subdirList, fileList in os.walk(rootDir):
        print(fileList)
        for f in fileList:
            if f[0]==prefix:
                iteration = f[1:-4]
                files[int(iteration)] = f
    keys = list(files.keys())
    keys.sort()

    print("Keys: "+str(keys))
    
    for k in keys:
        path = rootDir+'/'+files[k]
        im = imageio.imread(path)
        (h, w, _) = im.shape

    images = []
    for k in keys:
        images.append(imageio.imread(rootDir+'/'+files[k]))
    for _ in range(end_pause):
        images.append(imageio.imread(rootDir+'/'+files[keys[-1]]))
    kargs = { 'duration': interval }
    imageio.mimsave('gifs/'+filename+'.gif', images, **kargs)

    # delete all of the png files
    shutil.rmtree(rootDir)
    os.mkdir(plotroot)

def plot_individual(filename, data, xs, ys, n, dpi=400):
    lims = [1/n - 0.01, 1.01]
    fig, ax = plt.subplots(figsize=(10, 5))

    zs = data['z']
    zs = zs.reshape(xs.shape).T
    zs = np.sqrt(np.log(zs+1))# the probs will be very small- this just transforms by a monotone fn
    ax.pcolormesh(xs, ys, zs, shading='gouraud', cmap=plt.cm.Greys_r)
    ax.plot(data['X'],data['Y'], 'r.', ms=5)
    # ax.title.set_text(data['title'])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_size_inches(3, 3)
    plt.tight_layout()
    plt.savefig('temp/'+filename, dpi=dpi)

def plot_density_8axes(filename, data, xs, ys, n):
    lims = [1/n - 0.01, 1.01]
    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(10, 5))
    k=0
    for i in range(2):
        for j in range(4):
            k+=1
            zs = data[k]['z']
            # z = np.zeros(xs.shape)
            # n, m = xs.shape
            # for ii in range(n):
            #     for jj in range(m): z[ii, jj] = zs[ii*m+jj]
            zs = zs.reshape(xs.shape).T
            zs = np.sqrt(np.log(zs+1))# the probs will be very small- this just transforms by a monotone fn
            axes[i,j].pcolormesh(xs, ys, zs, shading='gouraud', cmap=plt.cm.Greys_r)
            axes[i,j].plot(data[k]['X'],data[k]['Y'], 'r.', ms=5)
            # axes[i,j].title.set_text(data[k]['title'])
            axes[i,j].set_xlim(lims)
            axes[i,j].set_ylim(lims)
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])

    fig.set_size_inches(6.7, 3)
    plt.tight_layout()
    plt.savefig('plots/'+filename, dpi=600)
    plt.show()