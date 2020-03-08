#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:51:23 2019

@author: louisrobinson
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
from pylab import fill
import imageio
import os
import shutil

def makeGIF(plotroot, filename, interval=0.1, end_pause=12):
    # Set the directory you want to start from
    rootDir = './'+plotroot+'/'
    files={}
    for dirName, subdirList, fileList in os.walk(rootDir):
        for f in fileList:
            if f[0]=='V':
                iteration = f[1:-4]
                files[int(iteration)] = f
    keys = list(files.keys())
    keys.sort()
    
    
    for k in keys:
        path = rootDir+files[k]
        im = imageio.imread(path)
        (h, w, _) = im.shape


    images = []
    for k in keys:
        images.append(imageio.imread(rootDir+files[k]))
    for _ in range(end_pause):
        images.append(imageio.imread(rootDir+files[keys[-1]]))
    kargs = { 'duration': interval }
    imageio.mimsave(filename+'.gif', images, **kargs)


    # delete all of the png files
    shutil.rmtree(rootDir)
    os.mkdir(plotroot)

def plotMazePolicy(agents, bs, gx, gy, w, h, **kwargs):#show and save default to false. title, colorbar-label, filename
    fig, ax1 = plt.subplots()
    COLOUR_PALATINATE = "#72246C"

    # make grid
    minor_locator1 = AutoMinorLocator(2)
    minor_locator2 = FixedLocator([j for j in range(h)])
    plt.gca().xaxis.set_minor_locator(minor_locator1)
    plt.gca().yaxis.set_minor_locator(minor_locator2)
    for i in range(7): plt.plot([i,i], [0,3], lw=0.5, alpha=0.5, c='grey')
    for i in range(4): plt.plot([0,8], [i,i], lw=0.5, alpha=0.5, c='grey')

    if 'title' in kwargs and kwargs['title']!=None: 
        ax1.set_title(kwargs['title'])

    onGoal = False
    # Shade and label agents
    for i, (sx, sy) in zip([1,2,3], agents):
        if sy==3: onGoal=True
        ax1.text(sx+0.3, sy+0.4, '$A_'+str(i)+'$', color=COLOUR_PALATINATE, fontsize=20)
        fill([sx,sx+1,sx+1,sx], [sy,sy,sy+1,sy+1], COLOUR_PALATINATE, alpha=0.4, edgecolor=COLOUR_PALATINATE)

    # Shade and label goal cell
    if not onGoal:
        ax1.text(gx+0.18, gy+0.4, 'Goal', color="g", fontsize=15)
        fill([gx,gx+1,gx+1,gx], [gy,gy,gy+1,gy+1], 'g', alpha=0.3, edgecolor='g')

    # Make obstacles:
    # for (x, y) in obstacles: 
    #     fill([x,x+1,x+1,x], [y,y,y+1,y+1], 'k', alpha=0.2, edgecolor='k')
    # for (x, y) in obstacles: 
    b1 = [4,7,7,4]
    b2 = [0,3,3,0]
    if bs==0:
        b2 = [1,4,4,1]
    elif bs==6:
        b1 = [3,6,6,3]
    fill(b1, [3,3,4,4], 'k', alpha=0.55, edgecolor='k')
    fill(b2, [3,3,4,4], 'k', alpha=0.55, edgecolor='k')
        
    bs = [1,5] if bs==3 else [2,5] if bs==0 else [1,4]
    for i, x in zip([1, 2], bs): 
        ax1.text(x+0.3, 3.4, '$B_'+str(i)+'$', color="k", fontsize=20)

    # make grid lines on center
    plt.xticks([0.5+i for i in range(w)], [i for i in range(w)])
    plt.yticks([0.5+i for i in range(h)], [i for i in range(h)])
    plt.xlim(0,w)
    plt.ylim(0,h)

    ax1.yaxis.set_major_locator(plt.NullLocator())
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    plt.tight_layout()

    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'], dpi=kwargs['dpi'])
    plt.close(fig)

def maze_record(iteration, tt, agents, w, h, blocker_state, up=True, dpi=40):
    agents = [(j,i) for i, j in agents]
    gx, gy = blocker_state, 3
    fn = "plots/temp-plots/temp-plots1/V"+str(iteration)+".png"
    if up: fn = "../"+fn
    plotMazePolicy(agents, blocker_state, gx, gy, h, w, show=False, filename=fn, title=tt, cbarlbl="Value", dpi=dpi)


# blocker_state = 3
# agents = [(0,0), (0,2), (2,0)]
# gx, gy = blocker_state, 3
# fn = "TEST.png"
# plotMazePolicy(agents, blocker_state, gx, gy, 7, 4, show=True, filename=fn, title="Blocker Task", cbarlbl="Value", dpi=400)

#def plotQuality(V, ep, fname, found, dpi=200, show=True):
#    L = np.dot( V, V.T )
#    S = np.zeros( L.shape )
#    for i in range(L.shape[0]):
#        for j in range(L.shape[1]):
#            S[i, j] = L[i, j]/np.sqrt(L[i, i]*L[j, j])
#            
#    q = np.diag(L)[:21].reshape((3, 7))
#    plt.imshow(np.abs( q ), 'coolwarm')
#    plt.colorbar()
#    if found: 
#        plt.title('Quality, ep: '+str(ep)+', found goal')
#    else:
#        plt.title('Quality, episode: '+str(ep))
#        
#    plt.xticks(np.arange(7), ('1','2','3','4','5','6','7'))
#    plt.yticks(np.arange(3), ('1','2','3'))
#    
#    plt.savefig( fname, dpi=dpi )
#    if show: plt.show()
#    plt.close()
#
#def plotSimilarity( V, ep, fname, found, dpi=200, show=True, plotL=False ):
#    L = np.dot( V, V.T )
#    S = np.zeros( L.shape )
#    for i in range(L.shape[0]):
#        for j in range(L.shape[1]):
#            S[i, j] = L[i, j]/np.sqrt(L[i, i]*L[j, j])
#           
#    if plotL:
#        plt.imshow( L, 'coolwarm')
#    else:
#        plt.imshow( np.abs( S[:21, :21] ), 'coolwarm')
#    plt.colorbar()
#    
#    pt1 = 'L, ' if plotL else 'Similarity, '
#    if found: 
#        plt.title(pt1+'ep: '+str(ep)+', found goal')
#    else:
#        plt.title(pt1+'ep: '+str(ep))
#    
#    labels = [str((i, j)) for i in [1,2,3,4] for j in [1,2,3,4,5,6,7]]
#    if not plotL:
#        labels = labels[:-7]
#    plt.xticks(np.arange(len(labels)), tuple(labels))
#    plt.yticks(np.arange(len(labels)), tuple(labels))
#    plt.rc('axes', labelsize=8)
#    plt.xticks(rotation=270)
#    
#    plt.savefig(fname, dpi=dpi)
#    if show: plt.show()
#    plt.close()
    
def plotQuality(V, ep, fname, found, dpi=200, show=True):
    L = np.dot( V, V.T )
    S = np.zeros( L.shape )
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            S[i, j] = L[i, j]/np.sqrt(L[i, i]*L[j, j])
            
    q = np.diag(L).reshape((4, 7))
    plt.imshow(np.abs( q ), 'coolwarm')
    plt.colorbar()
    if found: 
        plt.title('Quality, ep: '+str(ep)+', found goal')
    else:
        plt.title('Quality, episode: '+str(ep))
        
    plt.xticks(np.arange(7), ('1','2','3','4','5','6','7'))
    plt.yticks(np.arange(4), ('1','2','3','4'))
    
    plt.savefig( fname, dpi=dpi )
    if show: plt.show()
    plt.close()

def plotSimilarity( V, ep, fname, found, dpi=200, show=True, plotL=False ):
    L = np.dot( V, V.T )
    S = np.zeros( L.shape )
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            S[i, j] = L[i, j]/np.sqrt(L[i, i]*L[j, j])
           
    if plotL:
        plt.imshow( L, 'coolwarm')
    else:
        plt.imshow( np.abs( S ), 'coolwarm')
    plt.colorbar()
    
    pt1 = 'L, ' if plotL else 'Similarity, '
    if found: 
        plt.title(pt1+'ep: '+str(ep)+', found goal')
    else:
        plt.title(pt1+'ep: '+str(ep))
    
    labels = [str((i, j)) for i in [1,2,3,4] for j in [1,2,3,4,5,6,7]]

    plt.xticks(np.arange(len(labels)), tuple(labels))
    plt.yticks(np.arange(len(labels)), tuple(labels))
    plt.rc('axes', labelsize=8)
    plt.xticks(rotation=270)
    
    plt.savefig(fname, dpi=dpi)
    if show: plt.show()
    plt.close()