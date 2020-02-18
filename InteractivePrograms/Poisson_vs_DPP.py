import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.linalg import eigh
import sys
sys.path.append('../')
from DPP import DPP

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
    return D, V, xf, yf

def generate_samples(n):
    D, V, xf, yf = makeDV(n)
    dpp = DPP(D, V)
    dpp_sample = dpp.sample()
    ind_sample = np.random.randint(n*n, size=len(dpp_sample))
    return xf[dpp_sample], yf[dpp_sample], xf[ind_sample], yf[ind_sample]


lims = [0, 1.02]
fig, (a1x, a2x) = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.15)
slider1_init = 10
slider_1_delta = 1
xs, ys, xs2, ys2 = generate_samples(slider1_init)
l, = a1x.plot(xs, ys, 'b.', ms=5)
l2, = a2x.plot(xs2, ys2, 'r.', ms=5)
a1x.margins(x=0)
a2x.margins(x=0)
a1x.set_title("DPP")
a2x.set_title("Independent")
a1x.set_xlim(lims)
a1x.set_ylim(lims)
a1x.set_xticks([])
a1x.set_yticks([])
a2x.set_xlim(lims)
a2x.set_ylim(lims)
a2x.set_xticks([])
a2x.set_yticks([])

axcolor = 'lightgoldenrodyellow'
ax1 = plt.axes([0.18, 0.1, 0.65, 0.03], facecolor=axcolor)
# ax2 = plt.axes([0.25,  0.05, 0.65, 0.03], facecolor=axcolor)
# ax3 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

slider_1 = Slider(ax1, 'n', 1, 30, valinit=slider1_init, valstep=slider_1_delta)
# slider_2 = Slider(ax2, 'yy', 0.1, 1.0, valinit=slider2_init, valstep=slider_1_delta)
# slider_3 = Slider(ax3, 'xy', 0.0, 1.0, valinit=slider3_init)


def update(val):
    n = int(slider_1.val)
    # yy = slider_2.val
    # xy = slider_3.val
    xs, ys, xs2, ys2 = generate_samples(n)
    l.set_xdata(xs)
    l.set_ydata(ys)
    l2.set_xdata(xs2)
    l2.set_ydata(ys2)
    fig.canvas.draw_idle()


slider_1.on_changed(update)
# slider_2.on_changed(update)
# slider_3.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_1.reset()
    # slider_2.reset()
    # slider_3.reset()
button.on_clicked(reset)

plt.show()
