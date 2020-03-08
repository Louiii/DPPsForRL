import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import os
import json

beta_init = np.array([1.1, 1.5, 2, 5, 7, 10, 15, 20])
beta_frac = np.array([20000, 30000, 40000, 50000, 60000])
eta_start = np.array([20000, 30000, 40000, 50000, 60000])
eta_const = np.array([0.0002, 0.0005, 0.001, 0.003, 0.01])
all_params = [beta_init, beta_frac, eta_start, eta_const]

eta = [(es, ec) for es in eta_start for ec in eta_const]
eta = {e:i for i,e in enumerate(eta)}
eta_arr = np.array([[es, ec] for es in eta_start for ec in eta_const])

path = '../ReinforcementLearningWithDPPs/HamiltonDir/'
entries = os.listdir(path+'data/')

with open(path+'parameter_list.json', 'r') as f: lookup = json.load(f)
lookup_dict = {str(params):i for i, params in enumerate(lookup)}

# load all data
data = {}
for i, e in enumerate(entries):
    with open(path+e, 'r') as f: 
        d = json.load(f)
    dmax = max([sum(runs)/len(runs) for time, runs in d])
    if e[1:-5].isdigit(): 
        data[int(e[1:-5])] = dmax

def find_index4params(params):
    ''' map 4 values: eta_start, eta_const, beta_init, beta_frac -> index '''
    # (bi, bf, es, ec) = params
    prms = []
    for p, arr in zip(params, all_params):
        i = np.argmin(np.absolute(arr - p))
        prms.append(arr[i])
    return lookup_dict[str(prms)]


d_simp = {}
for (es, ec), i in eta.items():
    m = np.zeros((len(beta_init), len(beta_frac)))
    for a, bi in enumerate(beta_init):
        for b, bf in enumerate(beta_frac):
            m[a, b] = data[find_index4params((bi, bf, es, ec))]
    d_simp[i] = m

print('Loading complete')

def find_data2params(es, ec):
    ''' map two values, eta_start and eta_const -> matrix of (beta_init, beta_frac):max '''
    a = eta_arr - np.array([es, ec])
    print(a)
    i = np.argmin(np.absolute(np.sum(a, axis=1)))
    return d_simp[i]

while True:
    es = int(input('es'))
    ec = float(input('ec'))
    m = find_data2params(es, ec)
    print('next\n\n')


# lims = [0, 1.02]
# fig, (a1x) = plt.subplots(1, 1)
# plt.subplots_adjust(bottom=0.15)
# slider1_init = 10
# slider_1_delta = 1
# xs, ys, yerr = find_data(slider1_init)

# l = a1x.matshow(computeL(U, V))
# plt.colorbar(l)

# axcolor = 'lightgoldenrodyellow'
# ax1 = plt.axes([0.18, 0.1, 0.65, 0.03], facecolor=axcolor)
# ax2 = plt.axes([0.25,  0.05, 0.65, 0.03], facecolor=axcolor)

# slider_1 = Slider(ax1, 'n', 1, 30, valinit=slider1_init, valstep=slider_1_delta)
# slider_2 = Slider(ax2, 'yy', 0.1, 1.0, valinit=slider2_init, valstep=slider_1_delta)


# def update(val):
#     a = int(slider_1.val)
#     b = slider_2.val
#     xs, ys = find_data(n)
#     l.set_xdata(xs)
#     l.set_ydata(ys)
#     fig.canvas.draw_idle()


# slider_1.on_changed(update)
# slider_2.on_changed(update)

# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# def reset(event):
#     slider_1.reset()
#     slider_2.reset()
# button.on_clicked(reset)

# plt.show()
