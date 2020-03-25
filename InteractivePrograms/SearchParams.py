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
entries = [e for e in entries if e[-4:]=='json']

with open(path+'parameter_list.json', 'r') as f: lookup = json.load(f)
lookup_dict = {str(params):i for i, params in enumerate(lookup)}

# load all data
data = {}
for i, e in enumerate(entries):
    with open(path+'data/'+e, 'r') as f:
        d = json.load(f)
    # print(e)
    # for time, runs in d.items():
    #     print(runs)
    # dmax = max([sum(runs)/len(runs) for time, runs in d.items()])
    # dmax = max(list(d['average'].values()))
    if e[1:-5].isdigit(): 
        data[int(e[1:-5])] = max(list(d['average'].values()))

def find_index4params(params):
    ''' map 4 values: eta_start, eta_const, beta_init, beta_frac -> index '''
    # (bi, bf, es, ec) = params
    prms = []
    for p, arr in zip(params, all_params):
        i = np.argmin(np.absolute(arr - p))
        prms.append(int(arr[i]) if arr[i]%1==0 else arr[i])
    return lookup_dict[str(prms)]


d_simp = {}
mxx = ((beta_init[0], beta_frac[0], eta_start[0], eta_const[0]), -1)
for (es, ec), i in eta.items():
    m = np.zeros((len(beta_init), len(beta_frac)))
    for a, bi in enumerate(beta_init):
        for b, bf in enumerate(beta_frac):
            m[a, b] = data[find_index4params((bi, bf, es, ec))]
            if mxx[1] < m[a, b]:
                mxx = ((bi, bf, es, ec), m[a, b])
    d_simp[i] = m.copy()

print('Loading complete')
print(mxx)

def find_data2params(es, ec):
    ''' map two values, eta_start and eta_const -> matrix of (beta_init, beta_frac):max '''
    a = eta_arr - np.array([es, ec])
    i = np.argmin(np.absolute(np.sum(a, axis=1)))
    return d_simp[i]

# while True:
#     es = int(input('es'))
#     ec = float(input('ec'))
#     m = find_data2params(es, ec)
#     print(m)
#     print('next\n\n')


#####################################################################
###############          Slider code           ######################
#####################################################################

#es [20000, 30000, 40000, 50000, 60000]
#ec [0.0002, 0.0005, 0.001, 0.003, 0.01]

lims = [0, 1.02]
fig, (a1x) = plt.subplots(1, 1)
plt.subplots_adjust(bottom=0.15)
s1_init, d_s1 = 20000, 10000
s2_init, d_s2 = 0.0002, 0.00245
ec_ax = np.array([s2_init+d_s2*i for i in range(5)])
l = a1x.matshow(find_data2params(es, ec).T)
a1x.set_xlabel("$\\beta_{init}$")
a1x.set_ylabel("$\\beta_{frac}$")
a1x.set_xticklabels(['']+[str(b) for b in beta_init])
a1x.set_yticklabels(['']+[str(b) for b in beta_frac])
plt.text(0.5, 1.14, 'Average reward obtained over 5\nruns for each parameter value',
         horizontalalignment='center',
         fontsize=12,
         transform = a1x.transAxes)
plt.colorbar(l)

axcolor = 'lightgoldenrodyellow'
ax1 = plt.axes([0.05, 0.1, 0.65, 0.03], facecolor=axcolor)
ax2 = plt.axes([0.05,0.05, 0.65, 0.03], facecolor=axcolor)

slider_1 = Slider(ax1, '$\\eta_{st}$', eta_start[0], eta_start[-1], valinit=s1_init, valstep=d_s1)
slider_2 = Slider(ax2, '$\\eta_{cn}$', eta_const[0], eta_const[-1], valinit=s2_init, valstep=d_s2)


def update(val):
    es = slider_1.val
    ec = slider_2.val
    i = np.argmin(np.absolute(ec_ax-ec))
    ec = eta_const[i]
    l = a1x.matshow(find_data2params(es, ec).T)
    fig.canvas.draw_idle()


slider_1.on_changed(update)
slider_2.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_1.reset()
    slider_2.reset()
button.on_clicked(reset)

plt.show()
