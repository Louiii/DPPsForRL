import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def makeUV(q1, q2, alpha1, alpha2):
    x1, x2, y1, y2 = q1*np.cos(alpha1), q2*np.cos(alpha2), q1*np.sin(alpha1), q2*np.sin(alpha2)
    return np.array([x1, x2]), np.array([y1, y2])

def computeL(U, V):
    B = np.array([U.T,V.T])
    return np.dot(B.T, B)

def update(val):
    q1 = slider_1.val
    q2 = slider_2.val
    alpha1 = slider_3.val
    alpha2 = slider_4.val
    U, V = makeUV(q1, q2, alpha1, alpha2)
    Q.set_UVC(U,V)
    L = axL.matshow(computeL(U, V))
    fig.canvas.draw_idle()

q1 = 3
alpha1 = np.pi/2
q2 = 1
alpha2 = 3*np.pi/4



X, Y = np.zeros(2), np.zeros(2)

U, V = makeUV(q1, q2, alpha1, alpha2)

fig, (ax, axL) = plt.subplots(1,2)
Q = ax.quiver(X, Y, U, V, pivot='tail', color=['r','b'], scale = 1., units='xy', width=0.05)
L = axL.matshow(computeL(U, V))
axL.axis('off')
plt.colorbar(L)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

ax.set_title('Diversity vectors scaled by their quality')
axL.set_title('L')
plt.subplots_adjust(bottom=0.34)


q1_init = q1
q2_init = q2
alpha1_init = alpha1
alpha2_init = alpha2

ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
ax1 = plt.axes([0.25, 0.07, 0.65, 0.03], facecolor=axcolor)
ax2 = plt.axes([0.25, 0.12, 0.65, 0.03], facecolor=axcolor)
ax3 = plt.axes([0.25, 0.17, 0.65, 0.03], facecolor=axcolor)
ax4 = plt.axes([0.25, 0.22, 0.65, 0.03], facecolor=axcolor)


slider_1 = Slider(ax1, '$q_1$', 0.01, 5.0, valinit=q1_init, valstep=0.05)
slider_2 = Slider(ax2, '$q_2$', 0.01, 5.0, valinit=q2_init, valstep=0.05)
slider_3 = Slider(ax3, '$\\alpha_1$', 0.0, 2*np.pi, valinit=alpha1_init)
slider_4 = Slider(ax4, '$\\alpha_2$', 0.0, 2*np.pi, valinit=alpha2_init)


slider_1.on_changed(update)
slider_2.on_changed(update)
slider_3.on_changed(update)
slider_4.on_changed(update)

resetax = plt.axes([0.8, 0.015, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_1.reset()
    slider_2.reset()
    slider_3.reset()
    slider_4.reset()
button.on_clicked(reset)

# plt.legend()
plt.show()

