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
    Lmat = computeL(U, V)
    L = axL.matshow(Lmat)
    c = 1/np.linalg.det(Lmat+np.eye(2))
    P = np.linalg.det(Lmat) * c
    text1.set_text('$\\mathbb{P}(Y=\\{1, 2\\})$ = '+str(round(P, 4)))
    text2.set_text('$\\mathbb{P}(Y=\\{1\\})$ = '+str(round(Lmat[0,0] * c, 4)))
    text3.set_text('$\\mathbb{P}(Y=\\{2\\})$ = '+str(round(Lmat[1,1] * c, 4)))
    text4.set_text('$\\mathbb{P}(Y=\\emptyset)$ = '+str(round(c, 4)))
    fig.canvas.draw_idle()

q1, q2 = 3, 1.2
alpha1, alpha2 = np.pi/2, 0.1*np.pi



X, Y = np.zeros(2), np.zeros(2)

U, V = makeUV(q1, q2, alpha1, alpha2)

fig, (ax, axL) = plt.subplots(1,2)
Q = ax.quiver(X, Y, U, V, pivot='tail', color=['r','b'], scale = 1., units='xy', width=0.08)
Lmat = computeL(U, V)
L = axL.matshow(Lmat)
axL.axis('off')
plt.colorbar(L)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
c = 1/np.linalg.det(Lmat+np.eye(2))
P = np.linalg.det(Lmat) * c
text1 = fig.text(.56, .38, '$\\mathbb{P}(Y=\\{1, 2\\})$ = '+str(round(P, 4)))
text2 = fig.text(.56, .34, '$\\mathbb{P}(Y=\\{1\\})$ = '+str(round(Lmat[0,0] * c, 4)))
text3 = fig.text(.56, .30, '$\\mathbb{P}(Y=\\{2\\})$ = '+str(round(Lmat[1,1] * c, 4)))
text4 = fig.text(.56, .26, '$\\mathbb{P}(Y=\\emptyset)$ = '+str(round(c, 4)))
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

