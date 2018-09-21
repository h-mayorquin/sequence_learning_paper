import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import sys
sys.path.append('../')

np.set_printoptions(suppress=True, precision=2)

sns.set(font_scale=3.0)
sns.set_style(style='white')
epsilon = 10e-80
vmin = -1.0

from network import Protocol, NetworkManager, Network
from patterns_representation import PatternsRepresentation
from analysis_functions import calculate_persistence_time, calculate_recall_quantities
from plotting_functions import plot_weight_matrix, plot_network_activity_angle


def simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest, w_back):

    w = np.ones((minicolumns, minicolumns)) * 0.0
    for i in range(minicolumns):
        w[i, i] = w_self

    for i in range(minicolumns -1):
        w[i + 1, i] = w_next

    for i in range(minicolumns - 2):
        w[i + 2, i] = w_rest

    for i in range(1, minicolumns):
        w[i - 1, i] = w_back

    return w

strict_maximum = True

g_a = 1.0
g_I = 10.0
tau_a = 0.250
G = 1.0
sigma_out = 0.0
tau_s = 0.010
tau_z_pre = 0.025
tau_z_post = 0.005

hypercolumns = 1
minicolumns = 10
n_patterns = 5

w_self = 1.0
w_next = 0.5
w_rest = 0.25
w_back = -0.25

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'i', 'a', 'z_pre', 'z_post']

# Neural Network
nn = Network(hypercolumns, minicolumns, G=G, tau_s=tau_s, tau_z_pre=tau_z_pre, tau_z_post=tau_z_post,
                 tau_a=tau_a, g_a=g_a, g_I=g_I, sigma_out=sigma_out, epsilon=epsilon, prng=np.random,
                 strict_maximum=strict_maximum, perfect=False, normalized_currents=True)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
# Just to build the representations
manager.run_artificial_protocol(ws=w_self, wn=w_next, wb=w_back, alpha=0.5)
w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest, w_back)
nn.w = w
T_persistence = 0.100
manager.set_persistent_time_with_adaptation_gain(T_persistence=T_persistence)

nn.g_beta = 0.0

# Recall
T_recall = 1.0
T_cue = 0.080
I_cue = 0.0

n = 1

manager.run_network_recall(T_recall=T_recall, T_cue=T_cue, I_cue=I_cue, reset=True, empty_history=True)


# Extract quantities
norm = matplotlib.colors.Normalize(0, n_patterns)
cmap = matplotlib.cm.inferno_r

o = manager.history['o']
a = manager.history['a']
i_ampa = manager.history['i']

T_total = manager.T_recall_total
time = np.arange(0, T_total, dt)

# Plot
captions = True
annotations = True

gs = gridspec.GridSpec(3, 2)
fig = plt.figure(figsize=(22, 12))

# Captions
if captions:
    size = 35
    aux_x = 0.04
    fig.text(aux_x, 0.95, 'a)', size=size)
    fig.text(aux_x, 0.65, 'b)', size=size)
    fig.text(aux_x, 0.35, 'c)', size=size)
    fig.text(0.55, 0.95, 'd)', size=size)
    # fig.text(0.5, 0.40, 'e)', size=size)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

for pattern in range(n_patterns):
    width = 10.0 - pattern * 1.0
    if pattern == 0:
        label = 'Cue'
    else:
        label = str(pattern)

    ax1.plot(time, o[:, pattern], color=cmap(norm(pattern)), linewidth=width, label=label)
    ax2.plot(time, a[:, pattern], color=cmap(norm(pattern)), linewidth=width, label=label)
    ax3.plot(time, i_ampa[:, pattern] - g_a * a[:, pattern], color=cmap(norm(pattern)),
            linewidth=width)


ax1.set_title('Unit activity')
ax2.set_title('Adaptation current')
ax3.set_title('Self-Exc current minus adaptation')

ax3.axhline(w_next, ls='--', color='black', label=r'$w_{next}$')
ax3.legend(frameon=False, loc=3)
ax3.set_xlabel('Time (s)')
fig.tight_layout()

# Here we plot our connectivity matrix
rect = [0.46, 0.48, 0.40, 0.40]
# ax_conn = fig.add_subplot(gs[:2, 1])
ax_conn = fig.add_axes(rect)

ax_conn = plot_weight_matrix(manager, ax=ax_conn, vmin=vmin)



if annotations:
    letter_color = 'black'
    ax_conn.annotate(r'$w_{next}$', xy=(0, 0.7), xytext=(0, 4.5), color=letter_color,
                     arrowprops=dict(facecolor='red', shrink=0.15))

    ax_conn.annotate(r'$w_{self}$', xy=(0.0, 0), xytext=(4, 2), color=letter_color,
                arrowprops=dict(facecolor='red', shrink=0.05))

    ax_conn.annotate(r'$w_{back}$', xy=(4.9, 4.0), xytext=(6.8, 3.5), color=letter_color,
                arrowprops=dict(facecolor='red', shrink=0.05))

    ax_conn.annotate(r'$w_{rest}$', xy=(4, 6), xytext=(2.5, 9.0), color=letter_color,
                arrowprops=dict(facecolor='red', shrink=0.05))




# Let's plot our legends
# ax_legend = fig.add_subplot(gs[2, 1])
# lines = ax1.get_lines()
handles, labels = ax1.get_legend_handles_labels()
# ax_legend.legend(ax1.get_legend_handles_labels())

fig.legend(handles=handles, labels=labels, loc=(0.65, 0.09), fancybox=True, frameon=True, facecolor=(0.9, 0.9, 0.9),
           fontsize=28, ncol=2)


# plt.show()
fig.savefig('./plot_producers/simple_bcpnn_recall.pdf', frameon=False, dpi=110, bbox_inches='tight')
plt.close()