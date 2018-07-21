import numpy as np
import matplotlib.pyplot as plt
from connectivity_functions import fill_connection
from activity_representation import create_canonical_activity_representation
from network import Protocol, NetworkManager, Network
from analysis_functions import calculate_pesistence_time
from plotting_functions import plot_weight_matrix

epsilon = 10e-80

strict_maximum = True

g_w_ampa = 1.0
g_w = 0.0
g_a = 1.0
g_I = 10.0
tau_a = 0.250
G = 1.0
sigma_out = 0.0
tau_s = 0.020
tau_z_pre = 0.025
tau_z_post = 0.005

hypercolumns = 2
minicolumns = 10
n_patterns = 5


# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'i', 'a']

# Neural Network
nn = Network(hypercolumns, minicolumns, G=G, tau_s=tau_s, tau_z_pre=tau_z_pre, tau_z_post=tau_z_post,
                 tau_a=tau_a, g_a=g_a, g_I=g_I, sigma_out=sigma_out, epsilon=epsilon, prng=np.random,
                 strict_maximum=strict_maximum, perfect=False, normalized_currents=True)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Protocol
manager.run_artificial_protocol(ws=2.0, wn=1.0, wb=-1.0,alpha=0.5)
T_recall = 1.0
T_cue = tau_s
I_cue = 0
manager.run_network_recall(T_recall=10.0, T_cue=0.0, I_cue=None, reset=True, empty_history=True)