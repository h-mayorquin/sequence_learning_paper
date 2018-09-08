import IPython
import numpy as np
import matplotlib.pyplot as plt
from connectivity_functions import fill_connection
from patterns_representation import create_canonical_activity_representation
from network import Protocol, NetworkManager, Network
from patterns_representation import PatternsRepresentation
from analysis_functions import calculate_persistence_time, calculate_recall_quantities
from plotting_functions import plot_weight_matrix, plot_network_activity_angle
from analysis_functions import calculate_probability_theo, calculate_joint_probabilities_theo
from analysis_functions import calculate_self_probability_theo, calculate_get_weights_theo

epsilon = 10e-80

np.seterr(over='raise')

strict_maximum = True

g_a = 1.0
g_I = 2.0
tau_a = 0.250
G = 1.0
sigma_out = 0.0
tau_s = 0.010
tau_z_pre = 0.025
tau_z_post = 0.005

hypercolumns = 10
minicolumns = 40
n_patterns = minicolumns
representation_overlap = 1.0
sequence_overlap = 0.5


# Training protocol
ws = 1.0
wn = 0.5
wb = -0.5
alpha = 0.5
alpha_back = 3.0

T_persistence = 0.050

# Manager properties
dt = 0.001
values_to_save = ['o']


# Neural Network
nn = Network(hypercolumns, minicolumns, G=G, tau_s=tau_s, tau_z_pre=tau_z_pre, tau_z_post=tau_z_post,
             tau_a=tau_a, g_a=g_a, g_I=g_I, sigma_out=sigma_out, epsilon=epsilon, prng=np.random,
             strict_maximum=strict_maximum, perfect=False, normalized_currents=True)


# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
x = manager.canonical_activity_representation
to_modify = int(representation_overlap * len(x[0]))
start_point = int(0.5 * len(x))
end_point = len(x)
start_changing = int(0.5 * (end_point - start_point) * (1 - sequence_overlap)) + start_point
sequence_overlap_size = int((end_point - start_point) * sequence_overlap)
for sequence_index in range(start_changing, start_changing + sequence_overlap_size):
    pattern = x[sequence_index]
    pattern[:to_modify] = manager.canonical_activity_representation[sequence_index - start_point][:to_modify]




