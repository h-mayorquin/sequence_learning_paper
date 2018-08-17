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

def covariance(t1, t2, tau_s, sigma_in):
    factor = tau_s * sigma_in * 0.5
    diff = np.abs(t1 - t2)
    addition = t1 + t2

    return factor * (np.exp(-diff / tau_s) + np.exp(-addition / tau_s))

tau_s = 0.010
dt = 0.001
T = 0.005
time = np.arange(0, T, dt)
nt = time.size
time1, time2 = np.meshgrid(time, time)
cov = np.zeros((nt, nt))
sigma_in = 1.0
cov = covariance(time1, time2, tau_s, sigma_in)