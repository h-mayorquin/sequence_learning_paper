import  pprint
import  subprocess
import sys
sys.path.append('../')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


plt.rcParams['figure.figsize'] = (16, 12)

np.set_printoptions(suppress=True, precision=5)

sns.set(font_scale=2.5)

from network import Protocol, NetworkManager, Network
from patterns_representation import PatternsRepresentation
from analysis_functions import calculate_persistence_time, calculate_recall_quantities, deterministic_solution
from plotting_functions import plot_weight_matrix, plot_network_activity_angle, plot_persistent_matrix

epsilon = 10e-80
vmin = -10.0

lw = 12
ms = 22
alpha_graph = 0.3
colors = sns.color_palette()

plot_three_traces = True

#################
# Functions
#################


def run_sigma_sweep(sigma_number, samples, sigma_max, manager, T_persistence=0.050):

    manager.set_persistent_time_with_adaptation_gain(T_persistence=T_persistence)
    tau_s = manager.nn.tau_s
    T_recall = T_persistence * minicolumns
    T_cue = 3 * tau_s
    nr = manager.canonical_network_representation

    sigma_vector = np.linspace(0, sigma_max, num=sigma_number)
    sigma_normalized = sigma_vector

    successes_collection = np.zeros((sigma_number, samples))
    persistent_times_collection = {index: [] for index in range(sigma_number)}
    sequence_collection = {index: [] for index in range(sigma_number)}

    for index, sigma in enumerate(sigma_normalized):
        nn.sigma_in = sigma * np.sqrt(2 / tau_s)
        for sample in range(samples):
            aux = calculate_recall_quantities(manager, nr, T_recall, T_cue, remove=0.005,
                                              reset=True, empty_history=True)
            success, pattern_sequence, persistent_times, timings = aux
            successes_collection[index, sample] = success
            persistent_times_collection[index].append(persistent_times[:n_patterns])
            sequence_collection[index].append(pattern_sequence)

    results_dic = {'manager': manager, 'sigma_vector': sigma_vector, 'success': successes_collection,
                   'persistent times': persistent_times_collection, 'sequences': sequence_collection,
                   'T_persistence': T_persistence, 'sigma_norm': sigma_normalized}

    return results_dic


def plot_mean_success_vs_sigma(results_dictionary, index=0, label=0, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    successes_collection = results_dictionary['success']
    sigma_vector = results_dictionary['sigma_vector']

    std = successes_collection.std(axis=1)
    mean_success = successes_collection.mean(axis=1)
    sigma05_arg = np.where(mean_success <= 0.5)[0][0]
    sigma05 = sigma_vector[sigma05_arg]
    # Plot the mean
    ax.plot(sigma_vector, mean_success, 'o-', lw=lw, ms=ms, color=colors[index], label=label)
    # Plot sigma05
    ax.plot(sigma05, 0.5, '*', ms=ms, color='black')
    # Plot the interval
    low = mean_success - std
    low[low < 0.0] = 0.0
    high = mean_success + std
    ax.fill_between(sigma_vector, low, high, color=colors[index], alpha=alpha_graph)

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel('Success')
    ax.legend()

    return ax


def plot_persistent_time_vs_sigma(results_dictionary, index=0, label=0, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    persistent_times = results_dictionary['persistent times']
    sigma_vector = results_dictionary['sigma_vector']
    T_persistence = results_dictionary['T_persistence']
    mean_persistent = np.zeros(sigma_vector.size)
    std = np.zeros(sigma_vector.size)
    for sigma_index in range(sigma_vector.size):
        flatted = [item for l in persistent_times[sigma_index] for item in l[1:-1]]
        mean_persistent[sigma_index] = np.mean(flatted)
        std[sigma_index] = np.std(flatted)

    # Plot the mean
    ax.plot(sigma_vector, mean_persistent, 'o-', lw=lw, ms=ms + 5, color=colors[index], label=label)
    # Plot the error interval
    low = mean_persistent - std
    low[low < 0.0] = 0.0
    high = mean_persistent + std
    ax.fill_between(sigma_vector, low, high, color=colors[index], alpha=alpha_graph)

    ax.axhline(T_persistence, ls='--', color=colors[index])

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')
    ax.axhline(0.009, ls='--', color='red', label='limit')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$T_{persistence}$')
    ax.legend()

    return ax


def find_success_rate(manager, T_recall, n_samples, sigma_out):
    manager.nn.sigma_in = sigma_out * np.sqrt(2 / tau_s)
    manager.nn.sigma_out = sigma_out
    T_cue = manager.nn.tau_s
    I_cue = 0
    nr = manager.canonical_network_representation
    success_list = []
    for index in range(n_samples):
        aux = calculate_recall_quantities(manager, nr, T_recall, T_cue, remove=0.009, reset=True, empty_history=True)
        success, pattern_sequence, persistent_times, timings = aux
        success_list.append(success)

    return np.mean(success_list)


def find_right_length(length, manager, n_samples, T_recall):
    found = False
    number = 1.0
    while (not found):
        sigma_out = number * length
        success_rate = find_success_rate(manager, T_recall=T_recall, n_samples=n_samples, sigma_out=sigma_out)
        if success_rate < 0.50:
            found = True
        else:
            number += 1.0

    return number


def find_root(number, length, manager, T_recall, n_samples=50, deep=10, verbose=False):
    left = (number - 1) * length
    right = number * length
    if verbose:
        print('bounds', left, right)

    for i in range(deep):
        sigma_test = (left + right) * 0.5
        result = find_success_rate(manager, T_recall=T_recall, n_samples=n_samples, sigma_out=sigma_test)
        if result > 0.5:
            left = sigma_test
        else:
            right = sigma_test

        if np.abs(sigma_test - 0.5) < 0.01:
            break
        if verbose:
            print('result', result)
            print('bounds', left, right)

    return sigma_test, result


def find_p50(ws, wn, wb, alpha, minicolumns, hypercolumns, T_persistence, n_samples=100, deep=15, verbose=False):
    strict_maximum = True

    g_a = 1.0
    g_I = 10.0
    tau_a = 0.250
    G = 1.0
    sigma_out = 0.0
    tau_s = 0.010
    tau_z_pre = 0.025
    tau_z_post = 0.005

    n_patterns = minicolumns

    # Manager properties
    dt = 0.001
    values_to_save = ['o']

    # Neural Network
    nn = Network(hypercolumns, minicolumns, G=G, tau_s=tau_s, tau_z_pre=tau_z_pre, tau_z_post=tau_z_post,
                 tau_a=tau_a, g_a=g_a, g_I=g_I, sigma_out=sigma_out, epsilon=epsilon, prng=np.random,
                 strict_maximum=strict_maximum, perfect=False, normalized_currents=True)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Protocol
    manager.run_artificial_protocol(ws=ws, wn=wn, wb=wb, alpha=alpha)

    manager.set_persistent_time_with_adaptation_gain(T_persistence=T_persistence)

    # First ffind the half length
    n_samples_length = 30
    length = ws - (wn - alpha)
    T_recall = T_persistence * minicolumns + tau_s
    number = find_right_length(length, manager, n_samples_length, T_recall)
    sigma05, p05 = find_root(number, length, manager, T_recall, n_samples=n_samples, deep=deep, verbose=verbose)
    return sigma05, p05

#######
# Illustrative example -three traces-
########

if plot_three_traces:
    strict_maximum = True

    g_a = 1.0
    g_I = 100.0
    tau_a = 0.250
    G = 1.0
    sigma_out = 0.0
    tau_s = 0.010
    tau_z_pre = 0.025
    tau_z_post = 0.005

    hypercolumns = 1
    minicolumns = 5
    n_patterns = minicolumns

    # Training
    ws = 1.0
    wn = 0.0
    wb = -20.0
    alpha = 1.0

    T_persistence = 0.050

    # Manager properties
    dt = 0.001
    values_to_save = ['o']

    # Neural Network
    nn = Network( hypercolumns, minicolumns, G=G, tau_s=tau_s, tau_z_pre=tau_z_pre, tau_z_post=tau_z_post,
                  tau_a=tau_a, g_a=g_a, g_I=g_I, sigma_out=sigma_out, epsilon=epsilon, prng=np.random,
                  strict_maximum=strict_maximum, perfect=False, normalized_currents=True)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Protocol
    alpha_list = [0.5, 1.0, 1.5]

    sigma05_list = []
    results_dictionary_collection = {}
    sigma_number = 120
    samples = 1000
    sigma_max = 30

    for alpha in alpha_list:
        manager.run_artificial_protocol(ws=ws, wn=wn, wb=wb, alpha=alpha)
        results_dictionary = run_sigma_sweep(sigma_number, samples, sigma_max, manager, T_persistence=T_persistence)
        #process_sequence_statistics(results_dictionary)
        results_dictionary_collection[alpha] = results_dictionary


    fig1 = plt.figure()
    ax11 = fig1.add_subplot(111)
    for index, alpha in enumerate(alpha_list):
        label = r'$\Delta w_{rest}$ = ' + str(alpha)
        results_dictionary = results_dictionary_collection[alpha]
        ax11 = plot_mean_success_vs_sigma(results_dictionary,
                                        index=index, label=label, ax=ax11)
    fig1.savefig('./plot_producers/noise_illustration.pdf', frameon=False, dpi=110, bbox_inches='tight')
    plt.close(fig1)

########
# Times
#########

########
# Ratio
########
r_num = 10
r_vector = np.linspace(1, 1.00, num=r_num)
sigma05_vector = np.zeros(r_num)
p05_vector = np.zeros(r_num)


ws = 1.0
wn = 0.0
wb = -20.0
alpha = 9.0
minicolumns = 5
hypercolumns = 1
T_persistence = 0.050
n_samples = 500
deep = 10

if False:
    for index, alpha in enumerate(alpha_list):
        print(alpha)
        sigma05, p05 = find_p50(ws, wn, wb, alpha, minicolumns, hypercolumns, T_persistence,
                                n_samples=n_samples, deep=deep, verbose=False)
        print('--------')
        sigma05_vector[index] = sigma05
        p05_vector[index] = p05
