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


def update_saving_dictionary(values_dictionary, update_index, from_index, to_index,
                             training_times, manager):
    tau_z_pre = manager.nn.tau_z_pre
    tau_z_post = manager.nn.tau_z_post
    g_a = manager.nn.g_a

    T1 = training_times[from_index]
    T2 = training_times[to_index]
    Ts = T1
    T_start = sum(training_times[:from_index])
    Ttotal = manager.T_training_total

    # Get the pre-synaptic probabilities
    p_pre_theo = calculate_probability_theo(T1, T_start, Ttotal, tau_z_pre)
    p_post_theo = calculate_probability_theo(T2, T_start, Ttotal, tau_z_post)
    # Get the post-synaptic probabilities
    p_pre_sim = manager.nn.p_pre[from_index]
    p_post_sim = manager.nn.p_post[to_index]

    # Get the joint-probabilities
    p_co_theo = calculate_joint_probabilities_theo(T1, Ts, T2, Ttotal, tau_z_pre, tau_z_post)
    p_co_sim = nn.P[to_index, from_index]

    # Calculate the theoretical weights
    aux = calculate_get_weights_theo(T1, T2, Ttotal, tau_z_pre, tau_z_post)
    w_self_theo, w_next_theo, w_rest_theo, w_back_theo = aux

    # Calculate the simulation weights
    w_self_sim = manager.nn.w[from_index, from_index]
    w_next_sim = manager.nn.w[to_index, from_index]
    w_rest_sim = manager.nn.w[to_index + 1, from_index]
    w_back_sim = manager.nn.w[from_index - 1, from_index]

    # Calculate the inertia parameters B
    p_post_from_theo = calculate_probability_theo(T1, T_start, Ttotal, tau_z_post)
    p_post_to_theo = calculate_probability_theo(T2, T_start, Ttotal, tau_z_post)
    w_diff_theo = w_self_theo - w_next_theo
    beta_diff_theo = np.log10(p_post_from_theo) - np.log10(p_post_to_theo)
    B_theo = (w_diff_theo + beta_diff_theo) / g_a

    w_diff_sim = w_self_sim - w_next_sim
    beta_diff_sim = manager.nn.beta[from_index] - manager.nn.beta[to_index]
    B_sim = (w_diff_sim + beta_diff_sim) / g_a

    # Update the dictionary
    for value in values_dictionary.keys():
        aux = locals()[value]
        values_dictionary[value][update_index] = aux

    manager.T_training_total = 0.0


values_to_save_string = ['p_pre_theo', 'p_pre_sim', 'p_post_theo', 'p_post_sim', 'p_co_theo', 'p_co_sim',
                         'w_self_theo', 'w_self_sim', 'w_next_theo', 'w_next_sim', 'w_rest_theo', 'w_rest_sim',
                         'w_back_theo', 'w_back_sim', 'B_theo', 'B_sim', 'w_diff_theo', 'w_diff_sim',
                         'beta_diff_theo', 'beta_diff_sim']


strict_maximum = True

g_w_ampa = 1.0
g_w = 0.0
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
n_patterns = 10

# Training protocol
training_times = 0.100
training_times = [training_times for i in range(n_patterns)]
inter_pulse_intervals = 0.0
inter_sequence_interval = 0.0
resting_time = 0.0
epochs = 1

# Manager properties
dt = 0.0001
values_to_save = ['o']

from_index = 2
to_index = 3

# Neural Network
nn = Network(hypercolumns, minicolumns, G=G, tau_s=tau_s, tau_z_pre=tau_z_pre, tau_z_post=tau_z_post,
                 tau_a=tau_a, g_a=g_a, g_I=g_I, sigma_out=sigma_out, epsilon=epsilon, prng=np.random,
                 strict_maximum=strict_maximum, perfect=False, normalized_currents=True)


# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
# Build the representation
representation = PatternsRepresentation(manager.canonical_activity_representation[:n_patterns],
                                        minicolumns=minicolumns)


num = 30

training_times_vector = np.linspace(0.020, 3.0, num=num)
values_dictionary_training_times_equal = {name:np.zeros(num) for name in values_to_save_string}

for index, training_times in enumerate(training_times_vector):
    training_times = [training_times for i in range(n_patterns)]

    # Build the protocol
    protocol = Protocol()
    protocol.simple_protocol(representation, training_times=training_times, inter_pulse_intervals=inter_pulse_intervals,
                        inter_sequence_interval=inter_sequence_interval, epochs=epochs, resting_time=resting_time)

    # Run the protocol
    timed_input = manager.run_network_protocol_offline(protocol=protocol)

    # Update values
    update_saving_dictionary(values_dictionary_training_times_equal, index, from_index, to_index,
                             training_times, manager)

values_dictionary_training_times_equal['independent'] = 'training_times_vector'
values_dictionary_training_times_equal['independent values'] = training_times_vector
