import IPython
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from connectivity_functions import fill_connection
from patterns_representation import create_canonical_activity_representation
from network import Protocol, NetworkManager, Network
from patterns_representation import PatternsRepresentation
from analysis_functions import calculate_persistence_time, calculate_recall_quantities
from plotting_functions import plot_weight_matrix, plot_network_activity_angle
from analysis_functions import calculate_probability_theo, calculate_joint_probabilities_theo
from analysis_functions import calculate_self_probability_theo, calculate_get_weights_theo

A = [0 for i in range(7)]
A[0] = 9
A[1] = 3
A[2] = 9
A[3] = 3
A[4] = 9
A[5] = 7
A[6] = 9

print(A)


def return_second(first, A):
    value = A[first]
    second = first + 1
    while(A[first] != A[second]):
        second += 1

    return second

def return_sum(first, second, A):
    counter = 0
    for index in range(first, second):
        counter += A[index]

    return counter

A.sort()
here = 0
for i in range(0, len(A), 2):
    if A[i] == A[i + 1]:
        print('cool for i', i)
    else:
        here = i
        break

