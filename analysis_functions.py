import numpy as np
import itertools


def deterministic_solution(time, beta, w, tau_s, tau_a, g_a, s0, a0, unit_active):
    """

    :param time: the time of the solution
    :param beta: the bias
    :param w: the weight that its receiving
    :param tau_s: time constant of the unit
    :param tau_a: adaptation time constatn
    :param g_a: adaptation gain
    :param s0: initial value for the synaptic current
    :param a0: initial value of the adaptation curent
    :param unit_active: whether the unit is active or not.
    :return:
    """
    fixed_point = beta + w
    charge = a0
    r = tau_s / tau_a
    f = 1.0 / (1 - r)

    if unit_active:
        fixed_point -= g_a
        charge -= 1.0

    slow_component = g_a * f * charge * np.exp(-time / tau_a)
    fast_component = (s0 - fixed_point + g_a * charge * f) * np.exp(-time / tau_s)
    s = fixed_point - slow_component + fast_component

    return s


def calculate_pesistence_time(tau_a, w_diff, beta_diff, g_a, tau_s, perfect=False):
    """
    Formula for approximating the persistence time, the assumption for this is
    that the persistent time is >> than tau_s
    :param tau_a: the time constant of the adaptation
    :param w_diff: the difference between the weighs
    :param b_diff: the difference in the bias
    :param g_a: the adaptation current gain
    :param tau_s: the time constant of the unit
    :param perfect: whether the unit is a perfect integrator (capacitor)
    :return:
    """

    B = (w_diff + beta_diff)/ g_a
    T = tau_a * np.log(1 / (1 - B))
    if not perfect:
        r = tau_s / tau_a
        T += tau_a * np.log(1 / (1 - r))
    return T


def calculate_recall_quantities(manager, nr, T_recall, T_cue, remove=0.009, reset=True, empty_history=True):
    n_seq = nr.shape[0]
    I_cue = nr[0]

    # Do the recall
    manager.run_network_recall(T_recall=T_recall, I_cue=I_cue, T_cue=T_cue,
                               reset=reset, empty_history=empty_history)

    distances = calculate_angle_from_history(manager)
    winning = calculate_winning_pattern_from_distances(distances)
    timings = calculate_patterns_timings(winning, manager.dt, remove=remove)

    # Get the element of the sequence without consecutive duplicates
    aux = [x[0] for x in timings]
    pattern_sequence = [i for i, x in itertools.groupby(aux)]

    # Assume successful until proven otherwise
    success = 1.0
    for index, pattern_index in enumerate(pattern_sequence[:n_seq]):
        pattern = manager.patterns_dic[pattern_index]
        goal_pattern = nr[index]
        # Compare arrays of the recalled pattern with the goal
        if not np.array_equal(pattern, goal_pattern):
            success = 0.0
            break

    persistent_times = [x[1] for x in timings]
    return success, pattern_sequence, persistent_times, timings


def calculate_angle_from_history(manager):
    """
    :param manager: A manager of neural networks, it is used to obtain the history of the activity and
     the patterns that were stored

    :return: A vector with the distances to the stored patterns at every point in time.
    """
    if manager.patterns_dic is None:
        raise ValueError('You have to run a protocol before or provide a patterns dic')


    history = manager.history
    patterns_dic = manager.patterns_dic
    stored_pattern_indexes = np.array(list(patterns_dic.keys()))
    num_patterns = max(stored_pattern_indexes) + 1

    o = history['o'][1:]
    if o.shape[0] == 0:
        raise ValueError('You did not record the history of unit activities o')

    distances = np.zeros((o.shape[0], num_patterns))

    for index, state in enumerate(o):
        # Obtain the dot product between the state of the network at each point in time and each pattern
        nominator = [np.dot(state, patterns_dic[pattern_index]) for pattern_index in stored_pattern_indexes]
        # Obtain the norm of both the state and the patterns to normalize
        denominator = [np.linalg.norm(state) * np.linalg.norm(patterns_dic[pattern_index])
                       for pattern_index in stored_pattern_indexes]

        # Get the angles and store them
        dis = [a / b for (a, b) in zip(nominator, denominator)]
        distances[index, stored_pattern_indexes] = dis

    return distances


def calculate_winning_pattern_from_distances(distances):
    # Returns the number of the winning pattern
    return np.argmax(distances, axis=1)


def calculate_patterns_timings(winning_patterns, dt, remove=0):
    """

    :param winning_patterns: A vector with the winning pattern for each point in time
    :param dt: the amount that the time moves at each step
    :param remove: only add the patterns if they are bigger than this number, used a small number to remove fluctuations

    :return: pattern_timins, a vector with information about the winning pattern, how long the network stayed at that
     configuration, when it got there, etc
    """

    # First we calculate where the change of pattern occurs
    change = np.diff(winning_patterns)
    indexes = np.where(change != 0)[0]

    # Add the end of the sequence
    indexes = np.append(indexes, winning_patterns.size - 1)

    patterns = winning_patterns[indexes]
    patterns_timings = []

    previous = 0
    for pattern, index in zip(patterns, indexes):
        time = (index - previous + 1) * dt  # The one is because of the shift with np.change
        if time >= remove:
            patterns_timings.append((pattern, time, previous*dt, index * dt))
        previous = index

    return patterns_timings
