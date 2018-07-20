import numpy as np
import IPython


def create_orthogonal_canonical_representation(minicolumns, hypercolumns):
    aux = []
    for i in range(minicolumns):
        aux.append(i * np.ones(hypercolumns))

    return np.array(aux, dtype='int')


def build_network_representation(matrix, minicolumns, hypercolumns):
    network_representation = np.zeros((len(matrix), minicolumns * hypercolumns), dtype='int')

    for pattern, indexes in enumerate(matrix):
        for hypercolumn_index, minicolumn_index in enumerate(indexes):
            index = hypercolumn_index * minicolumns + minicolumn_index
            network_representation[pattern, index] = 1

    return network_representation


def transform_normal_to_neural_single(normal, minicolumns=2):

    neural = np.zeros((normal.size, minicolumns))

    #IPython.embed()

    for index, value in enumerate(normal):
        neural[index, int(value)] = 1

    #transformed_input[:, input] = 1

    neural = neural.flatten()
    return neural


def transform_neural_to_normal_single(neural, minicolumns=2):

    hypercolumns = int(np.sum(neural))
    normal = np.zeros(hypercolumns)

    for index, position in enumerate(np.where(neural == 1)[0]):
        normal[index] = position % minicolumns

    return normal


def transform_neural_to_normal(neural_matrix, minicolumns=2):
    """
    Transforms a matrix from the neural representation to the neural one

    :param neural_matrix: the neural representation
    :param quantization_value:  the number of values that each element is quantized

    :return: the normal matrix representation
    """

    number_of_elements, number_of_units = neural_matrix.shape

    normal_matrix = np.zeros((number_of_elements, number_of_units / minicolumns))

    for index, neural in enumerate(neural_matrix):
        normal_matrix[index, :] = transform_neural_to_normal_single(neural_matrix[index, :], minicolumns)

    return normal_matrix


def transform_singleton_to_normal(number, hypercolumns):

    return np.ones(hypercolumns) * number


def produce_pattern(number, hypercolumns, minicolumns):
    normal = transform_singleton_to_normal(number, hypercolumns)

    return transform_normal_to_neural_single(normal, minicolumns)

def build_ortogonal_patterns(hypercolumns, minicolumns):
    """
    This funtions builds the whole set of ortogonal patterns for a given
    number of hypercolumns and minicolumns
    :param hypercolums: The number of hypercolumns
    :param minicolumns: The number of minicolumns

    :return: A dictionary with the singleton as the key and the pattern in
     neural representation as the value.
    """

    patterns = {}
    patterns[None] = None

    for pattern_number in range(minicolumns):
        patterns[pattern_number] = produce_pattern(pattern_number, hypercolumns, minicolumns)

    return patterns




