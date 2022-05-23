import numpy as np


def convert_probablities_arrays_to_move(probabilities_array):
    max = probabilities_array[0]
    max_index = 0
    for i in range(len(probabilities_array)):
        if probabilities_array[i] > max:
            max = probabilities_array[i]
            max_index = i
    return max_index

def convert_probablities_array_to_move(probabilities_array):
    total = []
    for i in range(len(probabilities_array)):
        total.append(convert_probablities_arrays_to_move(probabilities_array[i]))
    return np.array(total)
