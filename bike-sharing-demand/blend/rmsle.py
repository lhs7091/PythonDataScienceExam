import numpy as np


def rmsle(predicted_values, actual_values, convertExp=True):
    if convertExp:
        predicted_values = np.exp(predicted_values)
        actual_values = np.exp(actual_values)

    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    difference = np.log(predicted_values + 1) - np.log(actual_values + 1)
    difference = np.square(difference)
    score = np.sqrt(difference.mean())

    return score