import numpy as np
import math
import random
import os


# Function
def target_function():
    return


# Function: Initialize Variables
def initial_position(pack_size, min_values, max_values, target_function, input, output, Ax):
    position = np.zeros((pack_size, len(min_values) + 1))
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        err, blank = target_function(input, output, position[i, 0:position.shape[1] - 1], Ax)
        position[i, -1] = err
    return position


# Function: Initialize Alpha
def alpha_position(dimension, target_function, input, output, Ax):
    alpha = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        alpha[0, j] = 0.0
    err, blank = target_function(input, output, alpha[0, 0:alpha.shape[1] - 1], Ax)
    alpha[0, -1] = err
    return alpha


# Function: Initialize Beta
def beta_position(dimension, target_function, input, output, Ax):
    beta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        beta[0, j] = 0.0
    err, blank = target_function(input, output, beta[0, 0:beta.shape[1] - 1], Ax)
    beta[0, -1] = err
    return beta


# Function: Initialize Delta
def delta_position(dimension, target_function, input, output, Ax):
    delta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        delta[0, j] = 0.0
    err, blank = target_function(input, output, delta[0, 0:delta.shape[1] - 1], Ax)
    delta[0, -1] = err
    return delta


# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i, -1] < alpha[0, -1]):
            alpha[0, :] = np.copy(updated_position[i, :])
        if (updated_position[i, -1] > alpha[0, -1] and updated_position[i, -1] < beta[0, -1]):
            beta[0, :] = np.copy(updated_position[i, :])
        if (updated_position[i, -1] > alpha[0, -1] and updated_position[i, -1] > beta[0, -1] and updated_position[
            i, -1] < delta[0, -1]):
            delta[0, :] = np.copy(updated_position[i, :])
    return alpha, beta, delta


# Function: Updtade Position
def update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function, input, output, Ax):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range(0, len(min_values)):
            r1_alpha = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            r2_alpha = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            a_alpha = 2 * a_linear_component * r1_alpha - a_linear_component
            c_alpha = 2 * r2_alpha
            distance_alpha = abs(c_alpha * alpha[0, j] - position[i, j])
            x1 = alpha[0, j] - a_alpha * distance_alpha
            r1_beta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            r2_beta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            a_beta = 2 * a_linear_component * r1_beta - a_linear_component
            c_beta = 2 * r2_beta
            distance_beta = abs(c_beta * beta[0, j] - position[i, j])
            x2 = beta[0, j] - a_beta * distance_beta
            r1_delta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            r2_delta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
            a_delta = 2 * a_linear_component * r1_delta - a_linear_component
            c_delta = 2 * r2_delta
            distance_delta = abs(c_delta * delta[0, j] - position[i, j])
            x3 = delta[0, j] - a_delta * distance_delta
            updated_position[i, j] = np.clip(((x1 + x2 + x3) / 3), min_values[j], max_values[j])
        err, blank = target_function(input, output, updated_position[i, 0:updated_position.shape[1] - 1], Ax)
        updated_position[i, -1] = err
    return updated_position


# GWO Function
def grey_wolf_optimizer(pack_size, min_values, max_values, iterations,
                        target_function, input, output, Ax):
    count = 0
    alpha = alpha_position(len(min_values), target_function, input, output, Ax)
    beta = beta_position(len(min_values), target_function, input, output, Ax)
    delta = delta_position(len(min_values), target_function, input, output, Ax)
    position = initial_position(pack_size, min_values, max_values, target_function, input, output, Ax)
    while (count <= iterations):
        print("Iteration = ", count, " f(x) = ", alpha[-1])
        a_linear_component = 2 - count * (2 / iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position = update_position(position, alpha, beta, delta, a_linear_component, min_values,
                                   max_values, target_function, input, output, Ax)
        count = count + 1
    print(alpha[-1])
    return alpha
