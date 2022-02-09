import random
import math

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import multivariate_normal

def synthetic_function_a(x):
    # x.shape == (2,1)
    cov = np.array([[0.25, 0],
                    [0, 0.25]])
    mean = np.array([[-1,1], [0,1], [1,1],
                     [-1,0], [0,0], [1,0],
                     [-1,-1], [0,-1], [1,-1]])
    result = 0
    for each_mean in mean:
        result += multivariate_normal.pdf(x, each_mean, cov)
    return result

def synthetic_sample_a(number_of_samples):
    cov = np.array([[0.25, 0],
                    [0, 0.25]])
    mean = np.array([[-1,1], [0,1], [1,1],
                     [-1,0], [0,0], [1,0],
                     [-1,-1], [0,-1], [1,-1]])
    samples = np.random.multivariate_normal(mean[0], cov, int(number_of_samples/9)) 
    for i in range(1, mean.shape[0], 1):
        temp_samples = np.random.multivariate_normal(mean[i], cov, int(number_of_samples/9))
        samples = np.vstack((samples, temp_samples))
    np.random.shuffle(samples)
    return samples

def synthetic_function_b(x):
    # x.shape == (2,1)
    result = 0
    for i in range(8):
        result += multivariate_normal.pdf(x, generate_mean_synthetic_function_b(i),
                                             generate_cov_synthetic_function_b(i))
    return result

def synthetic_sample_b(number_of_samples):
    samples = np.random.multivariate_normal(generate_mean_synthetic_function_b(0),
                                            generate_cov_synthetic_function_b(0),
                                            int(number_of_samples/8)) 
    for i in range(1, 8, 1):
        temp_samples = np.random.multivariate_normal(generate_mean_synthetic_function_b(i),
                                                     generate_cov_synthetic_function_b(i),
                                                     int(number_of_samples/8))
        samples = np.vstack((samples, temp_samples))
    np.random.shuffle(samples)
    return samples

def generate_mean_synthetic_function_b(i):
    inside = i*math.pi/4
    return 3 * np.array([math.cos(inside), math.sin(inside)])

def generate_cov_synthetic_function_b(i):
    inside = i*math.pi/4
    cos_inside = math.cos(inside)
    sin_inside = math.sin(inside)
    width = 0.4**2
    length = 1**2
    top_left = length * cos_inside**2 + width * sin_inside**2
    top_right = (length-width) * sin_inside * cos_inside
    bottom_left = (length-width) * sin_inside * cos_inside
    bottm_right = length * sin_inside**2 + width * cos_inside**2
    return np.array([[top_left, top_right],
                     [bottom_left, bottm_right]])