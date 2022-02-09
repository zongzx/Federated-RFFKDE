import random
import math

import pandas
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

class KernelDensityEstimation:
    def __init__(self, bandwidth=1):
        self.bandwidth = bandwidth
        
    def fit(self, X):
        # X: known data points
        # Consider 2d problem: X.shape == (n, 2)
        self.X = X
        self.n = X.shape[0] # Number of training data
        
    def estimate(self, x):
        # x: validation data points
        # Consider 2d problem: x.shape == (m, 2)
        m = x.shape[0] # m is the number of test or validation data points
        x = np.repeat(x, self.n, axis=0).reshape((m, self.n, 2))
        each_point_results = (1 / (math.pi * self.bandwidth)) * np.exp( (-1/2) * (1/self.bandwidth) 
                                * np.power(np.linalg.norm(self.X-x, 2, axis=2, keepdims=True), 2) )
        kernel_density_results = np.sum(each_point_results, axis=1) / self.n
        return kernel_density_results

    def estimate_for_likelihood(self, x, epsilon=1, delta=0.1):
        m = x.shape[0] # m is the number of test or validation data points
        kernel_density_results = np.zeros((m, 1))
        epsilon = 1
        delta = 0.1/m
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / epsilon
        for i in range(m):
            kernel_density_results[i, 0] = np.sum( (1 / (math.pi * self.bandwidth)) * np.exp( (-1/2) * (1/self.bandwidth)
                                * np.power(np.linalg.norm(self.X-x[i], 2, axis=1, keepdims=True), 2) ) ) / self.n + np.random.normal(loc=0, scale=sigma)
        kernel_density_results[kernel_density_results<=0] = 1e-300
        transformer = Normalizer(norm='l1')
        kernel_density_results = transformer.fit_transform(kernel_density_results.reshape((1, -1)))
        kernel_density_results = kernel_density_results.reshape((-1, ))
        return kernel_density_results

    def estimate_w_iter(self, x, epsilon=1, delta=0.1, dp=False):
        '''m = x.shape[0] # m is the number of test or validation data points
        each_point_results = []
        kernel_density_results = np.zeros((m, 1))
        for i in range(m):
            kernel_density_results[i, 0] = np.sum( (1 / (math.pi * self.bandwidth)) * np.exp( (-1/2) * (1/self.bandwidth) 
                                * np.power(np.linalg.norm(self.X-x[i], 2, axis=1, keepdims=True), 2) ) ) / self.n
        kernel_density_results[kernel_density_results<=0] = 1e-300
        transformer = Normalizer(norm='l1')
        kernel_density_results = transformer.fit_transform(kernel_density_results.reshape((1, -1)))
        kernel_density_results = kernel_density_results.reshape((-1, ))'''
        m = x.shape[0] # m is the number of test or validation data points
        if dp:
            delta /= m
            sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / epsilon
        each_point_results = []
        kernel_density_results = np.zeros((m, 1))
        for i in range(self.n):
            if dp:
                kernel_density_results += np.random.normal(loc=0, scale=sigma, size=(m,1))
            kernel_density_results +=  (1 / (math.pi * self.bandwidth)) * np.exp( (-1/2) * (1/self.bandwidth) 
                                * np.power(np.linalg.norm(self.X[i]-x, 2, axis=1, keepdims=True), 2) ) / self.n
        kernel_density_results[kernel_density_results<=0] = 1e-300
        transformer = Normalizer(norm='l1')
        kernel_density_results = transformer.fit_transform(kernel_density_results.reshape((1, -1)))
        kernel_density_results = kernel_density_results.reshape((-1, ))
        return kernel_density_results
    
    def likelihood(self, x, kernel_density_results=None):
        m = x.shape[0] # m is the number of test or validation data points
        if kernel_density_results is None:
            kernel_density_results = self.estimate_w_iter(x)
        '''scaler = MinMaxScaler()
        kernel_density_results = scaler.transform(kernel_density_results.reshape((-1,1)))'''
        return np.sum( np.log(kernel_density_results) ) / m