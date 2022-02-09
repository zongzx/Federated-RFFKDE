import random
import math

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

class KernelDensityEstimation_w_RFF:
    def __init__(self, bandwidth=1, num_fourier_features=1):
        self.bandwidth = bandwidth
        self.num_fourier_features = num_fourier_features
        
    def fit(self, X):
        # X: known data points
        # Consider 2d problem: X.shape == (n, 2)
        self.X = X
        self.n = X.shape[0] # Number of training data
        # self.w = math.sqrt(2*self.bandwidth) * np.random.normal(size=(self.n, self.num_fourier_features, 2))
        self.w = math.sqrt(1 / self.bandwidth) * np.random.normal(size=(self.n, self.num_fourier_features, 2))
        train_data_temp = self.w @ X.reshape(self.n, 2, 1)
        train_data_temp = train_data_temp.reshape((train_data_temp.shape[0], train_data_temp.shape[1]))
        self.train_data_random_features = np.hstack((np.cos(train_data_temp), np.sin(train_data_temp)))
        return self.w
        
    def estimate(self, x, norm=True):
        # x: validation data points
        # Consider 2d problem: x.shape == (m, 2)
        m = x.shape[0] # m is the number of test or validation data points
        sum_of_each_client_results = np.zeros((m,))
        privacy_loss = np.zeros((self.n,))
        x_for_mul = x.reshape((x.shape[0], 2, 1))
        for client in range(self.n):
            current_w = self.w[client].reshape((1, self.num_fourier_features, 2))
            rep_current_w = np.repeat(current_w, x.shape[0], axis=0)
            single_client_temp = rep_current_w @ x_for_mul
            single_client_temp = single_client_temp.reshape(single_client_temp.shape[0],
                                                          single_client_temp.shape[1])
            single_client_random_features = np.hstack((np.cos(single_client_temp), 
                                                     np.sin(single_client_temp)))
            single_client_estimation = single_client_random_features @ self.train_data_random_features[client].T
            sum_of_each_client_results += single_client_estimation
            ground_truth_position = self.X[client]
            peak_coordinates = x[single_client_estimation.argmax()]
            privacy_loss[client] = np.linalg.norm(peak_coordinates-ground_truth_position)
            #results.append(single_client_estimation)
        # sum_results = (1 / (math.pi * self.bandwidth)) * np.sum(results, axis=0) / (self.n * self.num_fourier_features)
        sum_results = (1 / (math.pi * self.bandwidth)) * sum_of_each_client_results / (self.n * self.num_fourier_features)
        if norm:
            sum_results[sum_results<=0] = 1e-300
            transformer = Normalizer(norm='l1')
            sum_results = transformer.fit_transform(sum_results.reshape((1, -1)))
            sum_results = sum_results.reshape((-1, ))
        return sum_results, privacy_loss #, results
    
    def estimate_all_local_maxima_privacy(self, x, scale, privacy_measure=False, ep=1.01, grid_size=100):
        # x: validation data points
        # Consider 2d problem: x.shape == (m, 2)
        m = x.shape[0] # m is the number of test or validation data points
        left, right, bottom, top = scale
        sum_of_each_client_results = np.zeros((m,))
        privacy_loss = np.zeros((self.n,))
        x_for_mul = x.reshape((x.shape[0], 2, 1))
        for client in range(self.n):
            current_w = self.w[client].reshape((1, self.num_fourier_features, 2))
            rep_current_w = np.repeat(current_w, x.shape[0], axis=0)
            single_client_temp = rep_current_w @ x_for_mul
            single_client_temp = single_client_temp.reshape(single_client_temp.shape[0],
                                                          single_client_temp.shape[1])
            single_client_random_features = np.hstack((np.cos(single_client_temp), 
                                                     np.sin(single_client_temp)))
            single_client_estimation = single_client_random_features @ self.train_data_random_features[client].T
            sum_of_each_client_results += single_client_estimation
            
            if privacy_measure:
                global_maxima = single_client_estimation.max()
                x_range = np.linspace(left, right, 100)
                y_range = np.linspace(bottom, top, 100)
                x_mesh, y_mesh = np.meshgrid(x_range, y_range)
                z = np.stack((x_mesh.flatten(), y_mesh.flatten()), axis=1)
                
                local_maxima_coorindates = []
                local_maxima_estimation = []
                single_client_estimation = single_client_estimation.reshape((grid_size, grid_size))
                coordinates = x.reshape((grid_size, grid_size, 2))
                for i in range(grid_size):
                    for y in range(grid_size):
                        if y-1 > 0 and single_client_estimation[i, y] > single_client_estimation[i, y-1] and\
                            i+1 < grid_size and single_client_estimation[i, y] > single_client_estimation[i+1, y] and\
                            i-1 > 0 and single_client_estimation[i, y] > single_client_estimation[i-1, y] and\
                            y+1 < grid_size and single_client_estimation[i, y] > single_client_estimation[i, y+1]:

                            if single_client_estimation[i, y] >= global_maxima / ep:
                                local_maxima_estimation.append(single_client_estimation[i, y])
                                local_maxima_coorindates.append(coordinates[i,y])
                if len(local_maxima_estimation) == 0:
                    ground_truth_position = self.X[client]
                    peak_coordinates = x[single_client_estimation.argmax()]
                    privacy_loss[client] = np.linalg.norm(peak_coordinates-ground_truth_position)
                    continue
                
                local_maxima_coorindates = np.array(local_maxima_coorindates)
                local_maxima_estimation = np.array(local_maxima_estimation)
                transformer = Normalizer(norm='l1')
                local_maxima_estimation = transformer.fit_transform(local_maxima_estimation.reshape((1, -1)))[0]
                ground_truth_location = self.X[client]
                privacy_loss[client] = (np.linalg.norm(local_maxima_coorindates-ground_truth_location, axis=1)).mean()

        sum_results = (1 / (math.pi * self.bandwidth)) * sum_of_each_client_results / (self.n * self.num_fourier_features)
        sum_results[sum_results<=0] = 1e-300
        transformer = Normalizer(norm='l1')
        sum_results = transformer.fit_transform(sum_results.reshape((1, -1)))
        sum_results = sum_results.reshape((-1, ))
        return sum_results, privacy_loss #, results
    
    def likelihood(self, x, kernel_density_results=None):
        m = x.shape[0] # m is the number of test or validation data points
        if kernel_density_results is None:
            kernel_density_results = self.estimate(x)[0]
        return np.sum( np.log(kernel_density_results) ) / m
