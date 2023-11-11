from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    """
    Dataset class
    """
    
    def __init__(self, data_path):
        """
        Description:
            Load a csv file using 'load_data' method.
            Convert the dataframe to numpy array.
        
        Args:
            data_path (str): Path of the csv file.
                       
        Returns:
            
        """
        assert type(data_path)==str, 'The type of data_path has to be string'
        
        self.path = data_path
        self.features, self.data = self.load_data()
        self.min = None
        self.max = None        
                
    def load_data(self):
        """
        Description:
            Load the csv file.
            Print the head of data. 
        
        Args: 
            
            
        Returns:
            features (list): The names of the features.
            data (numpy array): Data.
        """
        dataframe = pd.read_csv(self.path)
        print(dataframe.head())
        features = list(dataframe)[:-1]
        data = dataframe.values
        return features, data
    
    def normalize(self):
        """
        Description:
            Do a min-max normalization
        
        Args:
        
        Returns:
            data (numpy array): Normalized data
            
        """
        self.min = self.data.min(axis=0)
        self.max = self.data.max(axis=0)
        data = (self.data - self.min) / (self.max - self.min)
        return data

    def parse_data(self, features):
        """
        Description:
            Parse the data using 'features'
            Do a normalize using 'normalize' method.
            Add a bias to the data.
        
        Args:
            features (list): The names of feature we use.
        
        Returns:
            X (numpy array): Input data
            y (numpy arary): Target (feature name: 'MEDV')
            
        """
        assert type(features)==list, 'The type of feature_names has to be list'
        assert all([isinstance(feature, str) for feature in features]), 'The element of features has to be string'
        
        data = self.normalize()
            
        indices = [self.features.index(feature) for feature in features]
        X = data[:, indices]
        y = data[:, -1]
        
        bias = np.ones([len(self.data), 1])
        X = np.concatenate((bias, X), axis=1)
        
        return X, y

    
class LinearRegressor:
    """
    Linear regressor class
    """
    def __init__(self, lr, tau, dim):
        """
        Description:
            Set the following attributes. 
                
                lr: learning rate
                tau: convergence tolerance
                dim: dimension of weight
                weight: regression coefficient
                loss_history: history of loss over the number of iterations.
        
        Args:
            lr (float): Learning rate.
            tau (float): Convergence condition.
            dim (int) : Dimension of weight.
            
        Returns:
            
        """
        
        ### CODE HERE ###
        weight = np.zeros(dim-1)
        self.weight = np.concatenate(([1], weight), axis=0)
        self.tau = tau
        self.lr = lr
        self.loss_history = np.empty(0)
        self.number_of_iteration = 1

    #################
    
    def prediction(self, X):
        """
        Description: 
            Predict the target variable
            
        Args:
            X (numpy array): Input data
            
        Returns:
            pred (numpy array or float): Predicted target.
        """
        
        ### CODE HERE ###
        bias = [1]
        X = np.append(bias, X)
        
        pred = np.dot(X, self.weight)
        #################
        
        return pred
    
    def compute_gradient(self, X, y):
        """
        Description:
            Calculate error and gradient.
        
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        
        Returns:
            error (numpy array or float): Error.
            gradient (numpy array): Gradient.
        """
        
        ### CODE HERE ###
        N = X.shape[0]
        X_T = X.transpose()
        pred_y = np.dot(X, self.weight)
        residual = y - pred_y
        residual_T = residual.transpose()
        error = np.dot(residual_T, residual)
        gradient = ((-2) * (np.dot(X_T, residual)))/N
        
        average_error = error / N
        loss = np.array(average_error)
        self.loss_history = np.append(self.loss_history, loss)
        #################
        
        return error, gradient
    
    def LR_with_gradient_descent(self, X, y):
        """
        Description:
            Do a gradient descent.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        
        ### CODE HERE ###
        _ , gradient = self.compute_gradient(X, y)
        abs_gradient = abs(gradient)
        infinity_norm = np.max(abs_gradient)

        while (infinity_norm > self.tau):
            self.weight -= self.lr * gradient
            _ , gradient = self.compute_gradient(X, y)
            abs_gradient = abs(gradient)
            infinity_norm = np.max(abs_gradient)
            self.number_of_iteration += 1

    #################
        
        
    def plot_loss_history(self):
        """
        Description:
            Plot the history of the loss.
        
        Args:
        
        Returns:
        
        """
        
        ### CODE HERE ###
        x = np.arange(self.number_of_iteration)
        y = self.loss_history
        plt.figure(figsize=(5, 5))
        plt.title('Average loss over # of Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Average loss')
        plt.plot(x, y)
        #################
        
        
    def calculate_house_price(self, input_features, min_value, max_value):
        """
        Description:
            Predict the house price.
        
        Args:
            input_features (numpy array): Input data.
            min_value (float): The minimum value of target.
            max_value (float): The maximum value of target.
        
        Returns:
            price (float): Predicted price.
            
        
        """
        
        ### CODE HERE ###
        min_features = min_value[:-1]
        max_features = max_value[:-1]
        normalizedInput = (input_features - min_features) / (max_features - min_features)
        pred = self.prediction(normalizedInput)
        price = pred*(max_value[-1] - min_value[-1]) + min_value[-1]
        #################
        
        return price

    
def LR_with_closed_form(X, y):
    
    """
    Description:
        Do linear regression without iteration.
            
    Args:
        X (numpy array): Input data.
        y (numpy array or float): Target data.
            
    Returns:
            
    """
    ### CODE HERE ###
    X_T = X.transpose()
    y.shape = (X.shape[0],1)
    weight = np.dot(np.linalg.inv(np.dot(X_T, X)), np.dot(X_T, y))
    weight = np.array(weight).flatten()
    #################
    return weight
    

def visualize(weight, X, y, features):
    """
    Description:
        Plot the data distribution and lines using weight vector.
        
    Args:
        weight (numpy array): Weight vector.
        X (numpy array): Input data.
        y (numpy array): Target data.
        
    Returns:
    
    """
    plt.figure(figsize=(30, 30))
    for i in range(1, X.shape[1]):
        pred_space = np.linspace(min(X[:, i]), max(X[:, i]), len(X[:, 0]))
        plt.subplot(4, 4, i)
        plt.scatter(X[:, i], y)
        plt.plot(pred_space, pred_space * weight[i] + weight[0], color='red', linewidth=4.0)
        plt.xlabel(features[i - 1])
        plt.ylabel('MEDV')
    plt.show()

