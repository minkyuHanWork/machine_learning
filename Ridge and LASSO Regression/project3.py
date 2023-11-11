from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
class LinearRegressor:
    """
    LinearRegressor class with 'coordinate descent'.
    """
    def __init__(self, tau, dim):
        """
        Description:
            Set the attributes. 
                
                tau: convergence tolerance.
                dim: dimension of weight.
                weight: regression coefficient.
                loss: history of RSS loss over the number of iterations.
        
        Args:
            tau (float): Convergence condition.
            dim (int) : Dimension of weight.
            
        Returns:
            
        """
        
        ### CODE HERE ###
        self.tau = tau
        self.dim = dim
        
        self.weight = np.zeros((self.dim,))
        self.initialize_weight()
        
        self.loss = []
        #################
    
    def initialize_weight(self):
        """
        Description: 
            Initialize the weight randomly.
            Use the normal distribution.
            
        Args:
            
        Returns:
            
        """
        np.random.seed(0)
        ### CODE HERE ###
        # random Gaussian with mean of 0 and standard deviation of 1
        mean, sigma = 0, 1 
        self.weight = np.random.normal(mean, sigma, size = self.dim)
        #################
    
    
    def prediction(self, X):
        """
        Description: 
            Predict the target variable.
            
        Args:
            X (numpy array): Input data
            
        Returns:
            pred (numpy array or float): Predicted target.
        """
        
        ### CODE HERE ###
        pred = np.dot(X, self.weight)
        #################
        return pred
    
    def compute_residual(self, X, y):
        """
        Description:
            Calculate residual between prediction and target.
        
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        
        Returns:
            residual (numpy array or float): residual.
        """
        
        ### CODE HERE ###
        pred = self.prediction(X)
        residual = y - pred
        #################
        return residual
    
    def LR_with_coordinate_descent(self, X, y):
        """
        Description:
            Do a coordinate descent.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        
        ### CODE HERE ###
        conv = np.ones(self.dim)
        while max(abs(conv))>=self.tau :  
          for j in range (self.dim):
              weight_j = self.weight[j]
              self.weight[j] = 0 # for computing p
              feature_j = X[:,j]
              z = np.dot(feature_j.T, feature_j) # for normalization
              p = np.sum(feature_j*self.compute_residual(X,y))
              self.weight[j] = p/z
              conv[j] = self.weight[j] - weight_j # weight_update - weight_before
              
          residual = self.compute_residual(X,y)
          RSS = np.dot(residual.T, residual)
          self.loss.append(RSS)  
        #################
        
        
    def plot_loss_history(self):
        """
        Description:
            Plot the history of the RSS loss.
        
        Args:
        
        Returns:
        
        """
        ### CODE HERE ###
        x = range(len(self.loss)) # x-axis = number of iterations
        plt.plot(x, self.loss)
        plt.xlabel('Iterations')
        plt.ylabel('RSS loss')
        plt.title('RSS loss over # of iterations')
        plt.show()
        #################
        
        
class RidgeRegressor(LinearRegressor):
    """
    RidgeRegressor class. 
    You should inherit the LinearRegressor as base class.
    """
    def __init__(self, tau, dim, lambda_):
        """
        Description:
            Set the attributes. You can use super().
                
                tau: convergence tolerance.
                dim: dimension of weight.
                weight: regression coefficient.
                loss: history of RSS loss over the number of iterations.
                lambda_ : hyperparameter for regularization.
        
        Args:
            tau (float): Convergence condition.
            dim (int): Dimension of weight.
            lambda_ (float or int): Hyperparameter for regularization.
            
        Returns:
            
        """
        ### CODE HERE ###
        self.tau = tau
        self.dim = dim
        self.weight = np.zeros((self.dim,))
        LinearRegressor.initialize_weight(self)
        self.loss = []
        self.lambda_ = lambda_
        #################
        
    def LR_with_coordinate_descent(self, X, y):
        """
        Description:
            Do a coordinate descent. Do not penalize the intercept term.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        ### CODE HERE ###
        conv = np.ones(self.dim)
        while max(abs(conv))>=self.tau :
          for j in range (self.dim):
              weight_j = self.weight[j]
              self.weight[j] = 0
              feature_j = X[:,j]
              z = np.dot(feature_j.T, feature_j)
              p = np.sum(feature_j*LinearRegressor.compute_residual(self, X, y))
              if j == 0:
                self.weight[j] = p/z
              else :
                self.weight[j] = p/(z + self.lambda_)
              conv[j] = self.weight[j] - weight_j

          residual = LinearRegressor.compute_residual(self, X, y)
          RSS = np.dot(residual.T, residual)
          self.loss.append(RSS)  
        #################
    
    
class LassoRegressor(LinearRegressor):
    """
    LassoRegressor class. 
    You should inherit the LinearRegressor as base class.
    """
    def __init__(self, tau, dim, lambda_):
        """
        Description:
            Set the attributes. You can use super().
                
                tau: convergence tolerance.
                dim: dimension of weight.
                weight: regression coefficient.
                loss: history of RSS loss over the number of iterations.
                lambda_: hyperparameter for regularization.
                
        Args:
            tau (float): Convergence condition.
            dim (int) : Dimension of weight.
            lambda_ (float or int): Hyperparameter for regularization.
            
        Returns:
            
        """
        ### CODE HERE ###
        self.tau = tau
        self.dim = dim
        self.weight = np.zeros((self.dim,))
        LinearRegressor.initialize_weight(self)
        self.loss = []
        self.lambda_ = lambda_
        #################
    
    def LR_with_coordinate_descent(self, X, y):
        """
        Description:
            Do a coordinate descent. Do not penalize the intercept term.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        ### CODE HERE ###
        conv = np.ones(self.dim)
        while max(abs(conv))>=self.tau :
          for j in range (self.dim):
              weight_j = self.weight[j]
              self.weight[j] = 0
              feature_j = X[:,j]
              z = np.dot(feature_j.T, feature_j)
              p = np.sum(feature_j*LinearRegressor.compute_residual(self, X, y))
              if j == 0:
                self.weight[j] = p/z
              elif p < (-self.lambda_)/2:
                self.weight[j] = (p+self.lambda_/2)/z
              elif p > self.lambda_/2 :
                self.weight[j] = (p-self.lambda_/2)/z
              else :
                self.weight[j] = 0
              conv[j] = self.weight[j] - weight_j

          residual = LinearRegressor.compute_residual(self, X, y)
          RSS = np.dot(residual.T, residual)
          self.loss.append(RSS) 
        #################


def stack_weight_over_lambda(X, y, model_type, tau, dim, lambda_list):
    """
        Description:
            Calcualte the regression coefficients over lambdas and stack the results.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            mdoel_type (str): Type of model
            dim (int): Dimension of weight.
            lambda_list (list): List of lambdas.
            
        Returns:
            stacked_weight (numpy array): Weight stacked over lambda.
            
    """
    assert model_type in ['Lasso', 'Ridge'], f"model_type must be 'Ridge' or 'Lasso' but were given {model_type}"
    stacked_weight = np.zeros([len(lambda_list), X.shape[1]])
    ### CODE HERE ###
    num_lambda_list = len(lambda_list)
    if model_type == 'Ridge':
      for i in range(num_lambda_list):
        ridge_model = RidgeRegressor(tau, dim, lambda_list[i])
        ridge_model.LR_with_coordinate_descent(X, y)
        stacked_weight[i] = ridge_model.weight
    elif model_type == 'Lasso':
      for j in range(num_lambda_list):
        lasso_model = LassoRegressor(tau, dim, lambda_list[j])
        lasso_model.LR_with_coordinate_descent(X, y)
        stacked_weight[j] = lasso_model.weight
    #################
    return stacked_weight


def get_number_of_non_zero(weights):
    """
        Description:
            Find the number of non-zero weight in regression coefficients over lambdas.
            
        Args:
            weights (numpy array): Regression coefficients over lambdas.
            
        Returns:
            num_non_zero (list): Number of non-zero coefficients over lambdas.
    """
    num_non_zero = []
    ### CODE HERE ###
    for i in range(weights.shape[0]):
      temp_num_non_zero = 0
      for j in range(weights.shape[1]):
        if weights[i][j] != 0:
          temp_num_non_zero += 1
      num_non_zero.append(temp_num_non_zero) 
    #################
    return num_non_zero


def compute_errors(X, y, lambda_list, weights):
    """
        Description:
             Calcualte the RSS error between predictions and target values using 
             the output of stack_weight_over_lambda.
             
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            lambda_list (list): List of lambdas.
            weights (numpy array): Stacked weights.
            
        Returns:
            rss_errors (list): List of RSS errors calculated over lambdas.
    """
    assert len(lambda_list) == len(weights)
    rss_errors = []
    ### CODE HERE ###
    for i in range(len(lambda_list)):
      pred = np.dot(X, weights[i])
      residual = y - pred
      RSS = np.dot(residual.T, residual)
      rss_errors.append(RSS)
    #################
    return rss_errors

