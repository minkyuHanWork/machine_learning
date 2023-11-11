from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt


class PCA:
    """PCA (Principal Components Analysis) class."""
    def __init__(self, num_components):
        """
        Descriptions:
            Constructor
        
        Args:
            num_components: (int) number of component to keep during PCA.  
        
        Returns:
            
        """
        self.num_components = num_components
        
        assert isinstance(self.num_components, int)

    
    def find_principal_components(self, X):
        """
        Descriptions:
            Find the principal components. The number of components is num_components.
            Set the class attribute, X_mean which represent the mean of training samples.
            
        Args:
            X : (numpy array, shape is (number of samples, dimension of feature)) training samples
                  
        Returns:
            
            
        """
        ### CODE HERE ###
        N = X.shape[0] # #Sample
        self.X_mean = np.mean(X, axis=0) # mean
        self.X_std = np.std(X, axis=0) # std
        X_standardized = (X - self.X_mean) / self.X_std
        Cov = np.dot(X_standardized.T, X_standardized) / N # Covariance
        eig_val, eig_vec = np.linalg.eigh(Cov) # eigenvalue, eigenvector
        self.eig_val_index = np.flip(eig_val.argsort())[:self.num_components] # eigenvalue값이 큰 순서대로 
        self.eigenbasis = (eig_vec[:, self.eig_val_index]).T # reduce dimensionality를 위한 vector
        #################
        
        assert self.eigenbasis.shape == (self.num_components, X.shape[1])
                                 
        
    def reduce_dimensionality(self, samples):
        """
        Descriptions:
            Reduce the dimensionality of data using the principal components. Before project the samples onto eigenspace,
            you should standardize the samples.
            
        Args:
            samples: (numpy array, shape is (number of samples, dimension of features))
                
        Returns:
            data_reduced: (numpy array, shape is (number of samples, num_components).) Data representation with only
                          num_components of the basis vectors.
                
        """
        ### CODE HERE ###
        samples_standardized = (samples - np.mean(samples, axis=0)) / np.std(samples, axis=0) # standardize the samples
        data_reduced = np.dot(samples_standardized, self.eigenbasis.T) # Data representation
        #################        
        assert data_reduced.shape == (samples.shape[0], self.num_components)

        return data_reduced
    
    def reconstruct_original_sample(self, sample_decomposed):
        """
        Descriptions:
            Normalize the training samples.
            
        Args:
            sample_decomposed: (numpy array, shape is (num_components, ).) Sample which decomposed using principal components
            keeped from PCA.
                
        Returns:
            representations_onto_eigenbasis: (numpy array, shape is (num_components, dimension of original feature).) 
            New feature reperesntation using eigenbasis which keeped from PCA.
            
            sample_recovered: (numpy array, shape is (dimension of original feature).) 
            Sample which recovered with linearly combined eigenbasis.
                
        """
        ### CODE HERE ###
        sample_decomposed = sample_decomposed.reshape([self.num_components, 1])
        representations_onto_eigenbasis = sample_decomposed * self.eigenbasis 
        sample_recovered = np.sum(representations_onto_eigenbasis, axis = 0) # reconstructure
        sample_recovered = sample_recovered*self.X_std + self.X_mean # undo standardization
        #################
        
        return representations_onto_eigenbasis, sample_recovered
    
    
class FaceRecognizer(PCA):
    """FaceRecognizer class."""
    def __init__(self, num_components, X, y):
        """
        Descriptions:
            Constructor. Inherit the PCA class.
        
        Args:
            num_components: (int) number of component to keep during PCA.  
            X : (numpy array, shape is (number of samples, dimension of feature)) training samples.
            y : (numpy array, shape is (number of samples, )) lables of corresponding samples.
        
        Returns:
        """
        ### CODE HERE ###
        self.num_components = num_components
        self.X = X
        self.y = y
        #################
        
    
    def generate_database(self):
        """
        Descriptions:
            Generate database using eigenface.
        
        Args:
        
        Returns:
        """
        
        ### CODE HERE ###
        pca = PCA(num_components=self.num_components)
        pca.find_principal_components(self.X)
        self.database = pca.reduce_dimensionality(self.X)
        #################
        
    
    def find_nearest_neighbor(self, X):
        """
        Descriptions:
            Find the nearest sample in the database.
        
        Args:
            X : (numpy array, shape is (number of samples, dimension of feature)) Query samples.
        
        Returns:
            pred: (numpy array, shape is (number of queries, )) Predictions of each query sample.
            distance: (numpy array, shape is (number of queries, 1)) Distances between query samples and corresponding DB.
            db_indices: (numpy array, shape is (number of queries, )) Indices of nearest samples in DB.
        """
        
        ### CODE HERE ###
        pca = PCA(num_components=self.num_components)
        pca.find_principal_components(self.X)
        query = pca.reduce_dimensionality(X) 
        DB = self.database

        pred = []
        distances = []
        db_indices = []
        for i in range(query.shape[0]):
            dist = np.sqrt(np.sum((DB - query[i])**2,axis = 1))
            idx = np.argmin(dist)
            min_dist = dist[idx]
            predict = self.y[idx]

            pred.append(predict)
            distances.append(min_dist)
            db_indices.append(idx)
        pred = np.array(pred)
        distances = np.array(distances).reshape(len(distances),1)
        db_indices = np.array(db_indices)
        #################
        
        return pred, distances, db_indices  
    

        
