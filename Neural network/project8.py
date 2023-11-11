import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.utils import check_random_state


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)


def make_spiral(n_samples_per_class=300, n_classes=2, n_rotations=3, gap_between_spiral=0.0,
    gap_between_start_point=0.0, equal_interval=True, noise=None, seed=None):
    assert 1 <= n_classes and type(n_classes) == int

    generator = check_random_state(None)

    X = []
    theta = 2 * np.pi * np.linspace(0, 1, n_classes + 1)[:n_classes]

    for c in range(n_classes):

        t_shift = theta[c]
        x_shift = gap_between_start_point * np.cos(t_shift)
        y_shift = gap_between_start_point * np.sin(t_shift)

        power = 0.5 if equal_interval else 1.0
        t = n_rotations * np.pi * (2 * generator.rand(1, n_samples_per_class) ** (power))
        x = (1 + gap_between_spiral) * t * np.cos(t + t_shift) + x_shift
        y = (1 + gap_between_spiral) * t * np.sin(t + t_shift) + y_shift
        Xc = np.concatenate((x, y))

        if noise is not None:
            Xc += generator.normal(scale=noise, size=Xc.shape)

        Xc = Xc.T
        X.append(Xc)

    X = np.concatenate(X)
    labels = np.asarray([c for c in range(n_classes) for _ in range(n_samples_per_class)])

    return X, labels


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)


class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_output_dim)
            self.model['b3'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_output_dim))
            self.model['b3'] = np.zeros((1, nn_output_dim))

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            y_hat: (numpy array) Array of shape (N, C) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        ### CODE HERE ###
        h1 = np.dot(X, W1) + b1 # nn_input_dim * nn_hdim1
        z1 = sigmoid(h1) # Sigmoid
        h2 = np.dot(z1, W2) + b2 # nn_input_dim * nn_hdim2
        z2 = tanh(h2) # Tanh
        h3 = np.dot(z2, W3) + b3 # nn_input_dim * nn_output_dim
        y_hat = np.exp(h3) / np.sum(np.exp(h3), axis=1).reshape(-1,1) # Softmax
        ############################
        cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'y_hat': y_hat}
    
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) One-hot encoding of training labels (N, C)
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        h1, z1, h2, z2, h3, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['y_hat']

        ### CODE HERE ###
        
        # dL/dh3
        dh3 = y_hat - y 
        
        # dL/dz : activation function derivative
        dz2 = 1 - z2**2 # tanh
        dz1 = z1*(1-z1) # sigmoid

        dW3 = np.dot(z2.T, dh3) + 2*L2_norm*W3 # dL/dW3
        db3 = (np.sum(dh3, axis=0)).reshape(1,-1) # dL/db3
        
        d2_tmp = dz2*np.dot(dh3, W3.T) # dL/dh2
        dW2 = np.dot(z1.T, d2_tmp) + 2*L2_norm*W2 # dL/dW2 
        db2 = (np.sum(d2_tmp, axis=0)).reshape(1,-1) # dL/db2
        
        d1_tmp = dz1*np.dot(d2_tmp, W2.T) # dL/dh3
        dW1 = np.dot(X.T, d1_tmp) + 2*L2_norm*W1 # dL/dW1
        db1 = (np.sum(d1_tmp, axis=0)).reshape(1,-1) # dL/db1
        ############################
        
        grads = dict()
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Descriptions:
            Evaluate the total loss on the dataset
        
        Args:
            y_pred: (numpy array) Predicted target (N, C)
            y_true: (numpy array) Array of training labels (N,C)
        
        Returns:
            loss: (float) Loss (data loss and regularization loss) for training samples.
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        ### CODE HERE ###
        total_loss = 0
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                if (y_true[i][j]==1): # one hot ecoding된 y_true가 1의 값을 갖는 인덱스
                    total_loss -= np.log(y_pred[i][j])
        total_loss += L2_norm*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) # add regularization term
        ############################

        return total_loss
        

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        y_train_onehot = np.eye(y_train.max()+1)[y_train] # y_true
        
        for it in range(epoch):
            ### CODE HERE ###
            y_hat, cache = self.forward_propagation(X_train)
            loss = self.compute_loss(y_hat, y_train_onehot, L2_norm)
            grads = self.back_propagation(cache, X_train, y_train_onehot, L2_norm) # learning

            # update weight & bias
            self.model['W1'] -= learning_rate*grads['dW1']
            self.model['b1'] -= learning_rate*grads['db1']
            self.model['W2'] -= learning_rate*grads['dW2']
            self.model['b2'] -= learning_rate*grads['db2']
            self.model['W3'] -= learning_rate*grads['dW3']
            self.model['b3'] -= learning_rate*grads['db3']
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)

                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")
 
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        y_pred, _ = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)
        #################  



def tanh(x):
    ### CODE HERE ###
    x = np.tanh(x)
    #################  
    return x

def relu(x):
    ### CODE HERE ###
    x = x*(x>0)
    ############################
    return x 


def sigmoid(x):
    ### CODE HERE ###
    x = 1/(1+np.exp(-x))
    ############################
    return x

######################################################################################




class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear layer.
        
        Args:
            x: (numpy array) Array containing input data, of shape (N, D)
            w: (numpy array) Array of weights, of shape (D, M)
            b: (numpy array) Array of biases, of shape (M,)

        Returns: 
            out: (numpy array) output, of shape (N, M)
            cache: (tupe[numpy array]) Values needed to compute gradients
        """
        ### CODE HERE ###
        out = np.dot(x,w)+b
        cache = [x, w]
        #################
        return out, cache  

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for an linear layer.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: (numpy array) Gradient with respect to x, of shape (N, D)
            dw: (numpy array) Gradient with respect to w, of shape (D, M)
            db: (numpy array) Gradient with respect to b, of shape (M,)
        """

        ### CODE HERE ###
        dx = np.dot(dout, cache[1].T)
        dw = np.dot(cache[0].T, dout)
        db = (np.sum(dout, axis=0)).reshape(1,-1)
        #################
        return dx, dw, db  


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Args:
            x: (numpy array) Input

        Returns:
            out: (numpy array) Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = relu(x)
        cache = out
        ################# 
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        dx = (np.heaviside(cache, 0))*dout
        return dx
        #################

class Tanh(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Tanh.

        Args:
            x: Input

        Returns:
            out: Output, array of the same shape as x
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = tanh(x)
        cache = out
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Tanh.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        dx = (1 - cache**2)*dout
        return dx
        #################  

class Sigmoid(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Sigmoid.

        Args:
            x: Input

        Returns:
            out: Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = sigmoid(x)
        cache = out
        #################
        return out, cache  

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Sigmoid.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        dx = (cache*(1-cache))*dout
        return dx
        #################  


class SoftmaxWithCEloss(object): 

    @staticmethod
    def forward(x, y=None):
        """
        if y is None, computes the forward pass for a layer of softmax with cross-entropy loss.
        Else, computes the loss for softmax classification.
        Args:
            x: Input data
            y: One-hot encoding of training labels or None 
       
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N, C) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        y_hat = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)
        if y is None:
            return y_hat
        else:
            loss = (-1)*np.sum(np.log(np.sum(y*y_hat,axis = 1)))
            cache = [y_hat, y]
            return loss, cache
        #################

    @staticmethod
    def backward(cache, dout=None):
        """
        Computes the loss and gradient for softmax classification.
        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        dx = cache[0] - cache[1]
        return dx
        #################  


class NeuralNetwork_module(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_output_dim)
            self.model['b3'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_output_dim))
            self.model['b3'] = np.zeros((1, nn_output_dim))

    def forward(self, X, y=None):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            y: (numpy array) One-hot encoding of training labels (N, C) or None
            
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N, C) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
            
        """

        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        cache = {}
        
        ### CODE HERE ###
        h1, cache['Linear1'] = Linear.forward(X, W1, b1) # cache -> (X, W1)
        z1, cache['Sigmoid'] = Sigmoid.forward(h1) # cache -> (z1)
        h2, cache['Linear2'] = Linear.forward(z1, W2, b2) # cache -> (z1, W2)
        z2, cache['Tanh'] = Tanh.forward(h2) # cache -> (z2)
        out, cache['Linear3'] = Linear.forward(z2, W3, b3) # cache -> (z2, W3)
        #################  

        if y is None:
            y_hat = SoftmaxWithCEloss.forward(out)
            return y_hat
        else: 
            loss, cache['SoftmaxWithCEloss'] = SoftmaxWithCEloss.forward(out, y)
            return cache, loss
    
    def backward(self, cache, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters          

        """
        ### CODE HERE ###
        dh3 = SoftmaxWithCEloss.backward(cache['SoftmaxWithCEloss']) # dL/dh3
        dz2, dW3, db3 = Linear.backward(cache['Linear3'], dh3) # dL/dz2, dL/dW3, dL/db3
        dW3 += 2*L2_norm*(cache['Linear3'][1])

        dh2 = Tanh.backward(cache['Tanh'], dz2) # dL/dh3
        dz1, dW2, db2 = Linear.backward(cache['Linear2'], dh2) # dL/dz1, dL/dW2, dL/db2
        dW2 += 2*L2_norm*(cache['Linear2'][1])

        dh1 = Sigmoid.backward(cache['Sigmoid'], dz1) # dL/dh1
        _, dW1, db1 = Linear.backward(cache['Linear1'], dh1) # _ , dL/dW1, dL/db1
        dW1 += 2*L2_norm*(cache['Linear1'][1])

        ###########################################
        grads = dict()
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        y_train_onehot = np.eye(y_train.max()+1)[y_train]
        
        for it in range(epoch):
            #################
            y_hat = self.forward(X_train)
            cache, loss = self.forward(X_train, y_train_onehot)
            loss += L2_norm*(np.sum(np.square(self.model['W1'])) + np.sum(np.square(self.model['W2'])) + np.sum(np.square(self.model['W3'])))
            grads = self.backward(cache, L2_norm)

            self.model['W1'] -= learning_rate*grads['dW1']
            self.model['b1'] -= learning_rate*grads['db1']
            self.model['W2'] -= learning_rate*grads['dW2']
            self.model['b2'] -= learning_rate*grads['db2']
            self.model['W3'] -= learning_rate*grads['dW3']
            self.model['b3'] -= learning_rate*grads['db3']
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)

                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")

         
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
        #################  