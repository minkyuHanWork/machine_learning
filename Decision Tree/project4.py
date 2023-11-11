import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

INFINITY = np.inf
EPSILON = np.finfo('double').eps


def load_data(df):
    """
    Return X, y and features.
    
    Args:
        df: pandas.DataFrame object.
    
    Returns:
        Tuple of (X, y)
        X (ndarray): include the columns of the features, shape == (N, D)
        y (ndarray): label vector, shape == (N, )
    """
    
    N = df.shape[0] # the number of samples
    D = df.shape[1] - 1 # the number of features, excluding a label
    
    ### CODE HERE ###
    data = df.values
    X = data[:, 1:]
    y = data[:, 0]
    #################

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (N, D) and y.shape == (N, ), f'{(X.shape, y.shape)}'
    
    return X, y


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)

class DecisionTree(object):
    def __init__(self, max_depth, min_splits):
        self.max_depth = max_depth
        self.min_splits = min_splits

    def fit(self, X, y):
        """
        Description:
            Return X, y and features.
    
        Args:
            X (numpy array): Input data shape == (N, D)
            y (numpy array): label vector, shape == (N, ) 
    
        Returns:
        """
        
        self.X = X
        self.y = y
        
        self.build()
    
    def build(self):
        """
        Description:
            Build a binary tree in Depth First Search fashion
                - Make a internal node using split funtion or a leaf node using leaf_node function
                - Use a stack to build a binary tree
                - Consider stop condition & early stop condition
        Args:
        Returns:
        """
        ### CODE HERE ###
        index = np.arange(self.X.shape[0])
        
        node_number = 0
        current_node = self.best_split(index)
        stack = []
        self.root = current_node
        while True:
            if current_node['is_leaf'] == 0:
                stack.append(current_node)
                temp_depth = current_node['depth'] + 1
                current_node['left'] = self.best_split(current_node['left_index'])
                current_node['left']['parent'] = current_node
                current_node = current_node['left']
                node_number += 1
                current_node['node_number'] = node_number
                current_node['depth'] = temp_depth
            
            elif (stack != [] and current_node['is_leaf'] == 1) :
                current_node = stack.pop()
                temp_depth = current_node['depth'] + 1
                current_node['right'] = self.best_split(current_node['right_index'])
                current_node['right']['parent'] = current_node
                current_node = current_node['right']
                node_number += 1
                current_node['node_number'] = node_number
                current_node['depth'] = temp_depth

            else:
                while(current_node['parent'] != None):
                    current_node = current_node['parent']
                self.root = current_node
                break

            if (current_node['depth'] == self.max_depth):
                temp_parent = current_node['parent']
                temp_depth = current_node['depth']
                temp_node_number = current_node['node_number']
                if current_node == temp_parent['left']:  
                    current_node = self.leaf_node(current_node['idx'])
                    current_node['parent'] = temp_parent
                    current_node['parent']['left'] = current_node
                    current_node['depth'] = temp_depth
                    current_node['node_number'] = temp_node_number
                elif current_node == temp_parent['right']:
                    current_node = self.leaf_node(current_node['idx'])
                    current_node['parent'] = temp_parent
                    current_node['parent']['right'] = current_node
                    current_node['depth'] = temp_depth
                    current_node['node_number'] = temp_node_number

        ###################

    def compute_gini_impurity(self, left_index, right_index):
        """
        Description:
            Compute the gini impurity for the indice 
                - if one of arguments is empty array, it computes node impurity
                - else, it computes weighted impurity of both sub-nodes of that split.

        Args:
            left_index (numpy array): indice of data of left sub-nodes  
            right_index (numpy array): indice of data of right sub-nodes

        Returns:
            gini_score (float) : gini impurity
        """
        ### CODE HERE ###
        if(left_index == [] or right_index == []):
            index = []    
            if(left_index == []):
                index = right_index
            elif(right_index == []):
                index = left_index
            
            majority_class = self.node_prediction(index)
            num_majority_class = 0
            for indice in index:
                if self.y[indice] == majority_class:
                    num_majority_class += 1
                    
            gini_score =2*((num_majority_class/len(index))*(1-num_majority_class/len(index)))
        else:
            num_left = len(left_index)
            num_right = len(right_index)
            total_num = num_left+num_right

            left_majority_class = self.node_prediction(left_index)
            num_left_majority_class = 0
            for indice in left_index:
                if self.y[indice] == left_majority_class:
                    num_left_majority_class += 1
            weighted_left = 2*((num_left_majority_class/len(left_index))*(1-num_left_majority_class/len(left_index)))

            right_majority_class = self.node_prediction(right_index)
            num_right_majority_class = 0
            for indice in right_index:
                if self.y[indice] == right_majority_class:
                    num_right_majority_class += 1
            weighted_right = 2*((num_right_majority_class/len(right_index))*(1-num_right_majority_class/len(right_index)))
            
            gini_score = (num_left/total_num)*weighted_left + (num_right/total_num)*weighted_right
        
        return gini_score
        #################

    def leaf_node(self, index):
        """ 
        Description:
            Make a leaf node(dictionary)

        Args:
            index (numpy array): indice of data of a leaf node

        Returns:
            leaf_node (dict) : leaf node
        """
        ### CODE HERE ###
        leaf_node = {}
        key = ['impurity','prediction','is_leaf', 'node_number', 'depth', 'idx','parent']
    
        impurity = self.compute_gini_impurity(index,[])     
        prediction = self.node_prediction(index)

        values = [impurity, prediction, 1, 0, 0, index, None]
        leaf_node = dict(zip(key,values)) 
        return leaf_node 
        #################

    def node_prediction(self, index):
        """ 
        Description:
            Make a prediction(label) as the most common class

        Args:
            index (numpy array): indice of data of a node

        Returns:
            prediction (int) : a prediction(label) of that node
        """
        ### CODE HERE ###
        label_0 = 0
        label_1 = 0
        pred_label = -1
        
        for i in index:
            if self.y[i] == 0:
                label_0 += 1
            elif self.y[i] == 1:
                label_1 += 1
            
        if label_0 > label_1:
            pred_label = 0
        else:
            pred_label = 1

        return pred_label
        #################
    
    def best_split(self, index):
        """ 
        Description:
            Find the best split information using the gini score and return a node

        Args:
            index (numpy array): indice of data of a node

        Returns:
            node (dict) : a split node that include the best split information(e.g., feature, threshold, etc.)
        """
        ### CODE HERE ###      
        split_node = {}

        # The data samples for a node is below min_split.
        if len(index) < self.min_splits :
            return self.leaf_node(index)    
        
        # data들의 인덱스를 받았어
        minimum_gini_impurity = 1
        features = self.X.shape[1]
        thresholds = []
        for i in range(features):
            sorted_x = np.sort(self.X[index, i])
            induplicate_sorted_x = np.sort(list(set(sorted_x)))

            for j in range(len(induplicate_sorted_x) - 1):
                thresholds.append((induplicate_sorted_x[j] + induplicate_sorted_x[j+1])/2)          

            for threshold in thresholds:
                left = []
                right = []
                for indice in index:
                    if(self.X[indice, i] < threshold):
                        left.append(indice)
                    else:
                        right.append(indice)

                present_gini_impurity = self.compute_gini_impurity(left, right)
                if (minimum_gini_impurity > present_gini_impurity):
                    minimum_gini_impurity = present_gini_impurity
                    greedy_feature = i
                    greedy_threshold = threshold
                    greedy_left = left
                    greedy_right = right
            thresholds = []

        # All samples in a node have the same target value, 그럼 stop해라
         ## => greedy_left 또는 greedy_right가 0이다.
        if (greedy_left == [] or greedy_right == []):
            return self.leaf_node(index)   
        
        # The split does not improve the weighted Gini impurity
         ## improvement가 향상이 없다, 즉 0보다 작거나 같다.    
        impurity = self.compute_gini_impurity(index, [])
        improve = impurity - minimum_gini_impurity
        if improve <= 0:
            return self.leaf_node(index)
        
        split_node['idx'] = index
        split_node['feature'] = greedy_feature
        split_node['threshold'] = greedy_threshold
        split_node['left'] = None
        split_node['right'] = None
        split_node['impurity'] = impurity
        split_node['prediction'] = self.node_prediction(index)
        split_node['is_leaf'] = 0
        split_node['improvement'] = improve
        split_node['left_index'] = greedy_left
        split_node['right_index'] = greedy_right
        split_node['node_number'] = 0
        split_node['depth'] = 0
        split_node['parent'] = None

        return split_node
        #################

    def predict(self, X):
        """ 
        Description:
            Determine the class of unseen sample X by traversing through the tree.

        Args:
            X (numpy array): Input data, shape == (N, D)

        Returns:
            pred (numpy array): Predicted target, shape == (N,)
        """
        ### CODE HERE ###
        N = X.shape[0]
        pred = np.arange(N)

        for i in range(N):
          tree = self.root
          while True:
            value = X[i, tree['feature']]
            if value <= tree['threshold']:
              if tree['left']['is_leaf'] == 0:
                tree = tree['left']
              else:
                pred[i] = tree['left']['prediction']
                break
            else:
              if tree['right']['is_leaf'] == 0:
                tree = tree['right']
              else:
                pred[i] = tree['right']['prediction']
                break
        return pred
        #################
    
    def traverse(self):
        """ 
        Description:
            Traverse through the tree in Breadth First Search fashion to compute various properties.
        
        Args:
        
        Returns:
        """
        ### CODE HERE ###
        q = []
        tree = self.root

        if tree == None:
          return
        
        q.append(tree)
         
        for node in q:
          if(node['depth'] == 1):
            print('       ', end="")
          elif(node['depth'] == 2):
            print('               ', end="")
          elif(node['depth'] == 3):
            print('                       ', end="")
          elif(node['depth'] == 4):
            print('                               ', end="")

          if(node['is_leaf'] == 0):
            result_split = f'node={node["node_number"]} is a split node: go to left node {node["left"]["node_number"]} if self.X[:, {node["feature"]}] <= {node["threshold"]:.4f} else to right node {node["right"]["node_number"]}: Impurity {node["impurity"]:.4f}, Improvement {node["improvement"]:.4f}, Precdtion => {node["prediction"]}'
            print(result_split)
          else:
            result_leaf = f'node={node["node_number"]} is a leaf node: Impurity {node["impurity"]:.4f}, Prediction -> {node["prediction"]}'
            print(result_leaf)

          if node['is_leaf']==0:
            q.append(node['left'])
            q.append(node['right'])
        #################
            

def plot_graph(X_train, X_test, y_train, y_test, min_splits = 2):
    """
    Description:
        Plot the depth, the number of nodes and the classification accuracy on training samples and test samples by varying maximum depth levels of a decision tree from 1 to 15.
    Args:
        X_train, X_test, y_train, y_test (numpy array)

    Returns:
    """
    ### CODE HERE ###
    max_depth = np.arange(15) + 1
    Train_accuracy = []
    Test_accuracy = []
    Depth = []
    Number_of_nodes = []
    for depth in max_depth:
        my_clf = DecisionTree(depth,min_splits)
        my_clf.fit(X_train,y_train)

        #train
        y_pred = my_clf.predict(X_train)
        train_accuracy = accuracy(y_train, y_pred)
        Train_accuracy.append(train_accuracy)
        #test
        y_pred = my_clf.predict(X_test)
        Test_accuracy.append(accuracy(y_test,y_pred))

        tree = my_clf.root
        q = []
        q.append(tree)
        depth = 0
        for node in q:
          if node['depth'] > depth:
            depth = node['depth']
          if node['is_leaf']==0:
            q.append(node['left'])
            q.append(node['right'])

        Depth.append(depth)
        Number_of_nodes.append(len(q))      

    fig = plt.figure(figsize=(21, 6))

    plt.subplot(1, 3, 1)                
    plt.plot(max_depth, Train_accuracy, label = 'train_accuracy')
    plt.plot(max_depth, Test_accuracy, label = 'test_accuracy')
    plt.legend()
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 2)                
    plt.plot(max_depth, Depth)
    plt.xlabel('max_depth')
    plt.ylabel('Depth')

    plt.subplot(1, 3, 3)                
    plt.plot(max_depth, Number_of_nodes)
    plt.xlabel('max_depth')
    plt.ylabel('Number of nodes')

    plt.show()
    #################


    