import numpy as np 
import pandas as pd

from copy import deepcopy


def sign(x):
    return 2 * (x >= 0) - 1


class DecisionStump:
    """DecsionStump class"""
    
    def __init__(self):
        """
        Description:
            Set the attributes. 
                
                selected_feature (numpy.int): Selected feature for classification. 
                threshold: (numpy.float) Picked threhsold.
                left_prediction: (numpy.int) Prediction of the left node.
                right_prediction: (numpy.int) prediction of the right node.
        
        Args:
            
        Returns:
            
        """
        self.selected_feature = None
        self.threshold = None
        self.left_prediction = None
        self.right_prediction = None
    
    
    def fit(self, X, y):
        self.build_stump(X, y)            
        
    
    def build_stump(self, X, y):
        """
        Description:
            Build the decision stump. Find the feature and threshold. And set the predictions of each node. 
        
        Args:
            X: (N, D) numpy array. Training samples.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            
        """
        ### CODE HERE ###
        self.select_feature_split(X, y)
        #################
    
    
    def select_feature_split(self, X, y):       
        """
        Description:
            Find the best feature split. After find the best feature and threshold,
            set the attributes (selected_feature and threshold).
        
        Args:
            X: (N, D) numpy array. Training samples.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            
        """
        ### CODE HERE ###
        index = np.arange(X.shape[0])
        features = X.shape[1]
        min_error = 1
        for i in range(features):
            sorted_x = np.sort(X[index, i]) # X의 오름차순
            induplicate_sorted_x = np.sort(list(set(sorted_x))) # 동일한 값을 가진 value 제거

            # Find thresholds
            thresholds = []
            for j in range(len(induplicate_sorted_x) - 1):
                thresholds.append((induplicate_sorted_x[j] + induplicate_sorted_x[j+1])/2)

            for threshold in thresholds:
                # left / right 나누기 by feature, threshold
                left = []
                right = []
                for indice in index:
                    if (X[indice, i] < threshold):
                        left.append(indice)
                    else:
                        right.append(indice)
                
                # decide (left, right) majority class 
                left_majority = sign(sum(y[left]))
                right_majority = sign(sum(y[right]))

                # prediction value
                pred = np.zeros_like(y)
                pred[left] = left_majority
                pred[right] = right_majority

                # iteration -> compute error and update 
                temp_error = self.compute_error(pred, y)
                if (temp_error <= min_error):
                    min_error = temp_error
                    best_feature = i
                    best_threshold = threshold
                    best_left = left_majority
                    best_right = right_majority
        
        # attributes
        self.selected_feature = best_feature
        self.threshold = best_threshold
        self.left_prediction = best_left
        self.right_prediction = best_right           
        #################
        
        
    def compute_error(self, pred, y):
        """
        Description:
            Compute the error using quality metric in .ipynb file.
        
        Args:
            pred: (N, ) numpy array. Prediction of decision stump.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            out: (float)
            
        """
        ### CODE HERE ###
        array_len = len(y)
        num_error = 0
        for i in range(array_len):
            if pred[i] != y[i]:
                num_error += 1
        out = num_error/array_len
        #################
        return out
        
    
    def predict(self, X):
        """
        Description:
            Predict the target variables. Use the attributes.
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of decision stump.
            
        """
        ### CODE HERE ###
        x = X[:, self.selected_feature]
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if(x[i] < self.threshold):
                pred[i] = self.left_prediction
            else:
                pred[i] = self.right_prediction
        #################
        return pred


class AdaBoost:
    """AdaBoost class"""
    
    def __init__(self, num_estimators):
        """
        Description:
            Set the attributes. 
                
                num_estimator: int.
                error_history: list. List of weighted error history.
                classifiers: list. List of weak classifiers.
                             The items of classifiers (i.e., classifiers[1]) is the dictionary denoted as classifier.
                             The classifier has key 'coefficient' and 'classifier'. The values are the coefficient 
                             for that classifier and the Decsion stump classifier.

        
        Args:
            
        Returns:
            
        """
        np.random.seed(0)
        self.num_estimator = num_estimators
        self.classifiers = []
        self.error_history = []
        
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        ### CODE HERE ###
        # initialize the data weight
        self.data_weight = np.ones(self.y.shape[0])/y.shape[0]
        #################

        assert self.data_weight.shape == self.y.shape
        
        self.build_classifier()
        
    
    def build_classifier(self):
        """
        Description:
            Build adaboost classifier. Follow the procedures described in .ipynb file.
        
        Args:
            
        Returns:
            
        """
        ### CODE HERE ###
        # first DecisionStump
        stump = DecisionStump()
        stump.fit(self.X,self.y)
        pred = stump.predict(self.X)
        weighted_error = stump.compute_error(pred,self.y)
        self.error_history.append(weighted_error)
        coefficient = self.compute_classifier_coefficient(weighted_error)
        classifier = {'coefficient':coefficient, 'classifier' : stump}
        self.classifiers.append(classifier)
        self.data_weight = self.update_weight(pred, coefficient)
        self.data_weight = self.normalize_weight()

        for t in range(self.num_estimator-1):
            # learn decision tree 
            stump = DecisionStump()
            index_sampled = np.random.choice(np.arange(self.X.shape[0]), self.X.shape[0], p=self.data_weight)
            sampled_X = self.X[index_sampled]
            sampled_y = self.y[index_sampled]
            stump.fit(sampled_X,sampled_y)
            # compute coefficient & restore coefficient, classifier
            pred=stump.predict(self.X)
            weighted_error = 0
            for i in range(pred.shape[0]):
              if pred[i] != self.y[i]:
                weighted_error += self.data_weight[i]
            self.error_history.append(weighted_error)
            coefficient = self.compute_classifier_coefficient(weighted_error)
            classifier = {'coefficient':coefficient, 'classifier' : stump}
            self.classifiers.append(classifier)

            # Recompute & normailze weight
            self.data_weight = self.update_weight(pred, coefficient)
            self.data_weight = self.normalize_weight()
        #################
    
    
    def compute_classifier_coefficient(self, weighted_error):
        """
        Description:
            Compute the coefficient for classifier
        
        Args:
            weighted_error: numpy float. Weighted error for the classifier.
            
        Returns:
            coefficient: numpy float. Coefficient for classifier.
            
        """
        ### CODE HERE ###
        coefficient = (1/2)*np.log((1-weighted_error)/weighted_error)
        #################
        return coefficient
        
        
    def update_weight(self, pred, coefficient):
        """
        Description:
            Update the data weight. 
        
        Args:
            pred: (N, ) numpy array. Prediction of the weak classifier in one step.
            coefficient: numpy float. Coefficient for classifier.
            
        Returns:
            weight: (N, ) numpy array. Updated data weight.
            
        """
        ### CODE HERE ###
        weight = np.multiply(self.data_weight,np.exp((-1)*coefficient*np.multiply(self.y, pred)))
        #################
        return weight
        
        
    def normalize_weight(self):
        """
        Description:
            Normalize the data weight
        
        Args:
            
            
        Returns:
            weight: (N, ) numpy array. Norlaized data weight.
            
        """
        ### CODE HERE ###
        weight = self.data_weight/sum(self.data_weight)
        #################
        return weight
        
    
    
    def predict(self, X):
        """
        Description:
            Predict the target variables (Adaboosts' final prediction). Use the attribute classifiers.
            
            Note that item of classifiers list should be a dictionary like below
                self.classfiers[0] : classifier,  (dict)
                
            The dictionary {key: value} is composed,
                classifier : {'coefficient': (coefficient value),
                              'classifier' : (decision stump classifier)}
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of adaboost classifier. Output values are of 1 or -1.
            
        """
        ### CODE HERE ###
        N = X.shape[0]
        pred = np.zeros(N)
        temp_pred = np.zeros(N)
        # Classification
        for classfier in self.classifiers:
            temp_pred += classfier["coefficient"] * classfier['classifier'].predict(X)
        # sign
        for i in range(len(pred)):
            pred[i] = sign(temp_pred[i])
        #################
        return pred
    
    
    def predict_proba(self, X):
        """
        Description:
            Predict the probabilities of prediction of each class using sigmoid function. The shape of the output is (N, number of classes)
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            proba: (N, number of classes) numpy array. Probabilities of adaboost classifier's decision.
            
        """
        ### CODE HERE ###
        N = X.shape[0]
        pred = np.zeros(N)
        #classification
        for classifier in self.classifiers:
          pred += classifier['coefficient']*classifier['classifier'].predict(X)
        # sigmoid func
        for i in range(N):
          pred[i] = 1/(1+np.exp((-1)*pred[i]))
        # probabilities of prediction of each class
        proba = np.empty((N,2))
        proba[:, 0] = 1 - pred
        proba[:, 1] = pred
        #################
        return proba
        
    
def compute_staged_accuracies(classifier_list, X_train, y_train, X_test, y_test):
    """
        Description:
            Predict the accuracies over stages.
        
        Args:
            classifier_list: list of dictionary. Adaboost classifiers with coefficients.
            X_train: (N, D) numpy array. Training samples.
            y_train: (N, ) numpy array. Target variable, has the values of 1 or -1.
            X_test: (N', D) numpy array. Testing samples.
            y_test: (N', ) numpy array. Target variable, has the values of 1 or -1.
            
        Returns:
            acc_train: list. Accuracy on training samples. 
            acc_list: list. Accuracy on test samples.
                i.e, acc_train[40] =  $\hat{\mathbf{y}}=\text{sign} \left( \sum_{t=1}^{40} \hat{w_t} f_t(\mathbf{x}) \right)$
            
    """
    acc_train = []
    acc_test = []

    for i in range(len(classifier_list)):
    
        ### CODE HERE ###
        pred_train = np.zeros(y_train.shape[0])
        pred_test = np.zeros(y_test.shape[0])

        for j in range(i+1):
          pred_train += classifier_list[j]['coefficient']*classifier_list[j]['classifier'].predict(X_train)
          pred_test += classifier_list[j]['coefficient']*classifier_list[j]['classifier'].predict(X_test)

        for k in range(len(pred_train)):
          pred_train[k] = sign(pred_train[k])
        
        for l in range(len(pred_test)):
          pred_test[l] = sign(pred_test[l])

        train = np.average(pred_train == y_train)
        test = np.average(pred_test == y_test)

        acc_train.append(train)
        acc_test.append(test)
        #################
            
    return acc_train, acc_test
    
    
