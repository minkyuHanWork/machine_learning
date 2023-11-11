import numpy as np
from matplotlib import pyplot as plt

def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)

class GaussianKernel():
    """
    Description:
         Filter the value with a Gaussian smoothing kernel with lambda value, and returns the filtered value.
    """
    def __init__(self, l):
        self.lamdba = l
    
    def __call__(self, value):
        """
        __call_() : 클래스를 함수로써 호출 가능하게 함 -> 다른 함수의 파라미터로 사용 가능
        Args:
            value (numpy array) : input value
        Returns:
            value (numpy array) : filtered value
        """

        ### CODE HERE ###
        return np.exp((-1)*(value**2)/self.lamdba)
        ############################


class KNN_Classifier():
    def __init__(self,n_neighbors=5,weights=None):

        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        """
        Description:
            Fit the k-nearest neighbors classifier from the training dataset.
    
        Args:
            X (numpy array): input data shape == (N, D)
            y (numpy array): label vector, shape == (N, ) 
    
        Returns:
        """
        
        self.X = X
        self.y = y
        
    def kneighbors(self, X):
        """
        Description:
            Find the K-neighbors of a point.
            Returns indices of and distances to the neighbors of each point.
    
        Args:
            X (numpy array): Input data, shape == (N, D)
            
        Returns:
            dist(numpy array) : Array representing the pairwise distances between points and neighbors , shape == (N, self.n_neighbors)
            idx(numpy array) : Indices of the nearest points, shape == (N, self.n_neighbors)
                
        """
        
        N = X.shape[0]
        ### CODE HERE ###
        train_N = self.X.shape[0]
        dist = []
        idx = []
        for n in range(N):
            dist_list = [] 
            for i in range(train_N):    
                dist_element = np.sqrt(np.sum((X[n, :] - self.X[i, :])**2))
                dist_list.append(dist_element)
            idx_sort = np.argsort(dist_list)[0:self.n_neighbors]
            dist_sort = np.sort(dist_list)[0:self.n_neighbors]
            idx.append(idx_sort)
            dist.append(dist_sort)
        dist = np.array(dist)    
        idx = np.array(idx)
        ############################
        
        assert dist.shape == (N, self.n_neighbors)
        assert idx.shape == (N, self.n_neighbors)
        
        return dist, idx
        
    
    def make_weights(self, dist, weights):
        """
        Description:
            Make the weights from an array of distances and a parameter ``weights``.

        Args:
            dist (numpy array): The distances.
            weights : weighting method used, {'uniform', 'inverse distance' or a callable}

        Returns:
            (numpy array): array of the same shape as ``dist``
        """

        ### CODE HERE ###
        weight = np.ones_like(dist, dtype=float)
        if (self.weights == "uniform"):
            pass
        elif (self.weights == "inverse distance"):
            dist = np.where(dist==0,1,dist)
            weight = np.reciprocal(dist) 
        else:
            weight = self.weights(dist)

        return weight
        ############################

    def most_common_value(self, val, weights, axis=1):
        """
        Description:
            Returns an array of the most common values.

        Args:
            val (numpy array): 2-dim array of which to find the most common values.
            weights (numpy array): 2-dim array of the same shape as ``val``
            axis (int): Axis along which to operate
        Returns:
            (numpy array): Array of the most common values.
        """

        ### CODE HERE ###
        #MC_values = np.zeros(val.shape[0])
        #score = np.arange(len(set(self.y)))
        #for i in range(val.shape[0]):
        #    for j in range(val.shape[1]):
        #        idx = val[i][j] 
        #        score[idx] += weights[i][j]
        #        MC_value = np.argmax(score)
        #MC_values[i] = MC_value

        #return MC_values
        N = len(list(set(self.y))) # y에 존재하는 class의 개수
        MC_value = []
        if (axis==1):
            for k in range(val.shape[0]):
                values = []            
                for i in range(N):
                    temp = np.sum(weights[k, np.where(val[k, :]==i)])  
                    values.append(temp)
                MC_value.append(np.argsort(values)[-1])
            return np.array(MC_value)       
        elif (axis == 0):
            weights = np.ones(N)
            MC_value = []
            values = []
            for i in range(N):
                temp = np.sum(weights[np.where(val==i)])
                values.append(temp)
            MC_value.append(np.argsort(values)[-1])
            return np.array(MC_value)
        ############################

    def predict(self, X):
        """ 
        Description:
            Predict the class labels for the input data.
            When you implement KNN_Classifier.predict function, you should use KNN_Classifier.kneighbors, KNN_Classifier.make_weights, KNN_Classifier.most_common_value functions.

        Args:
            X (numpy array): Input data, shape == (N, D)

        Returns:
            pred (numpy array): Predicted target, shape == (N,)
        """

        ### CODE HERE ###
        dist, idx = self.kneighbors(X)
        val = self.y[idx]
        weights = self.make_weights(dist, self.weights)
        pred = self.most_common_value(val, weights)
        return pred
        ############################

def stack_accuracy_over_k(X_train, y_train, X_test, y_test, max_k=50, weights_list = ["uniform", "inverse distance", GaussianKernel(1000000)]):
    """ 
    Description:
        Stack accuracy over k.

    Args:
        X_train, X_test, y_train, y_test (numpy array)
        max_k (int): a maximum value of k
        weights_list (List[any]): a list of weighting method used
    Returns:
    """
    
    ### CODE HERE ###
    train_accs = []
    test_accs = []
    k_list = np.arange(1,max_k+1)
    n_weights = len(weights_list)
    for i in k_list:
        train_acc = []
        test_acc = []
        for w in range(n_weights):
            clf = KNN_Classifier(n_neighbors=i, weights=weights_list[w])
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            train_acc.append(accuracy(y_pred_train,y_train))
            test_acc.append(accuracy(y_pred_test,y_test))
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    train_accs = np.transpose(np.array(train_accs))
    test_accs = np.transpose(np.array(test_accs))

    figure = plt.figure(figsize = (20, 4))
    for n in range(n_weights):
        plt.subplot(1,n_weights,n+1)                
        plt.plot(k_list, train_accs[n,:],label='train accuracy')
        plt.plot(k_list, test_accs[n,:],label='test accuracy')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over k')
        plt.legend()
    plt.show()
    ############################
    
def knn_query(X_train, X_test, X_train_image, X_test_image, y_train, y_test, names, n_neighbors=5, n_queries=5):
    np.random.seed(42)
    my_clf = KNN_Classifier(n_neighbors=n_neighbors, weights="uniform")
    my_clf.fit(X_train, y_train)

    data = [(X_train, y_train, X_train_image), (X_test, y_test, X_test_image)]
    train = True
    for X, y, image in data:
        for i in range(n_queries):
            fig = plt.figure(figsize=(16, 6))
            rnd_indice = np.random.randint(low=X.shape[0], size=n_queries) # 데이터(X)의 인덱스 중 n_queries만큼 선택 / seed가 있어 동일 
            nn_dist, nn_indice = my_clf.kneighbors(X) # X와 Nearest한 dist와 indice / (X.shape[0], n_neighbors)

            idx = rnd_indice[i] # 위에서 랜덤하게 뽑은 인덱스가 반복마다 순서대로 들어감
            query = image[idx] # 샘플 이미지
            name = names[y[idx]] # 이미지의 이름
            prediction = my_clf.most_common_value(y_train[nn_indice[idx]], None, axis=0).astype(np.int8) # nearest로 뽑힌 y(class) : (n_neighbors, )
            prediction = names[prediction[0]]

            plt.subplot(1, n_neighbors + 1, 1)
            plt.imshow(query, cmap=plt.cm.bone)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.xlabel(f'Label: {name}\nPrediction: {prediction}')
            if i == 0:
                plt.title('query')

            for k in range(n_neighbors):
                nn_idx = nn_indice[idx, k]
                dist = nn_dist[idx, k]
                value = X_train_image[nn_idx]
                name = names[y_train[nn_idx]]
                
                plt.subplot(1, n_neighbors + 1, k + 2)
                plt.imshow(value, cmap=plt.cm.bone)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.xlabel(f'Label: {name}\nDistance: {dist:0.2f}')
            plt.tight_layout()
            if i == 0:
                if train:
                    plt.suptitle(f'k nearest neighbors of queries from the training dataset', fontsize=30)
                    train = False
                else:
                    plt.suptitle(f'k nearest neighbors of queries from the test dataset', fontsize=30)
        
       




