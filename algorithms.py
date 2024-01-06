from collections import Counter
import numpy as np
from itertools import combinations
from itertools import permutations
from sklearn.model_selection import train_test_split
import pandas as pd
import random

################################################################################### APPRIORI ALGORITHM ##############################################################################

from itertools import combinations

def generate_CK(k,L):
    if k==1:
        CK = L
        return CK
    
    items=[item for item, support in L]

    if k==2:
        CK = list(set(combinations(items, 2)))
        return CK
    
    CK=[]
    for i in range(len(items)-1):
        for j in range(i+1,len(items)):
            item=list(set(items[i]) | set(items[j]))
            combins = list(set(combinations(item,k-1)))
            c=True
            if item not in CK:
                for comb in combins:
                    if comb not in items:
                        c=False
                        break
                if c:
                    CK.append(item)
    return CK
            
    
    
def calculate_support(dataitems,Ck, k):
    supports = [0] * len(Ck)
    
    if k == 1:
        for i, item in enumerate(Ck):
            for index, row in dataitems.iterrows():
                if item in row['Items']:
                    supports[i] += 1
    else:
        for i, item in enumerate(Ck):
            for index, row in dataitems.iterrows():
                if all(i in row['Items'] for i in item):
                    supports[i] += 1
    
    
    # Create a dictionary where keys are itemsets and values are the support counts
    support_dict = {tuple(itemset): support for itemset, support in zip(Ck, supports)}
    
    return support_dict



def generate_Lk(df,Ck,k, supp_min):
    Lk = []
    support = calculate_support(df,Ck,k)
    for item, support in support.items():
        if support >= supp_min:
            Lk.append((item,support))
    return Lk

def apriori(df,supp_min):
    all_items=[item for items in df['Items'].values for item in items]
    Lk=list(set(all_items))

    k=1
    #supp_min=3
    Ck=[]
    frequent_pattern=[]
    
    while(Lk):
        Ck=generate_CK(k,Lk)
        Lk=generate_Lk(df,Ck,k,supp_min)
        frequent_pattern+=[item for item in Lk]
        k+=1
    return frequent_pattern



################################################################### ASSOCIATION RULES EXTRACTION ############################################################################
def confiance(item1,item2,data):
    P_A_B = 0
    P_A = 0
    for index, row in data.iterrows():
        if all(i in row['Items'] for i in item1) and all(i in row['Items'] for i in item2):
            P_A_B += 1
        if all(i in row['Items'] for i in item1):
            P_A += 1

    return round((P_A_B / P_A),2) if P_A != 0 else 0

def generate_association_rules(FP,conf_min,dataitems):
    
    frequent_2=[]
    for itemset,support in FP:
        if len(itemset) >= 2 and isinstance(itemset[0],tuple):
            frequent_2.append((itemset,support))
    #print(frequent_2)

    association_rules = []

    for itemset,support in frequent_2:
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                consequent = tuple(set(itemset) - set(antecedent))
                conf=confiance(antecedent,consequent,dataitems)
                if(conf>=conf_min):
                    association_rules.append((set(antecedent), set(consequent),support,conf))

    return association_rules

def recommandation_soil(observation,data):
    recommandations=[]
    for index, row in data.iterrows():
        if observation in row['Items']:
            recommandations.append(row['Transaction'])
        if len(recommandations)>0:
            return random.choice(list(set(recommandations)))
        else :
            return "No recommandation"



################################################# DECISON TREE ####################################################

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats  # no. of features to consider
        self.root = None
    
    def get_params(self, deep=True):
        return {'min_samples_split': self.min_samples_split, 'max_depth': self.max_depth, 'n_feats': self.n_feats}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_thresh, best_feature = self._best_split(X, y, feat_idxs)
        #create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        # grow the children that result from the split
        left_child = self._grow_tree(X[left_idxs,:], y[left_idxs], depth=depth+1)
        right_child = self._grow_tree(X[right_idxs,:], y[right_idxs], depth=depth+1)
        return Node(best_feature, best_thresh, left_child, right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_thresh, split_idx
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    

################################################# RANDOM FORESTS ##############################################
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_features
        self.trees = []

    def get_params(self, deep=True):
        return {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'n_features': self.n_features
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_feats=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
############################################### KNN ##############################################
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3,dist_type='euclidean'):
        self.k = k
        self.dist_type= dist_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [calc_distance(x, x_train,self.dist_type) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
#Clustering 
    
############################################## K_means #################################################

def calc_distance(instance_1,instance_2,dist_type):
    if(dist_type == 'euclidean'):
        return np.sqrt(sum([(float(x)-float(y))**2 for x,y in zip(instance_1,instance_2)]))
    elif(dist_type == 'manhattan'):
        return sum([abs(float(x)-float(y)) for x,y in zip(instance_1,instance_2)])
    elif(dist_type == 'minkowski'):
        return pow(sum([pow(abs(float(x)-float(y)),3) for x,y in zip(instance_1,instance_2)]),1/3)
    elif(dist_type=='cosine'):
        return 1-(np.dot(instance_1,instance_2)/(np.sqrt(np.sum(np.power(instance_1,2)))*np.sqrt(np.sum(np.power(instance_2,2)))))
class K_means:
    def __init__(self,K=2,nb_iterations=10,dist_type='minkowski'):
        self.K=K
        self.nb_iterations=nb_iterations
        self.distance= dist_type
        self.labels_=[]
        self.centroids_=[]
    def calculer_centroide(self,cluster):
        moy=[attribut for attribut in cluster[0]]

        for instance in cluster[1:]:
            for i,attribut in enumerate(instance):
                moy[i]+=attribut

        moy=[elem/len(cluster) for elem in moy]
        return moy

    def cluster_instance(self,instance,clusters):
        distances=[]
        for i, cluster in clusters.items():
            centroid=clusters[i].get('centroid')
            distances.append((i,calc_distance(instance,centroid,self.distance)))

        cluster = min(distances, key=lambda x: x[1])[0]
        return cluster

    def check_non_equality(self,prec_centroids):
        for i in range(len(prec_centroids)):
            if(np.array_equal(prec_centroids[i],self.centroids_[i])):
                return False    
        return True

    def fit(self,dataset):
        data=dataset.copy()
        self.centroids_=[data.iloc[i].values for i in np.random.choice(len(data),self.K, replace=False)]

        clusters={}
        for i in range(self.K):
            clusters[i]={'centroid':self.centroids_[i],'instances':[]}

        prec_centroids=[]
        #new_centroids=self.centroids
        while(self.check_non_equality(prec_centroids) and self.nb_iterations>0):
            prec_centroids=self.centroids_
            for instance in data.values:
                cluster=self.cluster_instance(instance,clusters)
                clusters[cluster].get('instances').append(instance)
            self.centroids_=[0]*self.K
            for i,cluster in clusters.items():
                self.centroids_[i]=(self.calculer_centroide(clusters[i].get('instances')))
            self.nb_iterations-=1

        self.labels_=[None]*len(data)
        data['Cluster']=None
        for i,cluster in clusters.items():
            instances=clusters[i].get('instances')

            for index,row in enumerate(data.iloc[:,:-1].values):
                if any(np.array_equal(row, instance) for instance in instances):
                    data.at[index,'Cluster']=i
                    self.labels_[index]=i
        return data


############################################## DBSCAN #################################################


class DBSCAN:   
    def __init__(self, eps=0.5, Minpts=5):
        self.eps = eps
        self.Minpts = Minpts
        self.labels_ = []
        self.distances_=[]

    def get_params(self, deep=True):
        return {
            'eps': self.eps,
            'Minpts': self.Minpts,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def epsilonVoisinage(self,data, point, eps):
        voisinage=[]
        for instance in data.values:
            distance=calc_distance(point,instance,'euclidean')
            if distance !=0:
                self.distances_.append(distance)
            if distance<=eps:
                voisinage.append(instance)
        return voisinage

    def etendreCluster(self,data, point, voisinage, C, eps, Minpts, visite,clusters):
        if C not in clusters.keys():
            clusters[C]=[]
        clusters[C].append((point))
        for ptsvois in voisinage:
            exist=False 
            if not any(np.array_equal(ptsvois, p) for p in visite):
                visite.append(ptsvois)
                vois=self.epsilonVoisinage(data,ptsvois,eps)

                if len(vois)>=Minpts:
                    voisinage.extend(vois)
            for clusters_values in clusters.values():
                    if any(np.array_equal(ptsvois, p) for p in clusters_values):
                        exist=True
                        break
            if (not exist):
                clusters[C].append(ptsvois)


    def fit(self,data):
        data=data.copy()
        clusters={}
        C=-1
        visite=[]
        Bruit=[]
        for point in data.values:
            if not any(np.array_equal(point, p) for p in visite):
                visite.append(point)
                voisinage=self.epsilonVoisinage(data, point, self.eps)

                if len(voisinage)<self.Minpts:
                    Bruit.append(point)
                else:
                    C+=1
                    self.etendreCluster(data, point, voisinage, C, self.eps, self.Minpts, visite,clusters)
        data['Cluster']=None
        self.labels_=[None]*len(data)

        for index,row in enumerate(data.iloc[:,:-1].values):
            for i,instances in clusters.items():
                for instance in instances:
                    if np.array_equal(row, instance):
                        data.at[index,'Cluster']=i
                        self.labels_[index]=i
                        
            if any(np.array_equal(row, b) for b in Bruit):    
                data.at[index,'Cluster']=-1
                self.labels_[index]=-1
        return data
    

#########################################" Utility Functions ##############################################
def split_train_test(data):
    # List of unique class labels
    unique_classes = data['Fertility'].unique()

    # Dictionary to store train and test sets for each class
    class_splits = {}

    # Loop through each class
    for class_label in unique_classes:
        # Extract rows corresponding to the current class
        class_data = data[data['Fertility'] == class_label]

        # Split the class data_discretise into train and test sets
        X, y = class_data.iloc[:, 1:].values, class_data.iloc[:, 0].values
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store the splits in the dictionary
        class_splits[class_label] = {'X_train': X_train_class, 'X_test': X_test_class, 'y_train': y_train_class, 'y_test': y_test_class}
    # Concatenate the train and test sets 
    X_train = np.concatenate([class_splits[label]['X_train'] for label in unique_classes], axis=0)
    X_test = np.concatenate([class_splits[label]['X_test'] for label in unique_classes], axis=0)
    y_train= np.concatenate([class_splits[label]['y_train'] for label in unique_classes], axis=0)
    y_test= np.concatenate([class_splits[label]['y_test'] for label in unique_classes], axis=0)

    return X_train, X_test, y_train, y_test

from sklearn.decomposition import PCA
def data_to_data_2d(data):
    # Extract the features from your dataset
    features = data.iloc[:, :].values  # Exclude the target variable

    # Standardize the features (optional but recommended for PCA)
    from sklearn.preprocessing import StandardScaler
    features_standardized = StandardScaler().fit_transform(features)

    # Apply PCA to reduce the features to 2 components
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_standardized)

    # Create a new DataFrame with the reduced features
    data_2d = pd.DataFrame(data=features_2d, columns=['PC1', 'PC2'])
    return data_2d