import numpy as np
from algorithms import calc_distance

######################################## Confusion Matrix ########################################
def confusion_matrix(y_true,y_pred):
    classes=np.unique(y_true)
    matrix=np.zeros((len(classes),len(classes)))
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]]+=1
    return matrix

def find_TP(c, matrix):
   # counts the number of true positives (y_true = 1, y_pred = 1) 
   return matrix[c][c]
def find_FN(c, matrix):
   # counts the number of false negatives (y_true = 1, y_pred = 0)
   return sum(matrix[c,:]) - matrix[c,c]
def find_FP(c, matrix):
   # counts the number of false positives (y_true = 0, y_pred = 1)
   return sum(matrix[:,c]) - matrix[c,c]
def find_TN(c, matrix):
   # counts the number of true negatives (y_true = 0, y_pred = 0)
   return sum(sum(matrix)) - find_TP(c, matrix) - find_FN(c, matrix) - find_FP(c, matrix)

######################################### Accuracy ########################################
def accuracy(y_true,y_pred,c=None):
    matrix=confusion_matrix(y_true,y_pred)
    if c==None:
        global_accuracy=0
        for i in range(len(matrix)):
            global_accuracy+=matrix[i][i]
        return global_accuracy/np.sum(matrix)
    else:
        TP=find_TP(c,matrix)
        TN=np.sum(matrix)-np.trace(matrix)-find_FN(c,matrix)-find_FP(c,matrix)
        FN=find_FN(c,matrix)
        if(TP+FN==0):
            return 0
        return (TP+TN)/(TP+FN)

######################################## Specificity ########################################
def specificity(y_true,y_pred,c=None):
    matrix=confusion_matrix(y_true,y_pred)
    classes=np.unique(y_true)
    if c is None:
        global_specifity=0
        for i in classes:
            TN=find_TN(i,matrix)
            FP=find_FP(i,matrix)
            if TN+FP==0:
                specifity=0
            else:
                specifity=TN/(TN+FP)
            global_specifity+=(specifity)

        return global_specifity/len(classes)


    else:
        TN=find_TN(c,matrix)
        FP=find_FP(c,matrix)
        return TN/(TN+FP)

######################################## Precision ########################################    
def precision(y_true,y_pred,c=None):
    matrix=confusion_matrix(y_true,y_pred)
    classes=np.unique(y_true)
    if c is None:
        global_precision=0
        for i in classes:
            TP=find_TP(i,matrix)
            FP=find_FP(i,matrix)
            if TP+FP==0:
                precision=0
            else:
                precision=TP/(TP+FP)

            global_precision+=(precision)

        return global_precision/len(classes)
    else:
        TP=find_TP(c,matrix)
        FP=find_FP(c,matrix)

        if TP+FP==0:
            return 0
        return TP/(TP+FP)

######################################## Recall ########################################
def recall(y_true,y_pred,c=None):
    matrix=confusion_matrix(y_true,y_pred)
    classes=np.unique(y_true)
    if c is None:
        global_recall=0
        for i in classes:
            TP=find_TP(i,matrix)
            FN=find_FN(i,matrix)
            if TP+FN==0:
                recall=0
            else:
                recall=TP/(TP+FN)
            global_recall+=recall

        return global_recall/len(classes)
    else:
        TP=find_TP(c,matrix)
        FN=find_FN(c,matrix)
        
        if TP+FN==0:
            return 0
        return TP/(TP+FN)

######################################## F-Score ######################################## 
def f_score(y_true,y_pred,c=None):
    matrix=confusion_matrix(y_true,y_pred)
    classes=np.unique(y_true)
    if c==None:
        global_fscore=0
        for i in classes:
            p=precision(y_true,y_pred,i)
            r=recall(y_true,y_pred,i)
            if(p==0 or r==0):
                fscore=0
            else:
                fscore=2/((1/p)+(1/r))
            global_fscore+=(fscore)
        return global_fscore/len(classes)

    else:
        p=precision(y_true,y_pred,c)
        r=recall(y_true,y_pred,c)
        if (p==0 or r==0):
            return 0

        return 2/((1/p)+(1/r))
########################################### Silhouette ##############################################
def dist_groupe(point,groupe):
    if len(groupe)==1:
        return 0
    distance=0
    for instance in groupe:
        if not np.array_equal(instance,point):
            distance+=(calc_distance(point,instance,'minkowski'))
    a=distance/(len(groupe)-1)
    return a

def dist_groupe_voisine(point,groupes):
    distance=0
    min = float('inf')
    for groupe in groupes:
        for instance in groupe:
            distance+=(calc_distance(point,instance,'minkowski'))
        b=distance/(len(groupe))
        if b<min:
            min=b
    return min

def silhouette_point(point,groupe,groupes):
    a=dist_groupe(point,groupe)
    b=dist_groupe_voisine(point,groupes)
    Si=(b-a)/max(a,b)
    return Si

def calculate_silhouette(data):
    clusters={}
    labels=data['Cluster'].unique()
    labels = labels[labels != -1]
    for label in labels:
        instances=[row for i,row in enumerate(data.iloc[:,:-1].values) if data.at[i,'Cluster']==label]
        clusters[label]=instances
    Silhouette=0
    S_avg={}
    for key,values in clusters.items():
        groupe=values
        groupes=[val for k,val in clusters.items() if k != key]
        Si_avg=0
        for instance in groupe:
            Si_avg+=silhouette_point(instance,groupe,groupes)
        Si_avg/=len(groupe)
        S_avg[key]=Si_avg
        Silhouette+=Si_avg
    Silhouette/=len(labels)

    return S_avg,Silhouette


from sklearn.metrics.pairwise import euclidean_distances
# using the maximal distance in a cluster
def intra_cluster(data,cluster):
    # Get the cluster labels assigned to the data points
    labels = cluster.labels_

    # Get the unique cluster labels
    unique_labels = np.unique(labels)
    # Calculate the intra-cluster distances
    
    intra_cluster_distances = []
    for label in unique_labels:
        # Get the indices of the data points in the current cluster
        indices = np.where(labels == label)[0]
        # Get the data points in the current cluster
        cluster_data = [data.iloc[indice,:] for indice in indices]
        # Calculate the pairwise distances between the data points in the current cluster
        distances = euclidean_distances(cluster_data)
        # Calculate the sum of the pairwise distances
        max_distances = np.max(distances)
        # Calculate the intra-cluster distance for the current cluster
        intra_cluster_distance = max_distances
        # Add the intra-cluster distance to the list
        intra_cluster_distances.append(intra_cluster_distance)

    # Calculate the average intra-cluster distance
    min_intra_cluster_distance = np.min(intra_cluster_distances)
    return min_intra_cluster_distance # the minimum between maximums

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
# Using the distance between the mean of clusters
def inter_cluster(data,cluster):   
    # Get the cluster labels assigned to the data points
    labels = cluster.labels_

    # Get the unique cluster labels
    unique_labels = np.unique(labels)

    # Get the cluster centroids
    centroids = np.array([np.mean(data[labels == label], axis=0) for label in unique_labels])

    # Calculate the inter-cluster distances
    inter_cluster_distances = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                # Calculate the pairwise distances between centroids of different clusters
                distances = pairwise_distances([centroids[i]], [centroids[j]], metric='euclidean')
                inter_cluster_distances.append(distances[0][0])

    # Calculate the average inter-cluster distance
    average_inter_cluster_distance = np.mean(inter_cluster_distances)
    return average_inter_cluster_distance