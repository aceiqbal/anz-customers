#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_transformation import cus_dataset as dataset

dataset = dataset.drop(['state'], axis=1)

#Reading the Dataset
X = dataset.iloc[:, 1:].values

#Encoding the Dependant Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])

#Use dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Observation Points')
plt.ylabel('Euclidean Distance')
plt.show()


#Train Hierachical Cluster Model with Dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#Visualise the results
results = np.concatenate((X, y_hc.reshape(-1, 1)), axis=1) 
y_cluster1 = X[y_hc == 0, :]
y_cluster2 = X[y_hc == 1, :]
y_cluster3 = X[y_hc == 2, :]
y_cluster4 = X[y_hc == 3, :]
