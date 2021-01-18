#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_transformation import cus_dataset as dataset


#Reading the Dataset
dataset = dataset.drop(['state'], axis=1)

#Reading the Dataset
X = dataset.iloc[:, 1:].values

#Encoding the Dependant Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])

#Use elbow to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss, color="red")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Train K Means Cluster Model on the Dataset
kmeans = KMeans(n_clusters = 4, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualise the results
results = np.concatenate((X, y_kmeans.reshape(-1, 1)), axis=1)
y_cluster1 = X[y_kmeans == 0, :]
y_cluster2 = X[y_kmeans == 1, :]
y_cluster3 = X[y_kmeans == 2, :]
y_cluster4 = X[y_kmeans == 3, :]
cluster_centers = kmeans.cluster_centers_