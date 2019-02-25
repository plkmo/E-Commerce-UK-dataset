# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 08:50:56 2018

@author: WT
"""
from src.cluster_plots import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors

dff = pd.read_csv("inter.csv")    
X = clean_engineered_df(dff)

#X = do_PCA(X,pca_n=3)

nb = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
distances, indices = nb.kneighbors(X)
distances = [d[-1] for d in distances]
distances.sort()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.scatter([i for i in range(len(distances))], distances)
ax.set_ylabel("KNN distance")
ax.set_xlabel("points")
ax.set_title("KNN plot")

db = DBSCAN(eps=1, min_samples=4)
y_hc = db.fit_predict(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Number of clusters: %d' % n_clusters_)
print('Number of noise points: %d' % n_noise_)
cluster_info(dff,y_hc)