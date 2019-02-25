# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:41:49 2018

@author: WT
"""

from src.clean_n_engineer import *
from src.cluster_plots import *
from src.assoc_funcs import *
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabaz_score

df = pd.read_csv('data/raw/data.csv', encoding='ISO-8859-1')
## check for NaN data
print("NaN Data:\n%s" % df.apply(lambda x: sum(x.isnull())))

#general statistics of data
print(df.describe())

#See how many categories are there in each categorical columns
categorical_columns = [x for x in df.dtypes.index if df.dtypes[x]=="object"]
for col in categorical_columns:
    print(f"\nFrequency of Categories for variable {col}: ")
    print(df[col].value_counts())
    
## Check negative UnitPrice and Quantity entries
print("No. of negative UnitPrice entries: ", len(df.loc[df["UnitPrice"]<0]))
print("No. of negative Quantity entries: ", len(df.loc[df["Quantity"]<0]))
sss = df.loc[df["Quantity"]<0]
print("Negative Quantity data:\n", sss.head())
print("No. of entries with InvoiceNo starting with C: ", len(df.loc[df["InvoiceNo"].str[0]=="C"]))
print("No. of entries with InvoiceNo starting with C whose Quantity is negative:",\
      len(sss.loc[sss["InvoiceNo"].str[0]=="C"]))

df = clean_df(df)
del df
#engineer_df(df)

##### Exploratory Data visualization
dff = pd.read_csv("inter.csv")
X = dff.drop(["CustomerID"],axis=1,inplace=False)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(data=X,columns=['NoOfInvoices', 'NoOfUniqueItems', \
                                       'QuantityPerInvoice', 'SpendingPerInvoice', 'TotalQuantity', \
                                       'UniqueItemsPerInvoice','UnitPriceMean','UnitPriceStd'])
### plot histograms of distributions
fig1, axes1 = plt.subplots(nrows=4, ncols=2,figsize=(13,25))
ax1 = axes1.flatten()
ax1[0].hist(X["NoOfInvoices"], bins=30, label="X")
ax1[0].set_ylabel("Count")
ax1[0].set_xlabel("NoOfInvoices")
ax1[0].legend(loc="upper right")
ax1[1].hist(X["NoOfUniqueItems"], bins=30, label="X")
ax1[1].set_ylabel("Count")
ax1[1].set_xlabel("NoOfUniqueItems")
ax1[1].legend(loc="upper right")
ax1[2].hist(X["QuantityPerInvoice"], bins=30, label="X")
ax1[2].set_ylabel("Count")
ax1[2].set_xlabel("QuantityPerInvoice")
ax1[2].legend(loc="upper right")
ax1[3].hist(X["SpendingPerInvoice"], bins=30, label="X")
ax1[3].set_ylabel("Count")
ax1[3].set_xlabel("SpendingPerInvoice")
ax1[3].legend(loc="upper right")
ax1[4].hist(X["TotalQuantity"], bins=30, label="X")
ax1[4].set_ylabel("Count")
ax1[4].set_xlabel("TotalQuantity")
ax1[4].legend(loc="upper right")
ax1[5].hist(X["UniqueItemsPerInvoice"], bins=30, label="X")
ax1[5].set_ylabel("Count")
ax1[5].set_xlabel("UniqueItemsPerInvoice")
ax1[5].legend(loc="upper right")
ax1[6].hist(X["UnitPriceMean"], bins=30, label="X")
ax1[6].set_ylabel("Count")
ax1[6].set_xlabel("UnitPriceMean")
ax1[6].legend(loc="upper right")
ax1[7].hist(X["UnitPriceStd"], bins=30, label="X")
ax1[7].set_ylabel("Count")
ax1[7].set_xlabel("UnitPriceStd")
ax1[7].legend(loc="upper right")

###### plot correlations
corr_matrix = X.corr()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(corr_matrix.columns.values)
ax.set_yticklabels(corr_matrix.columns.values)
plt.show()

######### plot raw scatters 
Xcols = list(X.columns)
Xcolspairs = []
dummy = []
for a in Xcols:
    for b in Xcols:
        if a!=b:
            if (a,b) not in dummy:
                Xcolspairs.append((a,b))
                dummy.append((b,a))
fig2, axes2 = plt.subplots(nrows=14, ncols=2,figsize=(13,88))
ax2 = axes2.flatten()
for idx, pair in enumerate(Xcolspairs):
    ax2[idx].scatter(X[f"{pair[0]}"],X[f"{pair[1]}"])
    ax2[idx].set_xlabel(f"{pair[0]}")
    ax2[idx].set_ylabel(f"{pair[1]}")

dff = pd.read_csv("inter.csv")    
X = clean_engineered_df(dff)

################### Dendograms for different linkages
plot_dendrograms(X)

######################### Elbow plots for different linkages
plot_elbows_agglo(X,n=15,links=["ward", "complete", "average"])

########## Do PCA
X = do_PCA(X,pca_n=3)
########### Elbow plot of silhouette score
plot_elbows_agglo(X,n=15,links=["ward"])

ac = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = ac.fit_predict(X)
s_score = silhouette_score(X,y_hc)
print("Silhouette Score: ", s_score)

######### plot clustered scatters 2D
pairplots(X,y_hc)

############### plot clustered scatters 3D
tripletplots(X,y_hc)

########## Check cluster properties ################
cluster_info(dff,y_hc)

########### Elbow plot of silhouette score for KMeans
plot_elbows_kmeans(X,n=15)

ac = KMeans(n_clusters=2, random_state=0)
y_hc = ac.fit_predict(X)
s_score = silhouette_score(X,y_hc)
print("Silhouette Score: ", s_score)

######### plot KMeans clustered scatters 2D
pairplots(X,y_hc)
########## Check cluster properties ################
cluster_info(dff,y_hc)

#################### GMM ###########################
########### Elbow plot of silhouette score for GMM
plot_elbows_GMM(X,n=15)

gmm = GaussianMixture(n_components=2, random_state=0)
y_hc = gmm.fit_predict(X)
s_score = silhouette_score(X,y_hc)
print("Silhouette Score: ", s_score)

######### plot GMM clustered scatters 2D
pairplots(X,y_hc)

cluster_info(dff,y_hc)

########### Elbow plot of various scores
plot_elbows_ac_diff_scores(X,n=15)

dff = pd.read_csv("inter.csv")
X = clean_engineered_df(dff)
################# Do PCA plot cum explained variance #########
PCA_cum_explained_ratio(X,pca_n=6)

################ PCA GMM ######################
dff = pd.read_csv("inter.csv")
X = clean_engineered_df(dff)
################# Do PCA plot  #########
do_PCA_plot2(X,pca_n=6)