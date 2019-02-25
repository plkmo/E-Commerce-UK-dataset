# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:39:19 2018

@author: WT
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, calinski_harabaz_score
from mpl_toolkits.mplot3d import Axes3D
##### Reads in engineered data for clustering and readies it for cluster modeling.
#### First we drop customerID, then scale the features, then drop correlated features TotalQuantity and 4
#### UniqueItemsPerInvoice
def clean_engineered_df(dff):
    X = dff.drop(["CustomerID"],\
             axis=1,inplace=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(data=X,columns=['NoOfInvoices', 'NoOfUniqueItems', \
                                       'QuantityPerInvoice', 'SpendingPerInvoice', 'TotalQuantity', \
                                       'UniqueItemsPerInvoice','UnitPriceMean','UnitPriceStd'])
    X = X.drop(["TotalQuantity","UniqueItemsPerInvoice",\
                ],axis=1,inplace=False)
    return X

########## Given dataset X, plots and compares the dendrograms for different linkages "ward", "complete","average"
def plot_dendrograms(X):
    ################### Dendograms for different linkages
    links = ["ward", "complete", "average"]
    fig2 = plt.figure(figsize=(13,10))

    for idx,link in enumerate(links):
        axes2 = fig2.add_subplot(int(f"13{idx+1}"))
        axes2.set_title(f"{link}")
        axes2.set_ylabel("Distance")
        axes2.set_xlabel("Features")
        Z = sch.linkage(X, method = f"{link}", metric='euclidean')
        den = sch.dendrogram(Z, truncate_mode='level',p=13)
        
####### Given dataset X, plots the elbow silhouette scores for different linkages, up to n number of clusters
####### AgglomerativeClustering is used
######################### Elbow plots for different linkages
def plot_elbows_agglo(X,n=15,links=["ward", "complete", "average"]):
    fig2 = plt.figure(figsize=(13,10))
    for idx,link in enumerate(links):
        scores = []
        for nn in range(2,n,1):
            ac = AgglomerativeClustering(n_clusters = nn, affinity = 'euclidean', linkage = f"{link}")
            y_hc = ac.fit_predict(X)
            scores.append(silhouette_score(X,y_hc))
        ax4 = fig2.add_subplot(int(f"13{idx+1}"))
        ax4.set_title(f"{link}")
        ax4.scatter([n for n in range(2,n,1)],scores)
        ax4.set_xlabel("n")
        ax4.set_ylabel("Silhouette Score")
    fig2.show()

####### Given dataset X, plots the elbow silhouette scores, up to n number of clusters
####### KMeansClustering is used
######################### Elbow plots
def plot_elbows_kmeans(X,n=15):
    fig2 = plt.figure(figsize=(13,10))
    scores = []
    for nn in range(2,n,1):
        ac = KMeans(n_clusters=nn, random_state=0)
        y_hc = ac.fit_predict(X)
        scores.append(silhouette_score(X,y_hc))
    ax4 = fig2.add_subplot(222)
    ax4.set_title(f"KMeans")
    ax4.scatter([n for n in range(2,n,1)],scores)
    ax4.set_xlabel("n")
    ax4.set_ylabel("Silhouette Score")
    fig2.show()
    
####### Given dataset X, plots the elbow silhouette scores, up to n number of clusters
####### GMM is used
######################### Elbow plots
def plot_elbows_GMM(X,n=15):
    fig2 = plt.figure(figsize=(13,10))
    scores = []
    for nn in range(2,n,1):
        ac = GaussianMixture(n_components=nn, random_state=0)
        y_hc = ac.fit_predict(X)
        scores.append(silhouette_score(X,y_hc))
    ax4 = fig2.add_subplot(222)
    ax4.set_title(f"GMM")
    ax4.scatter([n for n in range(2,n,1)],scores)
    ax4.set_xlabel("n")
    ax4.set_ylabel("Silhouette Score")
    fig2.show()
    
####### Given dataset X, plots the elbow silhouette, Davis Bouldin and Calinski harabaz scores, 
######## up to n number of clusters
####### AgglomerativeClustering is used
######################### Elbow plots
def plot_elbows_ac_diff_scores(X,n=15):
    scores = []
    db_scores = []
    ch_scores = []
    for nn in range(2,n,1):
        ac = AgglomerativeClustering(n_clusters = nn, affinity = 'euclidean', linkage = 'ward')
        y_hc = ac.fit_predict(X)
        scores.append(silhouette_score(X,y_hc))
        db_scores.append(davies_bouldin_score(X,y_hc))
        ch_scores.append(calinski_harabaz_score(X,y_hc))
    fig4 = plt.figure(figsize=(10,10))
    ax4 = fig4.add_subplot(111)
    ax4.scatter([n for n in range(2,n,1)],scores)
    ax4.set_xlabel("n")
    ax4.set_ylabel("Silhouette Score")
    fig4.show()
    fig5 = plt.figure(figsize=(10,10))
    ax5 = fig5.add_subplot(111)
    ax5.scatter([n for n in range(2,n,1)],db_scores)
    ax5.set_xlabel("n")
    ax5.set_ylabel("Davies Bouldin Score")
    fig5.show()
    fig6 = plt.figure(figsize=(10,10))
    ax6 = fig6.add_subplot(111)
    ax6.scatter([n for n in range(2,n,1)],ch_scores)
    ax6.set_xlabel("n")
    ax6.set_ylabel("Calinski harabaz Score")
    fig6.show()

###### Given dataset X, do PCA and transform X for n PCA components. Prints explained variance ratio
###### returns transformed X
def do_PCA(X,pca_n=3):
    pca = PCA(n_components=pca_n)
    X = pca.fit_transform(X)
    print("PCA explained variance ratio: ", pca.explained_variance_ratio_)
    X = pd.DataFrame(X, columns=[f"{i}" for i in range(pca_n)])
    return X

#### Plots all possible pairwise combinations of features, showing the clustering distribution of points by 
#### color. X is the features matrix and y_hc is the cluster labels
def pairplots(X,y_hc):
    ######### plot clustered scatters 2D
    Xcols = list(X.columns)
    Xcolspairs = []
    dummy = []
    for a in Xcols:
        for b in Xcols:
            if a!=b:
                if (a,b) not in dummy:
                    Xcolspairs.append((a,b))
                    dummy.append((b,a)); dummy.append((a,b))
    fig2, axes2 = plt.subplots(nrows=math.ceil(len(Xcolspairs)/2), ncols=2,figsize=(13,\
                               15*math.ceil(len(Xcolspairs)/2)))
    ax2 = axes2.flatten()
    for idx, pair in enumerate(Xcolspairs):
        ax2[idx].scatter(X[f"{pair[0]}"],X[f"{pair[1]}"],c=y_hc,cmap="rainbow")
        ax2[idx].set_xlabel(f"{pair[0]}")
        ax2[idx].set_ylabel(f"{pair[1]}")

#### Plots all possible triplet combinations of features, showing the clustering distribution of points by 
#### color. X is the features matrix and y_hc is the cluster labels
def tripletplots(X,y_hc):
    ############### plot clustered scatters 3D
    Xcols = list(X.columns)
    Xcoltriplets = []
    dummy = []
    for a in Xcols:
        for b in Xcols:
            for c in Xcols:
                if a!=b and b!=c:
                    if a!=b:
                        if a!=c:
                            if b!=c:
                                if (a,b,c) not in dummy:
                                    Xcoltriplets.append((a,b,c))
                                    dummy.append((c,b,a))
                                    dummy.append((c,a,b))
                                    dummy.append((b,c,a))
                                    dummy.append((a,c,b))
                                    dummy.append((b,a,c))
    fig2 = plt.figure(figsize=(25,25))
    axes2 = fig2.add_subplot(221,projection='3d')
    for idx, pair in enumerate(Xcoltriplets):
        axes2.scatter(X[f"{pair[0]}"],X[f"{pair[1]}"],X[f"{pair[2]}"],c=y_hc,cmap="rainbow")
        axes2.set_xlabel(f"{pair[0]}")
        axes2.set_ylabel(f"{pair[1]}")
        axes2.set_zlabel(f"{pair[2]}")

###### given cluster labels y_hc, get the cluster meta information from untransformed feature matrix dff 
##### and plots them
##### by cluster labels. Information includes average(av) and standard deviation(std) of SpendingPerInvoice,
##### NumOfInvoices, UnitPrice, TotalSpending. Also prints the number of points in each cluster
def cluster_info(dff,y_hc):
    ########## Check cluster properties ################
    av_spendingperinvoice = []
    std_spendingperinvoice = []
    av_num_invoices = []
    std_num_invoices = []
    av_total_spending = []
    std_total_spending = []
    av_unit_price = []
    std_unit_price = []
    cluster_size = []
    for label in np.unique(y_hc):
        label_bool = [i==label for i in y_hc]
        av_spendingperinvoice.append(math.log(dff[label_bool]["SpendingPerInvoice"].mean()))
        std_spendingperinvoice.append(math.log(dff[label_bool]["SpendingPerInvoice"].std()))
        av_num_invoices.append(dff[label_bool]["NoOfInvoices"].mean())
        std_num_invoices.append(dff[label_bool]["NoOfInvoices"].std())
        av_unit_price.append(dff[label_bool]["UnitPriceMean"].mean())
        std_unit_price.append(dff[label_bool]["UnitPriceStd"].mean())
        av_total_spending.append(math.log((dff[label_bool]["SpendingPerInvoice"]*dff[label_bool]["NoOfInvoices"]).mean()))
        std_total_spending.append(math.log((dff[label_bool]["SpendingPerInvoice"]*dff[label_bool]["NoOfInvoices"]).std()))
        cluster_size.append(len(dff[label_bool]))
    info = [av_spendingperinvoice,std_spendingperinvoice,av_num_invoices,std_num_invoices,av_total_spending,\
            std_total_spending, av_unit_price,std_unit_price]
    info_labels = ["av_spendingperinvoice","std_spendingperinvoice","av_num_invoices","std_num_invoices",\
                   "av_total_spending","std_total_spending","av_unit_price","std_unit_price"]
    fig2, axes2 = plt.subplots(nrows=4, ncols=2,figsize=(13,\
                               20))
    ax2 = axes2.flatten()
    for idx,i in enumerate(info):
        ax2[idx].scatter([a for a in range(len(i))],i)
        ax2[idx].set_xlabel("Cluster")
        if info_labels[idx] in ["av_spendingperinvoice","std_spendingperinvoice","av_total_spending",\
                      "std_total_spending"]:
            ax2[idx].set_ylabel("log(%s)" % info_labels[idx])
        else:
            ax2[idx].set_ylabel(f"{info_labels[idx]}")
    for idx,i in enumerate(cluster_size):
        print(f"No. of points in cluster {idx}: ", i)

############# Given feature matrix X, do PCA with n components to obtain plot of cumulative 
############# explained variance ratio with n
def PCA_cum_explained_ratio(X,pca_n=6):
    pca = PCA(n_components=pca_n)
    X = pca.fit_transform(X)
    print("PCA explained variance ratio: ", pca.explained_variance_ratio_)
    X = pd.DataFrame(X, columns=[f"{i}" for i in range(pca_n)])
    evr = pca.explained_variance_ratio_
    evr_cum = [evr[0:i+1].sum() for i in range(len(evr))]
    fig2 = plt.figure(figsize=(10,10))
    axes2 = fig2.add_subplot(221)
    axes2.scatter([i for i in range(len(evr))],evr_cum)
    axes2.set_title("Cumulative explained variance vs number of components")
    axes2.set_xlabel("Number of components")
    axes2.set_ylabel("Cumulative explained variance")
    
####### Given feature matrix X, do PCA with n components=7 and plot the first 2 highest variance transformed
####### features
def do_PCA_plot2(X,pca_n=7):
    pca = PCA(n_components=pca_n)
    X = pca.fit_transform(X)
    X = pd.DataFrame(X, columns=[f"{i}" for i in range(pca_n)])
    
    gmm = GaussianMixture(n_components=7, random_state=0)
    y_hc = gmm.fit_predict(X)
    
    ######### plot GMM clustered scatters 2D
    fig4 = plt.figure(figsize=(10,10))
    ax2 = fig4.add_subplot(111)
    ax2.scatter(X["0"],X["1"],c=y_hc,cmap="rainbow")
    ax2.set_xlabel(f"PCA 0")
    ax2.set_ylabel(f"PCA 1")