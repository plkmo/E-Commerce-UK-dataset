# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:17:11 2018

@author: WT
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

####### cleans the ecommerce dataset and outputs a dataframe
def clean_df(df):
    #Change the format for the invoice date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")

    ## Replace NaN values in Description with NaN string
    df["Description"].fillna("NaN", inplace=True)

    ## Drop all entries whose UnitPrice and Quantity are negative
    miss_bool = df["Quantity"]<0
    df.loc[miss_bool,"Quantity"] = df.loc[miss_bool,"Quantity"].apply(lambda x: None)
    miss_bool = df["UnitPrice"]<0
    df.loc[miss_bool,"UnitPrice"] = df.loc[miss_bool,"UnitPrice"].apply(lambda x: None)
    
    ### Drop all duplicates if any
    df.drop_duplicates(inplace=True)
    
    ##### Remove all missing CustomerID values
    df.dropna(inplace=True)
    
    #### Remove outliers in Quantity, UnitPrice
    df = df[((df["Quantity"] - df["Quantity"].mean()) / df["Quantity"].std()).abs() < 3]
    df = df[((df["UnitPrice"] - df["UnitPrice"].mean()) / df["UnitPrice"].std()).abs() < 3]
    return df

#### Given cleaned dataframe df, converts it into a stock table for association rules analysis, then
#### saves it
def make_stock_table(df):
    stocktable = pd.pivot_table(df, values="Quantity",index="InvoiceNo",columns="StockCode",aggfunc=np.sum,\
                            fill_value=0)
    stocktable[stocktable!=0] = 1
    ### convert to booleans
    stocktable = stocktable.astype(bool)
    stocktable.to_csv("stocktable.csv", index=True)
    return stocktable

#### get partitioned dataframes according to cluster labels, given cust_id_labels and df (data)
def get_df_for_alllabels(cust_id_labels, df):
        dfs = []
        for label in list(cust_id_labels["labels"].unique()):
            custIDs = list(cust_id_labels[cust_id_labels["labels"]==label]["CustomerID"])
            dfs.append(df.loc[df["CustomerID"].isin(custIDs)])
        return dfs

######### for each dataframe, obtain item_sets and association rules
def get_rules(df):
    stocktable = pd.pivot_table(df, values="Quantity",index="InvoiceNo",columns="StockCode",aggfunc=np.sum,\
                            fill_value=0)
    stocktable[stocktable!=0] = 1
    stocktable = stocktable.astype(bool)
    #### get frequent_itemsets given min_support
    frequent_itemsets = apriori(stocktable, min_support=0.02, use_colnames=True)
    ##### get association rules given min lift threshold
    min_threshold = 1.2
    a_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    return frequent_itemsets, a_rules