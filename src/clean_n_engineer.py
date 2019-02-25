# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:56:38 2018

@author: WT
"""
import pandas as pd
import numpy as np

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

###### Create engineering features for clustering, then saves into a inter.csv file
def engineer_df(df):
    cust_list = list(df["CustomerID"].unique().astype(int))
    num_invoices = []
    num_unique_items= []
    quantity_per_invoice = []
    spending_per_invoice = []
    total_quantity = []
    unique_items_per_invoice = []
    unit_price_mean = []
    unit_price_std = []
    for cust in cust_list:
        ss = df.loc[df["CustomerID"]== cust]
        u_invoices = ss["InvoiceNo"].unique()
        num_invoices.append(len(u_invoices))
        num_unique_items.append(len(ss["StockCode"].unique()))
        quantity_per_invoice.append(ss["Quantity"].sum()/len(u_invoices))
        spending_per_invoice.append((ss["Quantity"]*ss["UnitPrice"]).sum()/len(u_invoices))
        total_quantity.append(len(ss))
        unique_items_per_invoice.append(len(ss["StockCode"].unique())/len(u_invoices))
        unit_price_mean.append(ss["UnitPrice"].mean())
        unit_price_std.append(ss["UnitPrice"].std())
    data = [cust_list, num_invoices, num_unique_items, quantity_per_invoice, spending_per_invoice, total_quantity,\
            unique_items_per_invoice,unit_price_mean,unit_price_std]
    data = np.array(data).T
    data = pd.DataFrame(data=data,columns=['CustomerID', 'NoOfInvoices', 'NoOfUniqueItems', \
                                           'QuantityPerInvoice', 'SpendingPerInvoice', 'TotalQuantity', \
                                           'UniqueItemsPerInvoice','UnitPriceMean','UnitPriceStd'])
    data["UnitPriceStd"].fillna(0,inplace=True)
    data.to_csv("inter.csv", index=False)