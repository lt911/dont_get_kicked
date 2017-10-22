'''
Helper functions for anaysis, and data processing
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
import sklearn.metrics as skm
from collections import *
from IPython.display import display, HTML 

import warnings
warnings.filterwarnings('ignore')

def getStat(df):
	''' get stat summary of data'''
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	num_cols = df.select_dtypes(numerics).columns.values
	non_num_cols = df.select_dtypes(exclude=numerics).columns.values
	
	c_summ = []
	for c in num_cols:
		missing = sum(pd.isnull(df[c]))
		sumval = df[c].describe().drop(['count'])
		distinct = df[c].nunique()
		sumval = sumval.append(pd.Series([missing, distinct], index=['missing', 'distinct']))
		c_summ.append(sumval)
	
	for c in non_num_cols:
		missing = sum(pd.isnull(df[c]))
		sumval = df[c].describe().drop(['top', 'freq','count','unique'])
		distinct = df[c].nunique()
		sumval = sumval.append(pd.Series([missing, distinct], index=['missing','distinct']))
		c_summ.append(sumval)
	return pd.DataFrame(c_summ, index=np.append(num_cols, non_num_cols))





def dropAllNA_or_dropcol(df_input,dropall=False, drop_col= None):
	df = df_input.copy()
	if dropall:
		df.dropna(axis = 0, how = 'any', inplace = True)
	if drop_col:
		df.drop(drop_col ,1, inplace=True)
	return df




def impute_missing(input_df):
	df = input_df.copy()
	for c in df.columns:
		if c == 'MMRAcquisitionAuctionAveragePrice':
			df['MMRAcquisitionAuctionAveragePrice'].fillna(value=df['MMRAcquisitionAuctionAveragePrice'].median(), inplace=True)
		elif c == 'MMRAcquisitionAuctionCleanPrice':
			df['MMRAcquisitionAuctionCleanPrice'].fillna(value=df['MMRAcquisitionAuctionCleanPrice'].median(), inplace=True)
		elif c == 'MMRAcquisitionRetailAveragePrice':
			df['MMRAcquisitionRetailAveragePrice'].fillna(value=df['MMRAcquisitionRetailAveragePrice'].median(), inplace=True)
		elif c == 'MMRAcquisitonRetailCleanPrice':
			df['MMRAcquisitonRetailCleanPrice'].fillna(value=df['MMRAcquisitonRetailCleanPrice'].median(), inplace=True)
		elif c == 'MMRCurrentAuctionAveragePrice':
			df['MMRCurrentAuctionAveragePrice'].fillna(value=df['MMRCurrentAuctionAveragePrice'].median(), inplace=True)
		elif c == 'MMRCurrentAuctionCleanPrice':
			df['MMRCurrentAuctionCleanPrice'].fillna(value=df['MMRCurrentAuctionCleanPrice'].median(), inplace=True)
		elif c == 'MMRCurrentRetailAveragePrice':
			df['MMRCurrentRetailAveragePrice'].fillna(value=df['MMRCurrentRetailAveragePrice'].median(), inplace=True)
		elif c == 'MMRCurrentRetailCleanPrice':
			df['MMRCurrentRetailCleanPrice'].fillna(value=df['MMRCurrentRetailCleanPrice'].median(), inplace=True)
		elif c == 'Size':
			df['Size'].fillna(value="UNKNOWN", inplace = True)
		elif c == 'TopThreeAmericanName':
			df['TopThreeAmericanName'].fillna(value="OTHER", inplace=True)
		elif c == 'WheelType':
			df['WheelType'].fillna(value="UNKNOWN", inplace=True)
	return df





def get_MI(df, label, plot = False, df_display = True):
	X = df.drop([label], 1)
	cols = X.columns.values
	y = df[[label]].values
	
	MIs = []
	for c in cols:
		MIs.append(skm.normalized_mutual_info_score(y.ravel(), X[[c]].values.ravel()))
	
	MI_dict = OrderedDict(sorted(dict(zip(cols, MIs)).items(), key = lambda t: t[1], reverse=True))
	
	MI_df = pd.DataFrame(MI_dict.items(), columns = ['column', 'MI'])
	if df_display:
		display(MI_df)
	if plot:
		MI_df.plot(kind= 'bar',figsize=(12,6), x = MI_df.column.values)
#     return MI_df 



def preprocess(input_df):
	### drop unneeded columns
	df = dropAllNA_or_dropcol(input_df, drop_col=['RefId','PurchDate','WheelTypeID','IsOnlineSale','Transmission','Nationality','Color','PRIMEUNIT','AUCGUART','Model','SubModel','Trim','VNZIP1','BYRNO', 'VNST'])
	df = impute_missing(df)
	df = pd.get_dummies(df)
	return df
	
