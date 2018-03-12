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

import pickle as pkl

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
			df['Size'].fillna(value="OTHER", inplace = True)
		elif c == 'TopThreeAmericanName':
			df['TopThreeAmericanName'].fillna(value="OTHER", inplace=True)
		elif c == 'WheelType':
			df['WheelType'].fillna(value="OTHER", inplace=True)
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
	df = dropAllNA_or_dropcol(input_df, drop_col=['RefId','PurchDate','WheelTypeID','IsOnlineSale',
		'Transmission','Nationality','Color','PRIMEUNIT','AUCGUART','Model','SubModel','Trim',
		'VNZIP1','BYRNO', 'VNST'])
	df = impute_missing(df)
	df = pd.get_dummies(df)
	return df

def preprocess2(df, train = True):
	# 1) Chenge PurchDate into different date elements
	ts = pd.to_datetime(df['PurchDate'])
	df['PurchYear'] = ts.dt.year
	df['PurchMonth'] = ts.dt.month
	df['PurchQuarter'] = ts.dt.quarter
	df['PurchDayofweek'] = ts.dt.dayofweek

	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	num_cols = df.select_dtypes(numerics).columns.values
	non_num_cols = df.select_dtypes(exclude=numerics).columns.values

	# imputation
	if train:
		imputer = {}
		for c in df.columns:
			if df[c].dtype in numerics:
				df[c].fillna(value = df[c].median(), inplace = True)
				imputer[c] = df[c].median()
			else:
				df[c].fillna(value='OTHER', inplace = True)
				imputer[c] = 'OTHER'
		# for c in df.columns:
		# 	if c == 'MMRAcquisitionAuctionAveragePrice':
		# 		df['MMRAcquisitionAuctionAveragePrice'].fillna(value=df['MMRAcquisitionAuctionAveragePrice'].median(), inplace=True)
		# 		imputer['MMRAcquisitionAuctionAveragePrice'] = df['MMRAcquisitionAuctionAveragePrice'].median()
		# 	elif c == 'MMRAcquisitionAuctionCleanPrice':
		# 		df['MMRAcquisitionAuctionCleanPrice'].fillna(value=df['MMRAcquisitionAuctionCleanPrice'].median(), inplace=True)
		# 		imputer['MMRAcquisitionAuctionCleanPrice'] = df['MMRAcquisitionAuctionCleanPrice'].median()
		# 	elif c == 'MMRAcquisitionRetailAveragePrice':
		# 		df['MMRAcquisitionRetailAveragePrice'].fillna(value=df['MMRAcquisitionRetailAveragePrice'].median(), inplace=True)
		# 		imputer['MMRAcquisitionRetailAveragePrice'] = df['MMRAcquisitionRetailAveragePrice'].median()
		# 	elif c == 'MMRAcquisitonRetailCleanPrice':
		# 		df['MMRAcquisitonRetailCleanPrice'].fillna(value=df['MMRAcquisitonRetailCleanPrice'].median(), inplace=True)
		# 		imputer['MMRAcquisitonRetailCleanPrice'] = df['MMRAcquisitonRetailCleanPrice'].median()
		# 	elif c == 'MMRCurrentAuctionAveragePrice':
		# 		df['MMRCurrentAuctionAveragePrice'].fillna(value=df['MMRCurrentAuctionAveragePrice'].median(), inplace=True)
		# 		imputer['MMRCurrentAuctionAveragePrice'] = df['MMRCurrentAuctionAveragePrice'].median()
		# 	elif c == 'MMRCurrentAuctionCleanPrice':
		# 		df['MMRCurrentAuctionCleanPrice'].fillna(value=df['MMRCurrentAuctionCleanPrice'].median(), inplace=True)
		# 		imputer['MMRCurrentAuctionCleanPrice'] = df['MMRCurrentAuctionCleanPrice'].median()
		# 	elif c == 'MMRCurrentRetailAveragePrice':
		# 		df['MMRCurrentRetailAveragePrice'].fillna(value=df['MMRCurrentRetailAveragePrice'].median(), inplace=True)
		# 		imputer['MMRCurrentRetailAveragePrice'] = df['MMRCurrentRetailAveragePrice'].median()
		# 	elif c == 'MMRCurrentRetailCleanPrice':
		# 		df['MMRCurrentRetailCleanPrice'].fillna(value=df['MMRCurrentRetailCleanPrice'].median(), inplace=True)
		# 		imputer['MMRCurrentRetailCleanPrice'] = df['MMRCurrentRetailCleanPrice'].median()
		# 	elif 
			# elif c == 'Size':
			# # 	df['Size'].fillna(value="OTHER", inplace = True)
			# # 	imputer['Size'] = "OTHER"
			# # elif c == 'TopThreeAmericanName':
			# # 	df['TopThreeAmericanName'].fillna(value="OTHER", inplace=True)
			# # 	imputer['TopThreeAmericanName'] = "OTHER"
			# # elif c == 'WheelType':
			# # 	df['WheelType'].fillna(value="OTHER", inplace=True)
			# # 	imputer['WheelType'] = "OTHER"

		cat_dict = {}
		#2) Process VNZIP1
		# top50_ZIP = list(df.VNZIP1.value_counts()[:30].index)
		# df['VNZIP1'] = df['VNZIP1'].apply(lambda x: str(x) if x in top50_ZIP else 'OTHER')
		# cat_dict['VNZIP1'] = top50_ZIP
		# top15_State = list(df.VNST.value_counts()[:15].index)
		# df['VNST'] = df['VNST'].apply(lambda x: x if x in top15_State else 'OTHER')
		# cat_dict['VNST'] = top15_State

		# 3) process Color
		# top5_color = list(df.Color.value_counts()[:5].index)
		# df['Color'] = df['Color'].apply(lambda x: x if x in top5_color else 'OTHER')
		# cat_dict['Color'] = top5_color

		# 4) Process Make
		# top20_make = list(df.Make.value_counts()[:5].index)
		# df['Make'] = df['Make'].apply(lambda x: x if x in top20_make else 'OTHER')
		# cat_dict['Make'] = top20_make

		# 5) process Trim
		# top10_Trim = list(df.Trim.value_counts()[:5].index)
		# df['Trim'] = df['Trim'].apply(lambda x: x if x in top10_Trim else 'OTHER')
		# cat_dict['Trim'] = top10_Trim

		# process Size
		# df['Size'] = df['Size'].apply(lambda x: x.split()[0])
		# cat_dict['Size'] = list(df['Size'].value_counts().index)

		# 6) Process Model
		# top20_Model = list(df.Model.value_counts()[:10].index)
		# df['Model'] = df['Model'].apply(lambda x: x if x in top20_Model else 'OTHER')
		# cat_dict['Model'] = top20_Model

		# 7) Transmission
		# df['Transmission'] = df['Transmission'].str.upper()
		# cat_dict['Transmission'] = list(df.Transmission.value_counts().index)

		with open('imputer.pkl', 'wb') as fout:
			pkl.dump(imputer, fout)
		with open('cat_dict.pkl', 'wb') as fout:
			pkl.dump(cat_dict, fout)

	else:
		with open('imputer.pkl', 'rb') as fin:
			imputer = pkl.load(fin)

		with open('cat_dict.pkl', 'rb') as fin:
			cat_dict = pkl.load(fin)

		for c in df.columns:
			if c in imputer:
				df[c].fillna(value=imputer[c], inplace = True)
			if c in cat_dict:
				if c == 'VNZIP1':
					df[c] = df[c].apply(lambda x: str(x) if x in cat_dict[c] else 'OTHER')
				else:
					df[c] = df[c].apply(lambda x: x if x in cat_dict[c] else 'OTHER')

	# change to category
	# df['WheelType'] = df['WheelType'].astype('category')
	# df['VNZIP1'] = df['VNZIP1'].astype('category')
	# df['VNST']= df['VNST'].astype('category')
	# df['Color'] = df['Color'].astype('category')
	# df['Make'] = df['Make'].astype('category')
	# df['Model'] = df['Model'].astype('category')
	# df['Trim'] = df['Trim'].astype('category')
	# df['Nationality'] = df['Nationality'].astype('category')
	# df['Size'] = df['Size'].astype('category')
	# df['TopThreeAmericanName'] = df['TopThreeAmericanName'].astype('category')
	df['IsOnlineSale'] = df['IsOnlineSale'].astype('category')
	df['Auction'] = df['Auction'].astype('category')
	df['VehYear'] = df['VehYear'].astype('category')
	# df['Transmission'] = df['Transmission'].astype('category')
	df['PurchYear'] = df['PurchYear'].astype('category')
	df['PurchMonth'] = df['PurchMonth'].astype('category')
	df['PurchQuarter'] = df['PurchQuarter'].astype('category')
	df['PurchDayofweek'] = df['PurchDayofweek'].astype('category')

	# drop unneeded cols
	df.drop(['RefId', 'BYRNO', 'PRIMEUNIT', 'AUCGUART','SubModel', 'WheelTypeID', 'VNZIP1', 'PurchDate', 'VehicleAge','Model'], axis=1, inplace=True)