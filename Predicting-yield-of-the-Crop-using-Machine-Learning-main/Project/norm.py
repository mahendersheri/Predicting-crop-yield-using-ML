# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:14:13 2020

@author: MAHENDER
"""

from sklearn import preprocessing
import numpy as np
# Get dataset
df = pd.read_csv("E:\prediction\prd\pm.csv", sep=",")
# Normalize total_bedrooms column
x_array = np.array(df['PRODUCTION'])
normalized_X = preprocessing.normalize([x_array])




from sklearn import preprocessing
# Get column names first
names = df.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)



df1=pd.DataFrame(data=scaled_df,columns=["AREA","RAINFALL","AVG_TEMPERATURE","HUMIDITY","GROUND_WATERLEVEL","PRODUCTION"])
df1.to_csv('normdrychillies.csv',index=False)
