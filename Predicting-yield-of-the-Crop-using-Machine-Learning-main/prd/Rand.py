# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:31:06 2020

@author: MAHENDER
"""

import rfpimp
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

######################################## Data preparation #########################################

file = 'E:\prediction\prd\normrice.csv'
df = pd.read_csv(file)
features = ['PRODUCTION','RAINFALL','AVG_TEMPERATURE','HUMIDITY','GROUND_WATERLEVEL']

######################################## Train/test split #########################################

df_train, df_test = train_test_split(df, test_size=0.20)
df_train = df_train[features]
df_test = df_test[features]

X_train, y_train = df_train.drop('PRODUCTION',axis=1), df_train['PRODUCTION']
X_test, y_test = df_test.drop('PRODUCTION',axis=1), df_test['PRODUCTION']

################################################ Train #############################################

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

############################### Permutation feature importance #####################################

imp = rfpimp.importances(rf, X_test, y_test)

############################################## Plot ################################################

fig, ax = plt.subplots(figsize=(6, 3))

ax.barh(imp.index, imp['Importance'], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
ax.set_xlabel('Importance score')
ax.set_title('Permutation feature importance')
ax.text(0.8, 0.15, 'aegis4048.github.io', fontsize=12, ha='center', va='center',
        transform=ax.transAxes, color='grey', alpha=0.5)
plt.gca().invert_yaxis()

fig.tight_layout()