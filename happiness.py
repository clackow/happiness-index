#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:46:06 2019

@author: Paranoia
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

dataset = pd.read_csv('happiness_train_complete.csv', encoding='latin-1');
dataset = dataset.drop(['property_other'] ,axis = 1)
dataset = dataset.drop(['invest_other'] ,axis = 1)
dataset = dataset.drop(['edu_other'] ,axis = 1)
dataset = dataset.drop(['join_party'],axis = 1)


test = pd.DataFrame(dataset, columns = ['survey_type','province','city','gender','birth','nationality','religion','edu','income','floor_area','health','health_problem','depression','hukou','hukou_loc','socialize'
 ,'relax','learn','social_friend','socia_outing','equity','class_10_before','class_10_after','class_14','work_exper','insur_1','insur_2','insur_3'
 ,'insur_4','family_income','family_m','family_status','house','car','invest_0','invest_1','invest_2','invest_3','invest_4','invest_5','invest_6','invest_7','invest_8','son','daughter','minor_child','marital'
 ,'s_edu','s_political','s_hukou','s_income','s_work_exper','f_birth','f_edu','f_work_14','m_birth','m_work_14','status_peer','status_3_before','view','inc_ability','inc_exp'
 ,'neighbor_familiarity','public_service_1','public_service_2','public_service_3','public_service_3','public_service_4','public_service_5','public_service_6','public_service_7','public_service_8','public_service_9'
 ,'invest_0','invest_1','invest_2','invest_3','invest_4','invest_5','invest_6','invest_7','invest_8'])
test.head()

X = test.iloc[:,:].values
y = dataset.iloc[:,1].values
Z = test.isnull().sum()


imputer = Imputer(missing_values = "NaN", strategy = "most_frequent", axis = 0)
X = imputer.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

