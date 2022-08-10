# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn import model_selection as ms
from sklearn.ensemble import RandomForestRegressor
import random
#for maintaining randomness
random_state = 1


# read file from csv to pandas DataFrame
data = pd.read_csv(r'Cleaned_data.csv')

#Select relevant features from previous analysis
final_data = data[['country','year','co2','coal_co2','cement_co2','gas_co2','oil_co2','methane','population','gdp']]


#Remove Outliers (countries) with significantly  high range features
final_data = final_data[final_data['country'].isin(['Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Armenia',
       'Australia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium',
       'Benin', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana',
       'Bulgaria', 'Cameroon', 'Canada', 'Chile', 'Colombia', 'Croatia',
       'Cuba', 'Cyprus', 'Czechia', 'Denmark', 'Dominican Republic',
       'Egypt', 'Estonia', 'Finland', 'France', 'Georgia', 'Ghana',
       'Greece', 'Guatemala', 'Hungary', 'Iceland', 'Iraq', 'Ireland',
       'Israel', 'Italy', 'Jamaica', 'Jordan', 'Kazakhstan', 'Kyrgyzstan',
       'Latvia', 'Lebanon', 'Libya', 'Lithuania', 'Luxembourg',
       'Malaysia', 'Mexico', 'Moldova', 'Morocco', 'Mozambique',
       'Netherlands', 'New Zealand', 'North Macedonia', 'Norway',
       'Panama', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania',
       'Rwanda', 'Senegal', 'Serbia', 'Slovakia', 'Slovenia',
       'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Syria',
       'Tajikistan', 'Tanzania', 'Thailand', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Ukraine', 'United Arab Emirates',
       'United Kingdom', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Yemen'])]


#dimensionality reduction
final_data['ccgo'] = final_data['cement_co2'] + final_data['gas_co2'] + final_data['oil_co2'] + final_data['coal_co2']
final_data['gdp_per_capita'] = final_data['gdp'] / final_data['population']
final_data.head()


data = final_data.drop(['cement_co2','gas_co2','oil_co2','coal_co2','gdp','population'],axis=1)



#splitting dataset
ft_cols = ['year','methane','ccgo','gdp_per_capita']
lb_col = ['co2']

features = np.array(data[ft_cols])
label = np.array(data[lb_col]).ravel()

#Data splitting using sklearn train_test_split function
ft_train,ft_test,lb_train,lb_test = ms.train_test_split(features,label,test_size=0.3
                                                     ,shuffle = True, random_state= random_state)


RFR = RandomForestRegressor(max_depth = 9, max_features = 3, n_estimators = 40, random_state = random_state)
RFR.fit(ft_train, lb_train)

# Saving model to disk
pickle.dump(RFR, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
