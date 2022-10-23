#setup
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler

st.title('Boston House Price Prediction:house:')
st.subheader("""
Explore the model with different data
Predict on you own data....
""")

#choose dataset from slidebar
dataset_name = st.sidebar.selectbox("Select Dataset",("Boston House","Coming Soon"))
st.subheader('Share the dataset with us :')

# :Attribute Information (in order):
    #     - CRIM     per capita crime rate by town
    #     - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    #     - INDUS    proportion of non-retail business acres per town
    #     - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    #     - NOX      nitric oxides concentration (parts per 10 million)
    #     - RM       average number of rooms per dwelling
    #     - AGE      proportion of owner-occupied units built prior to 1940
    #     - DIS      weighted distances to five Boston employment centres
    #     - RAD      index of accessibility to radial highways
    #     - TAX      full-value property-tax rate per $10,000
    #     - PTRATIO  pupil-teacher ratio by town
    #     - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
    #     - LSTAT    % lower status of the population
    
#all features value take from users
CRIME_ZONE = st.slider('Capita crime rate by town',0,1000)
ZN_ZONE = st.slider('Proportion of residential land zoned for lots over 25,000 sq.ft.',0,1000)
INDUS_VAL = st.slider('Proportion of non-retail business acres per town',0,50)
CHAS_VAL = st.selectbox('Charles River 1 if tract bounds river; 0 otherwise',(0,1))
NOX_VAL = st.slider('Nitric oxides concentration (parts per 10 million)',0.0,2.5)
RM_VAL = st.slider('Average number of rooms per dwelling',1,20)
AGE_VAL = st.slider('Proportion of owner-occupied units built prior to 1940',0,500)
DIS_VAL = st.slider('Weighted distances to five Boston employment centres',0,10)
RAD_VAL = st.slider('Index of accessibility to radial highways',1,10)
TAX_VAL = st.slider('fFll-value property-tax rate per $10,000',1,500)
PTRATIO_VAL = st.slider('Pupil-teacher ratio by town',0,50)
B_VAL = st.slider('1000(Bk - 0.63)^2 where Bk is the proportion of black people by town',0,1000)
LSTAT_VAL = st.slider('% lower status of the population',0,50)

# loading model and 
regmodel=pickle.load(open("regmodel.pkl","rb"))
#scaling = load_model('scaling.pkl')

# Funcion for predicting the price of house
def predict():
    row = np.array([CRIME_ZONE,ZN_ZONE,INDUS_VAL,
                    CHAS_VAL,NOX_VAL,RM_VAL,AGE_VAL,
                    DIS_VAL,RAD_VAL,TAX_VAL,PTRATIO_VAL,
                    B_VAL,LSTAT_VAL
                    ])
    st.write('Predicting on your data.....')
    x = pd.DataFrame([row])
    #standard = scaling.stand(x)
    #standard = StandardScaler.transform(x)
    prediction = regmodel.predict(x)
    st.write("Price of the building shoulb be -",prediction)

#Button for prediction     
st.button('Predict', on_click=predict)


    