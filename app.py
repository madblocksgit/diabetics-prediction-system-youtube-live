import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

data=pd.read_csv('diabetes.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values

classifier=KNeighborsClassifier()
classifier.fit(X,Y)

Y_pred=classifier.predict(X)

st.header('Diabetes Prediction System')

pregnancies=st.text_input('Enter Pregnancies')
glucose=st.text_input('Enter Glucose')
bloodPressure=st.text_input('Enter BloodPressure')
skinThickness=st.text_input('Enter Skin Thickness')
insulin=st.text_input('Enter Insulin')
bmi=st.text_input('Enter BMI')
dpf=st.text_input('Enter Diabetics Pedigree Function')
age=st.text_input('Enter Age')

if st.button('Predict'):
    inputData=[[pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age]]
    if(classifier.predict(inputData)[0]==0):
        st.success('No Diabetes')
    else:
        st.warning('Diabetes Found')
    
