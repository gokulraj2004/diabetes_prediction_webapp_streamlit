# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:35:28 2024

@author: GOKUL RAJ
"""

import numpy as np
import pickle as p
import streamlit as st

#loading the saved trained model
loaded_model=p.load(open('C:/Users/GOKUL RAJ/OneDrive/Documents/RAJ/projects/streamlit projects/basic_diabetes_mlmodeldeploy/trained_model.sav','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    
    #giving a title for the diabetes webapp
    st.title('Diabetes prediction web app')
    #geting the input from the user Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome

    Pregnancies=st.text_input("No of pregnancies:- ")
    Glucose=st.text_input("Glucose level:- ")
    BloodPressure=st.text_input("BloodPressure value:- ")
    SkinThickness=st.text_input("SkinThickness value:- ")
    Insulin=st.text_input("Insulin value:- ")
    BMI=st.text_input("BMI value:- ")
    DiabetesPedigreeFunction=st.text_input("DiabetesPedigreeFunction value:- ")
    Age=st.text_input("Age:- ")

    #code for prediction
    diagnosis=''
    
    #creating a button for prediction
    if st.button("Diabetes test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)


if __name__=='__main__':
    main()