#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

#Importing the dataset
data = pd.read_csv('files\WA_Fn-UseC_-HR-Employee-Attrition.csv')

#Creating the basic shape of the streamlit app
st.title('Attrition Dashboard')
st.write('This is a dashboard to understand the attrition in the company')

#Creating the sidebar to navigate between the pages
page = st.sidebar.selectbox('Select a page', ['Dashboard', 'Attrition Prediction'])

#Creating the dashboard page

if page == 'Dashboard':
    st.write('This is the dashboard page')
    st.write('The number of employees who left the company are:')
    st.write(data['Attrition'].value_counts())
    st.write('The average age of the employees is:')
    st.write(data['Age'].mean())
    st.write('The average monthly income of the employees is:')
    st.write(data['MonthlyIncome'].mean())
    st.write('The average years at the company is:')
    st.write(data['YearsAtCompany'].mean())
    st.write('The average years since the last promotion is:')
    st.write(data['YearsSinceLastPromotion'].mean())
    st.write('The average years with the current manager is:')
    st.write(data['YearsWithCurrManager'].mean())
    st.write('The average years since the last salary hike is:')
    st.write(data['YearsSinceLastPromotion'].mean())
    st.write('The average years with the current role is:')
    st.write(data['YearsInCurrentRole'].mean())
    st.write('The average number of companies worked at is:')
    st.write(data['NumCompaniesWorked'].mean())
    st.write('The average number of training times last year is:')
    st.write(data['TrainingTimesLastYear'].mean())
    st.write('The average number of years in the current role is:')
    st.write(data['YearsInCurrentRole'].mean())
    st.write('The average number of years with the current manager is:')
    st.write(data['YearsWithCurrManager'].mean())
    st.write('The average number of years since the last promotion is:')
    st.write(data['YearsSinceLastPromotion'].mean())
    st.write('The average number of years since the last salary hike is:')
    st.write(data['YearsSinceLastPromotion'].mean())
    st.write('The average number of years at the company is:')
    st.write(data['YearsAtCompany'].mean())
    st.write('The average number of years in the current role is:')
    st.write(data['YearsInCurrentRole'].mean())
    st.write('The average number of years with the current manager is:')
    st.write(data['YearsWithCurrManager'].mean())
    st.write('The average number of years since the last promotion is:')
    st.write(data['YearsSinceLastPromotion'].mean())
    st.write('The average number of years since the last salary hike is:')
    st.write(data['YearsSinceLastPromotion'].mean())
    st.write('The average number of years at the company is:')
    st.write(data['YearsAtCompany'].mean())
    st.write('The average number of years in the current role is:')
    st.write(data['YearsInCurrentRole'].mean())
    st.write('The average number of years with the current manager is:')
    st.write(data['YearsWithCurrManager'].mean())
    st.write('The average number of years since the last promotion is:')
    st.write(data['YearsSinceLastPromotion'].mean())
    st.write('The average number of years since the last salary hike is:')
    st.write(data['YearsSinceLastPromotion'].mean())