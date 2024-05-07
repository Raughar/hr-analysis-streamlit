#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

#Importing the dataset
@st.cache_data
def load_data():
    return pd.read_csv('files/hr-attrition.csv')

data = load_data()
data = data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])
#Creating the title of the page
st.title('Attrition Prediction')
#Creating the subheader of the page
st.subheader('Prediction')
#Creating the prediction of the attrition
st.write('The following form allows you to predict the attrition of an employee')
#Creating the form to input the data, the values needed are: MonthlyIncome, Age, TotalWorkingYears, OverTime, YearsWithCurrManager, DailyRate, MonthlyRate, YearsAtCompany, StockOptionLevel, DistanceFromHome, HourlyRate, JobRole, NumCompaniesWorked, PercentSalaryHike, JobLevel, YearsInCurrentRole, EnvironmentSatisfaction, MaritalStatus, JobSatisfaction, YearsSinceLastPromotion, WorkLifeBalance, RelationshipSatisfaction, Education, TrainingTimesLastYear, JobInvolvement, BusinessTravel, EducationField, Department, Gender, PerformanceRating 
Age = st.number_input('Age', min_value=0, max_value=100)
Gender = st.selectbox('Gender', ['Male', 'Female'])
BusinessTravel = st.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
DailyRate = st.number_input('Daily Rate', min_value=0)
Department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
DistanceFromHome = st.number_input('Distance From Home', min_value=0)
Education = st.number_input('Education', min_value=0, max_value=5)
EducationField = st.selectbox('Education Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
EnvironmentSatisfaction = st.number_input('Environment Satisfaction', min_value=0, max_value=5)
MonthlyIncome = st.number_input('Monthly Income', min_value=0)
JobInvolvement = st.number_input('Job Involvement', min_value=0, max_value=5)
JobLevel = st.number_input('Job Level', min_value=0)
JobRole = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
JobSatisfaction = st.number_input('Job Satisfaction', min_value=0, max_value=5)
MaritalStatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
MonthlyRate = st.number_input('Monthly Rate', min_value=0)
HourlyRate = st.number_input('Hourly Rate', min_value=0)
NumCompaniesWorked = st.number_input('Number of Companies Worked', min_value=0)
OverTime = st.selectbox('Over Time', ['Yes', 'No'])
PercentSalaryHike = st.number_input('Percent Salary Hike', min_value=0)
PerformanceRating = st.number_input('Performance Rating', min_value=0, max_value=5)
RelationshipSatisfaction = st.number_input('Relationship Satisfaction', min_value=0, max_value=5)
StockOptionLevel = st.number_input('Stock Option Level', min_value=0)
TotalWorkingYears = st.number_input('Total Working Years', min_value=0)
TrainingTimesLastYear = st.number_input('Training Times Last Year', min_value=0)
WorkLifeBalance = st.number_input('Work Life Balance', min_value=0, max_value=5)
YearsAtCompany = st.number_input('Years At Company', min_value=0)
YearsInCurrentRole = st.number_input('Years In Current Role', min_value=0)
YearsSinceLastPromotion = st.number_input('Years Since Last Promotion', min_value=0)
YearsWithCurrManager = st.number_input('Years With Current Manager', min_value=0)
Attrition = st.selectbox('Attrition', ['Yes', 'No'])   



#final_data = final_data[['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Education', 'EnvironmentSatisfaction', 'JobLevel', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance', 'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
#Creating the prediction button
if st.button('Predict'):
    #Importing the libraries
    from imblearn.ensemble import EasyEnsembleClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from feature_engine.outliers import Winsorizer
    from sklearn.preprocessing import PowerTransformer

    # Create the dataframe with the input data
    input_data = pd.DataFrame({'Age': [Age], 'Gender' : [Gender], 'BusinessTravel': [BusinessTravel], 'HourlyRate' : [HourlyRate], 'DailyRate': [DailyRate], 'Department': [Department], 'DistanceFromHome': [DistanceFromHome], 'Education': [Education], 'EducationField': [EducationField], 'EnvironmentSatisfaction': [EnvironmentSatisfaction], 'MonthlyIncome': [MonthlyIncome], 'JobInvolvement': [JobInvolvement], 'JobLevel': [JobLevel], 'JobRole': [JobRole], 'JobSatisfaction': [JobSatisfaction], 'MaritalStatus': [MaritalStatus], 'MonthlyRate': [MonthlyRate], 'NumCompaniesWorked': [NumCompaniesWorked], 'OverTime': [OverTime], 'PercentSalaryHike': [PercentSalaryHike], 'PerformanceRating': [PerformanceRating], 'RelationshipSatisfaction': [RelationshipSatisfaction], 'StockOptionLevel': [StockOptionLevel], 'TotalWorkingYears': [TotalWorkingYears], 'TrainingTimesLastYear': [TrainingTimesLastYear], 'WorkLifeBalance': [WorkLifeBalance], 'YearsAtCompany': [YearsAtCompany], 'YearsInCurrentRole': [YearsInCurrentRole], 'YearsSinceLastPromotion': [YearsSinceLastPromotion], 'YearsWithCurrManager': [YearsWithCurrManager], 'Attrition': [Attrition]})

    # #Separating categorical and numerical columns
    cat_cols = data.select_dtypes(include=object)
    cat_cols = pd.concat([cat_cols, data[['Education', 'EnvironmentSatisfaction', 'JobLevel', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']]], axis=1)
    # #Adding the columns that are not in the categorical columns to the numerical columns
    num_cols = data.drop(columns=cat_cols.columns)

    #Encoding the categorical columns
    le = LabelEncoder()
    for col in cat_cols.columns:
        cat_cols[col] = le.fit_transform(cat_cols[col])

    #Transforming the data
    power_data = num_cols.copy()
    #Applying the power transformer to the numerical columns
    power = PowerTransformer()
    power_data[power_data.columns] = power.fit_transform(power_data[power_data.columns])

    #Concatenating the categorical and numerical columns
    final_data = pd.concat([cat_cols, power_data], axis=1)

    #Creating the X and y variables
    X = final_data.drop(columns='Attrition')
    y = final_data['Attrition']

    #Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #Creating the model
    eec = EasyEnsembleClassifier(n_estimators=250)
    model = eec.fit(X_train, y_train)

    #Separating the input data into categorical and numerical columns
    cat_cols_input = input_data.select_dtypes(include=object)
    cat_cols_input = pd.concat([cat_cols_input, input_data[['Education', 'EnvironmentSatisfaction', 'JobLevel', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']]], axis=1)
    #Adding the columns that are not in the categorical columns to the numerical columns
    num_cols_input = input_data.drop(columns=cat_cols_input.columns)

    #Transforming the data
    power_data_input = num_cols_input.copy()
    #Applying the power transformer to the numerical columns
    power_data_input[power_data_input.columns] = power.transform(power_data_input[power_data_input.columns])

    #Encoding the categorical columns
    for col in cat_cols_input.columns:
        cat_cols_input[col] = le.transform(cat_cols_input[col])

    #Concatenating the categorical and numerical columns
    final_input = pd.concat([cat_cols_input, power_data_input], axis=1)

    #Making the prediction
    prediction = model.predict(final_input)

    # Showing the prediction
    if prediction[0] == 0:
        st.write('The employee will not leave the company')
    else:
        st.write('The employee will leave the company')
#Creating the footer of the page
st.write('Made by: Samuel Carmona Skories')