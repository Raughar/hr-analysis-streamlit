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

#Creating the basic shape of the streamlit app
st.title('Attrition Dashboard')
st.write('This is a dashboard to understand the attrition in the company')

# #Creating the sidebar to navigate between the pages
# page = st.sidebar.selectbox('Select a page', ['Dashboard', 'Attrition Prediction'])

# #Creating the dashboard page

# if page == 'Dashboard':
#     #Creating the title of the page
#     st.header('Dashboard')
#     #Creating the subheader of the page
#     st.subheader('Basic Information')
#     #Creating the basic information of the dataset
#     st.write('The dataset contains information about the employees in the company')
#     st.write('The dataset contains', data.shape[0], 'rows and', data.shape[1], 'columns')

#     #Creating the subheader of the page
#     st.subheader('Data')
#     #Creating the data table
#     st.write(data)

#     #Creating the subheader of the page
#     st.subheader('Plots')
#     #Creating the plots for the dashboard
#     st.write('The following plots show the distribution of the data')
#     #Creating the pairplot
#     sns.pairplot(data)
#     st.pyplot()

#     #Creating the subheader of the page
#     st.subheader('Statistics')
#     #Creating the statistics of the data
#     st.write('The following table shows the statistics of the data')
#     st.write(data.describe())

# #Creating the attrition prediction page

# if page == 'Attrition Prediction':
#Creating the title of the page
st.header('Attrition Prediction')
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
    import pickle

    # Load the LabelEncoder from the file
    le = pickle.load(open('files/label-encoder','rb'))

    # Load the model from the file
    model = pickle.load(open('files/model','rb'))

    # Load the transformers from the file
    transformer = pickle.load(open('files/transformer','rb'))

    # Create the dataframe with the input data
    input_data = pd.DataFrame({'Age': [Age], 'Gender' : [Gender], 'BusinessTravel': [BusinessTravel], 'HourlyRate' : [HourlyRate], 'DailyRate': [DailyRate], 'Department': [Department], 'DistanceFromHome': [DistanceFromHome], 'Education': [Education], 'EducationField': [EducationField], 'EnvironmentSatisfaction': [EnvironmentSatisfaction], 'MonthlyIncome': [MonthlyIncome], 'JobInvolvement': [JobInvolvement], 'JobLevel': [JobLevel], 'JobRole': [JobRole], 'JobSatisfaction': [JobSatisfaction], 'MaritalStatus': [MaritalStatus], 'MonthlyRate': [MonthlyRate], 'NumCompaniesWorked': [NumCompaniesWorked], 'OverTime': [OverTime], 'PercentSalaryHike': [PercentSalaryHike], 'PerformanceRating': [PerformanceRating], 'RelationshipSatisfaction': [RelationshipSatisfaction], 'StockOptionLevel': [StockOptionLevel], 'TotalWorkingYears': [TotalWorkingYears], 'TrainingTimesLastYear': [TrainingTimesLastYear], 'WorkLifeBalance': [WorkLifeBalance], 'YearsAtCompany': [YearsAtCompany], 'YearsInCurrentRole': [YearsInCurrentRole], 'YearsSinceLastPromotion': [YearsSinceLastPromotion], 'YearsWithCurrManager': [YearsWithCurrManager], 'Attrition': [Attrition]})

    # Encoding the categorical data
    cat_cols = input_data.select_dtypes(include='object').columns
    cat_cols = pd.concat([pd.Series(cat_cols), input_data[['Education', 'EnvironmentSatisfaction', 'JobLevel', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']]], axis=1)
    for col in cat_cols.columns:
        cat_cols[col] = le.fit_transform(cat_cols[col])

    # Transforming the data in the numerical columns
    num_cols = input_data.drop(columns=cat_cols.columns)

    # Transform the numerical columns using the loaded transformer
    num_cols = transformer.transform(num_cols[num_cols.columns])

    # Applying Winsorizer to the data
    winsor_data = num_cols.copy()
    winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=list(winsor_data.columns))
    winsor_data[winsor_data.columns] = winsorizer.fit_transform(winsor_data)

    # Concatenating the data
    final_data = pd.concat([cat_cols, winsor_data], axis=1)

    # Reordering the data as to how the model was trained
    final_data = final_data[['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Education', 'EnvironmentSatisfaction', 'JobLevel', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance', 'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

    # Splitting the data
    X = final_data.drop(columns='Attrition')
    y = final_data['Attrition']

    # Predicting the attrition using the loaded model
    prediction = model.predict(X)

    # Showing the prediction
    if prediction[0] == 0:
        st.write('The employee will not leave the company')
    else:
        st.write('The employee will leave the company')
#Creating the footer of the page
st.write('Made by: Samuel Carmona Skories')