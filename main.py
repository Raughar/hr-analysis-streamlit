#Importing the libraries
import pandas as pd
import streamlit as st
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

#Importing the dataset
@st.cache_data
def load_data():
    return pd.read_csv('files/hr-attrition.csv')
data = load_data()
data = data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

#Creating the title of the page
st.title('Attrition Prediction App')

#Creating the subheader of the page
st.subheader('Prediction')

#Creating the prediction of the attrition
st.write('The following form allows you to predict the attrition of an employee')

#Creating the columns of the pageto input the data
col1, col2 = st.columns(2)

#Creating the form to input the data, the columns are created to make the form look better
with col1:
    Age = st.number_input('Age', min_value=0, max_value=100)
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    BusinessTravel = st.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    DistanceFromHome = st.number_input('Distance From Home', min_value=0)
    HourlyRate = st.number_input('Hourly Rate', min_value=0)
    DailyRate = st.number_input('Daily Rate', min_value=0)
    MonthlyRate = st.number_input('Monthly Rate', min_value=0)
    MonthlyIncome = st.number_input('Monthly Income', min_value=0)
    PercentSalaryHike = st.number_input('Percent Salary Hike', min_value=0)
    StockOptionLevel = st.number_input('Stock Option Level', min_value=0)
    Education = st.number_input('Education', min_value=0, max_value=5)
    EducationField = st.selectbox('Education Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
    EnvironmentSatisfaction = st.number_input('Environment Satisfaction', min_value=0, max_value=5)
    MaritalStatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    OverTime = st.selectbox('Over Time', ['Yes', 'No'])

with col2:
    Department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    JobRole = st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    JobLevel = st.number_input('Job Level', min_value=0)
    JobInvolvement = st.number_input('Job Involvement', min_value=0, max_value=5)
    JobSatisfaction = st.number_input('Job Satisfaction', min_value=0, max_value=5)
    NumCompaniesWorked = st.number_input('Number of Companies Worked', min_value=0)
    PerformanceRating = st.number_input('Performance Rating', min_value=0, max_value=5)
    RelationshipSatisfaction = st.number_input('Relationship Satisfaction', min_value=0, max_value=5)
    TotalWorkingYears = st.number_input('Total Working Years', min_value=0)
    TrainingTimesLastYear = st.number_input('Training Times Last Year', min_value=0)
    WorkLifeBalance = st.number_input('Work Life Balance', min_value=0, max_value=5)
    YearsAtCompany = st.number_input('Years At Company', min_value=0)
    YearsInCurrentRole = st.number_input('Years In Current Role', min_value=0)
    YearsSinceLastPromotion = st.number_input('Years Since Last Promotion', min_value=0)
    YearsWithCurrManager = st.number_input('Years With Current Manager', min_value=0)

Attrition = st.selectbox('Attrition', ['Yes', 'No'])   

#Creating the prediction button
if st.button('Predict'):
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
    fit_columns = power_data.columns

    # Reorder the columns of `power_data_input` to match `fit_data`
    power_data_input = power_data_input[fit_columns]

    # Now you can transform `power_data_input`
    power_data_input[power_data_input.columns] = power.transform(power_data_input)

    #Encoding the categorical columns
    for col in cat_cols_input.columns:
        cat_cols_input[col] = le.fit_transform(cat_cols_input[col])

    #Concatenating the categorical and numerical columns
    final_input = pd.concat([cat_cols_input, power_data_input], axis=1)
    # Remove the 'Attrition' column from final_input
    final_input = final_input.drop(columns=['Attrition'])

    # Get the column order of X_train
    column_order = X_train.columns

    # Reorder the columns of final_input to match X_train
    final_input = final_input[column_order]

    #Making the prediction
    prediction = model.predict(final_input)

    # Showing the prediction
    if prediction[0] == 0:
        st.subheader('Prediction Result:')
        st.write('The employee will not leave the company')
    else:
        st.subheader('Prediction Result:')
        st.write('The employee will most likely leave the company')
        st.write('The probability of the employee leaving the company is:', model.predict_proba(final_input)[0][1] * 100, '%')

#Creating the footer of the page
#Footer
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        " with ❤️ by ",
        link("https://www.linkedin.com/in/samuelcskories/", "Samuel Carmona Sköries"),
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer()