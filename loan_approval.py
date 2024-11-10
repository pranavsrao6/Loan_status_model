import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency


# Converting data into pandas df
loan_data = pd.read_csv('/Users/pranavsrao/Kaggle Assignments/Loan Approval/loan_data.csv')


# Converting categorical data into numerical
loan_data.loc[loan_data['person_gender'] == 'female', 'person_gender'] = 0
loan_data.loc[loan_data['person_gender'] == 'male', 'person_gender'] = 1




loan_data['person_education'] = loan_data['person_education'].map(
    {'High School': 0, 
     'Associate': 1, 
     'Bachelor': 2, 
     'Master': 3, 
     'Doctorate': 4})


loan_data['loan_intent'] = loan_data['loan_intent'].map(
    {
        
        'EDUCATION': 0,
         'MEDICAL': 1,
        'DEBTCONSOLIDATION': 4,
        'PERSONAL': 3,
        'VENTURE': 2,
        'HOMEIMPROVEMENT': 5,
    }
)

loan_data['person_home_ownership'] = loan_data['person_home_ownership'].map(
    {
        'RENT': 0,
        'MORTGAGE': 1,
        'OWN': 2,
        'OTHER': 3,
    }
)

loan_data['previous_loan_defaults_on_file'] = loan_data['previous_loan_defaults_on_file'].map(
    {
        'Yes': 1,
        'No': 0
    }
)



# Get a sample from the population (population data is heavily skewed to 0)

majority = loan_data.loc[loan_data['loan_status'] == 0]
minority = loan_data.loc[loan_data['loan_status'] == 1]


new_majority = majority.sample(n=len(minority))

new_data = pd.concat([new_majority, minority])

new_data = new_data.sample(frac=1).reset_index(drop=True)


# Find the independent variables with the highest correlation with loan_status

# Numerical Values

# correlation, p_value = pointbiserialr(loan_data['cb_person_cred_hist_length'], loan_data['loan_status'])
# print(f'Correlation: {correlation}')

# age: -0.02, income: -0.14, emp_exp: -0.02, loan_amnt: .11, loan_int_rate: 0.33, loan_percent_income: 0.38, credit_score: -0.01, 


# Categorical Values

# contingency_table = pd.crosstab(loan_data['person_emp_exp'], loan_data['loan_status'])

# chi2, p, dof, expected = chi2_contingency(contingency_table)
# print(f'Chi-square statistic: {chi2}, p-value: {p}')

# gender: bad, education: bad, person_home_ownership: good, loan_intent: bad, cb_person_cred_hist_length: bad, previous_loan_defaults_on_file: good



# Standardizing values
scaler = StandardScaler()
new_data[['loan_amnt', 'loan_int_rate', 'loan_int_rate', 'loan_percent_income', 'person_age', 'credit_score', 'person_emp_exp']] = scaler.fit_transform(new_data[['loan_amnt', 'loan_int_rate', 'loan_int_rate', 'loan_percent_income','person_age','credit_score', 'person_emp_exp']])



new_x = new_data[['loan_amnt', 'loan_int_rate', 'loan_int_rate', 'loan_percent_income', 'person_home_ownership', 'previous_loan_defaults_on_file','credit_score']]
new_y = new_data['loan_status']


# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(new_x,new_y, test_size= 0.25)


# Creating a logistic regression model using the train data
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)


betas = log_reg.coef_[0]
intercept = log_reg.intercept_[0]



# Use the model to predict values based on x_test
predictions = log_reg.predict(x_test)


# Evaluating the model using confusion matrix and classification report
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

