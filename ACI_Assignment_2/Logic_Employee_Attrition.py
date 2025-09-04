import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import preprocessing
import joblib

# 1. Load dataset (update the path as needed)
df = pd.read_csv('//ACI_Assignment_2/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# 2. Select relevant columns
cols = [
    'Age', 'Department', 'DistanceFromHome', 'Education', 'JobRole',
    'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
    'OverTime', 'TotalWorkingYears', 'YearsAtCompany', 'Attrition'
]
df = df[cols]

# 3. Encode categorical variables with separate encoders
encoders = {}
for col in ['Department', 'JobRole', 'MaritalStatus', 'OverTime', 'Attrition']:
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Print possible values for Department
print('Department value mapping:')
for idx, val in enumerate(encoders['Department'].classes_):
    print(f'  {val}: {idx}')

# Print possible values for Education
print('Education value mapping:')
edu_map = {
    1: 'Below College',
    2: 'College',
    3: 'Bachelor',
    4: 'Master',
    5: 'Doctor'
}
for k, v in edu_map.items():
    print(f'  {v}: {k}')

# Print possible values for JobRole
print('JobRole value mapping:')
for idx, val in enumerate(encoders['JobRole'].classes_):
    print(f'  {val}: {idx}')

# Print possible values for MaritalStatus
print('MaritalStatus value mapping:')
for idx, val in enumerate(encoders['MaritalStatus'].classes_):
    print(f'  {val}: {idx}')

# Print possible values for OverTime
print('OverTime value mapping:')
for idx, val in enumerate(encoders['OverTime'].classes_):
    print(f'  {val}: {idx}')

# Print possible values for JobSatisfaction
print('JobSatisfaction value mapping:')
print('  1: Low\n  2: Medium\n  3: High\n  4: Very High')

# Print possible values for NumCompaniesWorked, TotalWorkingYears, YearsAtCompany, Age, DistanceFromHome, MonthlyIncome
print('Note: Enter integer values for Age, DistanceFromHome, MonthlyIncome, NumCompaniesWorked, TotalWorkingYears, YearsAtCompany, Education, JobSatisfaction.')

# 4. Split features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 5. Train decision tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# 6. Extract rules
rules = export_text(clf, feature_names=list(X.columns))
print('Extracted Decision Tree Rules:')
print(rules)

# 7. Save the model for later use
joblib.dump(clf, 'attrition_tree.joblib')

# 8. Accept user input and predict
print('\nEnter employee details:')
user_data = {}
for col in X.columns:
    if col in encoders:
        print(f"Possible values for {col}: {list(encoders[col].classes_)}")
    if col == 'Education':
        print(f"Possible values for Education: {edu_map}")
    if col == 'JobSatisfaction':
        print('Possible values for JobSatisfaction: 1 (Low), 2 (Medium), 3 (High), 4 (Very High)')
    val = input(f'{col}: ')
    # Simple type conversion
    if col in ['Age', 'DistanceFromHome', 'Education', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany']:
        val = int(val)
    elif col in encoders:
        val = encoders[col].transform([val])[0]  # assumes user enters valid value
    user_data[col] = val

user_df = pd.DataFrame([user_data])
pred = clf.predict(user_df)[0]
attrition_label = encoders['Attrition'].inverse_transform([pred])[0]
print(f'Predicted Attrition: {attrition_label}')

# 9. Write rules to Prolog file (simple example)
with open('attrition_rules.pl', 'w') as f:
    f.write('% Prolog rules generated from decision tree\n')
    for line in rules.split('\n'):
        f.write(f'% {line}\n')
    f.write('% Implement actual rules based on the above logic\n')

print('Prolog rules written to attrition_rules.pl')
