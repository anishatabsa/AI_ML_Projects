import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EmployeeAttritionPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.label_encoders = {}
        self.tree_rules = []
        
    def load_real_dataset(self, file_path):
        """Load real IBM HR Analytics dataset"""
        try:
            df = pd.read_csv(Path(os.path.dirname(__file__) + file_path))
            print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            
            # Display basic info about the dataset
            print(f"\nAttrition distribution:")
            if 'Attrition' in df.columns:
                print(df['Attrition'].value_counts())
                print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\nMissing values found:")
                print(missing_values[missing_values > 0])
            
        
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found. Creating synthetic dataset instead.")
            return self.create_sample_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating synthetic dataset instead.")
           
        
    
    def preprocess_data(self, df):
        """Preprocess the data for decision tree"""
        df_processed = df.copy()
        
        # Automatically detect and encode all categorical variables
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category':
                print(f"Encoding categorical column: {col}")
                le = LabelEncoder()
                # Handle any missing values by converting to string first
                df_processed[col] = df_processed[col].astype(str)
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"  {col} classes: {list(le.classes_)}")
        
        return df_processed
    
    def train_decision_tree(self, df):
        """Train decision tree classifier"""
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Prepare features and target
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train decision tree
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Decision Tree Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))
        
        return X_test, y_test, y_pred
    
    def extract_rules(self):
        """Extract rules from the decision tree"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get tree rules in text format
        tree_rules = export_text(self.model, feature_names=self.feature_names)
        print("\nDecision Tree Rules:")
        print("=" * 50)
        print(tree_rules)
        
        # Extract simplified rules
        self.tree_rules = self.simplify_rules(tree_rules)
        return self.tree_rules
    
    def simplify_rules(self, tree_rules):
        """Convert tree rules to simplified format for Prolog"""
        rules = []
        lines = tree_rules.split('\n')
        
        current_conditions = []
        
        for line in lines:
            if '|---' in line:
                # Extract condition
                depth = line.count('|   ')
                condition = line.strip().replace('|--- ', '')
                
                # Adjust current conditions based on depth
                current_conditions = current_conditions[:depth]
                
                if 'class:' in condition:
                    # This is a leaf node with prediction
                    prediction = condition.split('class: ')[1].strip()
                    rule = {
                        'conditions': current_conditions.copy(),
                        'prediction': prediction
                    }
                    rules.append(rule)
                else:
                    # This is a condition
                    current_conditions.append(condition)
        
        return rules
    
    def generate_prolog_rules(self):
        """Generate Prolog rules from decision tree"""
        if not self.tree_rules:
            print("Rules not extracted yet!")
            return
        
        prolog_code = """% Employee Attrition Prediction Rules
% Generated from Decision Tree

% Facts for categorical mappings
department(0, 'HR').
department(1, 'R&D').
department(2, 'Sales').

job_role(0, 'Analyst').
job_role(1, 'Engineer').
job_role(2, 'Manager').
job_role(3, 'Sales Rep').
job_role(4, 'Technician').

marital_status(0, 'Divorced').
marital_status(1, 'Married').
marital_status(2, 'Single').

overtime(0, 'No').
overtime(1, 'Yes').

% Main prediction rules
"""
        
        # Add simplified rules
        for i, rule in enumerate(self.tree_rules[:10]):  # Limit to first 10 rules
            rule_name = f"attrition_rule_{i+1}"
            conditions_str = ""
            
            for condition in rule['conditions']:
                # Convert sklearn conditions to Prolog format
                if '<=' in condition:
                    feature, threshold = condition.split(' <= ')
                    conditions_str += f", {feature} =< {threshold}"
                elif '>' in condition:
                    feature, threshold = condition.split(' > ')
                    conditions_str += f", {feature} > {threshold}"
            
            prediction = "yes" if rule['prediction'] == '1' else "no"
            
            prolog_code += f"""
{rule_name}(Age, Department, Distance, Education, JobRole, Satisfaction, 
         Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    true{conditions_str}.

predict_attrition({prediction}, Age, Department, Distance, Education, JobRole, Satisfaction, 
                 Marital, Income, Companies, Overtime, WorkYears, CompanyYears) :-
    {rule_name}(Age, Department, Distance, Education, JobRole, Satisfaction, 
              Marital, Income, Companies, Overtime, WorkYears, CompanyYears).
"""
        
        # Add main prediction predicate
        prolog_code += """
% Main prediction predicate
employee_will_leave(Employee) :-
    employee(Employee, Age, Department, Distance, Education, JobRole, Satisfaction, 
             Marital, Income, Companies, Overtime, WorkYears, CompanyYears),
    predict_attrition(yes, Age, Department, Distance, Education, JobRole, Satisfaction, 
                     Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

employee_will_stay(Employee) :-
    employee(Employee, Age, Department, Distance, Education, JobRole, Satisfaction, 
             Marital, Income, Companies, Overtime, WorkYears, CompanyYears),
    predict_attrition(no, Age, Department, Distance, Education, JobRole, Satisfaction, 
                     Marital, Income, Companies, Overtime, WorkYears, CompanyYears).

% Sample employee facts (you can add more)
employee(john, 35, 1, 10, 3, 1, 4, 1, 75000, 2, 1, 15, 5).
employee(sarah, 28, 0, 25, 4, 0, 2, 2, 45000, 4, 1, 8, 2).

% Query examples:
% ?- employee_will_leave(john).
% ?- employee_will_stay(sarah).
% ?- predict_attrition(Result, 30, 2, 15, 3, 2, 3, 1, 60000, 3, 0, 10, 4).
"""
        
        return prolog_code
    
    def interactive_prediction(self):
        """Interactive employee attrition prediction"""
        print("\n" + "="*60)
        print("EMPLOYEE ATTRITION PREDICTION SYSTEM")
        print("="*60)
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        print(f"\nModel trained on features: {self.feature_names}")
        print(f"Available encodings: {list(self.label_encoders.keys())}")
        
        while True:
            try:
                print("\nEnter employee details:")
                input_dict = {}
                
                # Dynamically collect input based on trained features
                for feature in self.feature_names:
                    if feature in self.label_encoders:
                        # This is a categorical feature
                        encoder = self.label_encoders[feature]
                        print(f"{feature} options: {list(encoder.classes_)}")
                        value = input(f"{feature}: ").strip()
                        try:
                            encoded_value = encoder.transform([value])[0]
                            input_dict[feature] = encoded_value
                        except ValueError:
                            print(f"Invalid value for {feature}. Using first available option: {encoder.classes_[0]}")
                            input_dict[feature] = 0
                    else:
                        # This is a numerical feature
                        try:
                            value = float(input(f"{feature}: "))
                            input_dict[feature] = value
                        except ValueError:
                            print(f"Invalid numerical value for {feature}. Using 0.")
                            input_dict[feature] = 0
                
                # Prepare input array in the correct order
                input_data = np.array([[input_dict[feature] for feature in self.feature_names]])
                
                # Make prediction
                prediction = self.model.predict(input_data)[0]
                probability = self.model.predict_proba(input_data)[0]
                
                result = "likely to LEAVE" if prediction == 1 else "likely to STAY"
                confidence = probability[prediction] * 100
                
                print(f"\nPREDICTION: Employee is {result}")
                print(f"Confidence: {confidence:.1f}%")
                print(f"Probability of leaving: {probability[1]*100:.1f}%")
                print(f"Probability of staying: {probability[0]*100:.1f}%")
                
                if prediction == 1:
                    print("⚠️  HIGH ATTRITION RISK!")
                else:
                    print("✅ LOW ATTRITION RISK")
                
                # Show feature importance for this prediction
                feature_importance = self.model.feature_importances_
                important_features = sorted(
                    zip(self.feature_names, feature_importance),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                print("\nTop factors influencing predictions:")
                for feature, importance in important_features:
                    print(f"- {feature}: {importance:.3f}")
                
                continue_pred = input("\nPredict for another employee? (y/n): ").lower()
                if continue_pred != 'y':
                    break
                    
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again with valid values!")

def main():
    # Create predictor instance
    predictor = EmployeeAttritionPredictor()
    
    # Try to load real dataset first, fallback to synthetic
    dataset_path = '//WA_Fn-UseC_HR-Employee-Attrition.csv'
    
    if dataset_path:
        df = predictor.load_real_dataset(dataset_path)
    else:
        print("Creating sample dataset...")
        df = predictor.create_sample_dataset()
    
    print(f"Dataset loaded with {len(df)} employees")
    if 'Attrition' in df.columns:
        print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")
    
    print("\nDataset preview:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check data types
    print(f"\nData types:")
    print(df.dtypes)
    
    print("\nTraining Decision Tree...")
    X_test, y_test, y_pred = predictor.train_decision_tree(df)
    
    print("\nExtracting rules from Decision Tree...")
    rules = predictor.extract_rules()
    
    print("\nGenerating Prolog Knowledge Base...")
    prolog_code = predictor.generate_prolog_rules()
    
    # Save Prolog code to file
    try:
        with open('employee_attrition_rules.pl', 'w') as f:
            f.write(prolog_code)
        print("Prolog rules saved to 'employee_attrition_rules.pl'")
    except Exception as e:
        print(f"Error saving Prolog file: {e}")
    
    print("\nProlog Knowledge Base Preview:")
    print("=" * 50)
    print(prolog_code[:1000] + "..." if len(prolog_code) > 1000 else prolog_code)
    
    # Interactive prediction
    try_interactive = input("\nWould you like to try interactive prediction? (y/n): ").lower()
    if try_interactive == 'y':
        predictor.interactive_prediction()

if __name__ == "__main__":
    main()