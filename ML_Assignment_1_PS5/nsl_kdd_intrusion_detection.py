# Network Intrusion Detection System using NSL-KDD Dataset
# Machine Learning Assignment - Problem Statement 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")
print("="*50)

# =============================================================================
# 1. IMPORT LIBRARIES/DATASET
# =============================================================================

print("1. DATASET LOADING")
print("-" * 30)

# Define column names for NSL-KDD dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

# Load the dataset
# Note: You need to download KDDTrain+.txt from the GitHub repository
try:
    # Try to load the dataset
    df = pd.read_csv('KDDTrain+.txt', names=column_names)
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Dataset file not found. Please download KDDTrain+.txt from:")
    print("https://github.com/jmnwong/NSL-KDD-Dataset")
    # Create a sample dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        col: np.random.randn(n_samples) if col not in ['protocol_type', 'service', 'flag', 'class', 'difficulty'] 
        else np.random.choice(['tcp', 'udp', 'icmp'], n_samples) if col == 'protocol_type'
        else np.random.choice(['http', 'ftp', 'smtp'], n_samples) if col == 'service'
        else np.random.choice(['SF', 'S0', 'REJ'], n_samples) if col == 'flag'
        else np.random.choice(['normal', 'smurf', 'neptune', 'back'], n_samples) if col == 'class'
        else np.random.randint(1, 22, n_samples)
        for col in column_names
    })
    print("Using sample dataset for demonstration")

# Remove difficulty column as it's not needed for classification
if 'difficulty' in df.columns:
    df = df.drop('difficulty', axis=1)

print(f"Final dataset shape: {df.shape}")
print("="*50)

# =============================================================================
# 2. DATA VISUALIZATION AND EXPLORATION
# =============================================================================

print("2. DATA VISUALIZATION AND EXPLORATION")
print("-" * 40)

# 2a. Print 5 rows for sanity check
print("2a. Sample Data (First 5 rows):")
print(df.head())
print(f"\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"Features: {df.shape[1]-1}")
print(f"Samples: {df.shape[0]}")

# Check data types
print(f"\nData Types:")
print(df.dtypes.value_counts())

# 2b. Data Visualizations
print("\n2b. Data Visualizations:")

# Create binary classification labels
df['binary_class'] = df['class'].apply(lambda x: 'Normal' if x == 'normal' else 'Attack')

# Class distribution
plt.figure(figsize=(15, 12))

# Class distribution
plt.subplot(2, 3, 1)
class_counts = df['binary_class'].value_counts()
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Binary Class Distribution')

# Original class distribution
plt.subplot(2, 3, 2)
original_classes = df['class'].value_counts().head(10)
plt.bar(range(len(original_classes)), original_classes.values)
plt.title('Top 10 Attack Types')
plt.xticks(range(len(original_classes)), original_classes.index, rotation=45)

# Protocol type distribution
plt.subplot(2, 3, 3)
protocol_counts = df['protocol_type'].value_counts()
plt.bar(protocol_counts.index, protocol_counts.values)
plt.title('Protocol Type Distribution')
plt.xticks(rotation=45)

# Service distribution (top 10)
plt.subplot(2, 3, 4)
service_counts = df['service'].value_counts().head(10)
plt.bar(range(len(service_counts)), service_counts.values)
plt.title('Top 10 Services')
plt.xticks(range(len(service_counts)), service_counts.index, rotation=45)

# Numeric features distribution
plt.subplot(2, 3, 5)
numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
for col in numeric_cols:
    plt.hist(df[col], alpha=0.5, label=col, bins=30)
plt.title('Distribution of Numeric Features')
plt.legend()
plt.yscale('log')

# Attack vs Normal comparison for a key feature
plt.subplot(2, 3, 6)
if 'duration' in df.columns:
    sns.boxplot(data=df, x='binary_class', y='duration')
    plt.title('Duration by Class')

plt.tight_layout()
plt.show()

print("✓ Data visualizations completed")

# 2c. Correlation Analysis
print("\n2c. Correlation Analysis:")

# Select numeric columns for correlation
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df[numeric_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# Find highly correlated features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j], 
                correlation_matrix.iloc[i, j]
            ))

print(f"Highly correlated feature pairs (|correlation| > 0.8):")
for pair in high_corr_pairs[:10]:  # Show top 10
    print(f"  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")

print(f"\n📊 OBSERVATIONS - Data Exploration:")
print(f"• Dataset contains {df.shape[0]} samples with {df.shape[1]-1} features")
print(f"• Binary classification: {class_counts['Normal']} Normal, {class_counts['Attack']} Attack samples")
print(f"• Found {len(high_corr_pairs)} highly correlated feature pairs")
print(f"• Correlation analysis will help in feature selection by identifying redundant features")
print(f"• High correlation between features can lead to multicollinearity issues")
print("="*50)

# =============================================================================
# 3. DATA PRE-PROCESSING AND CLEANING
# =============================================================================

print("3. DATA PRE-PROCESSING AND CLEANING")
print("-" * 40)

# 3a. Data cleaning and preprocessing
print("3a. Data Cleaning:")

# Check for missing values
print(f"Missing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")

# Handle categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('binary_class')  # Remove target variable

print(f"Categorical columns to encode: {categorical_columns}")

# Label encoding for categorical features
label_encoders = {}
df_processed = df.copy()

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Convert boolean columns to numeric
boolean_columns = ['land', 'logged_in', 'root_shell', 'is_host_login', 'is_guest_login']
for col in boolean_columns:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].astype(int)

# Handle outliers using IQR method for key numeric features
print("\nOutlier detection and handling:")
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
outlier_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
outlier_cols = [col for col in outlier_cols if col in numeric_cols]

outliers_removed = 0
for col in outlier_cols:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_before = len(df_processed)
    df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
    outliers_after = len(df_processed)
    
    removed = outliers_before - outliers_after
    outliers_removed += removed
    if removed > 0:
        print(f"  {col}: Removed {removed} outliers")

print(f"Total outliers removed: {outliers_removed}")

print("\n📝 PRE-PROCESSING STEPS PERFORMED:")
print("• Checked for missing values and duplicates")
print("• Applied label encoding to categorical features")
print("• Converted boolean features to numeric")
print("• Removed outliers using IQR method for key features")
print("• Created binary classification target variable")

# 3b. Feature Engineering and Transformation
print("\n3b. Feature Engineering and Transformation:")

# Separate features and target
X = df_processed.drop(['class', 'binary_class'], axis=1)
y = df_processed['binary_class']

# Convert target to binary (0: Normal, 1: Attack)
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# Feature importance analysis
print("\nFeature Importance Analysis:")

# Method 1: Statistical test (F-test)
selector_f = SelectKBest(score_func=f_classif, k=20)
X_selected_f = selector_f.fit_transform(X, y_encoded)
f_scores = selector_f.scores_
f_feature_names = X.columns[selector_f.get_support()]

# Method 2: Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=20)
X_selected_mi = selector_mi.fit_transform(X, y_encoded)
mi_scores = selector_mi.scores_
mi_feature_names = X.columns[selector_mi.get_support()]

# Plot feature importance
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
top_f_indices = np.argsort(f_scores)[-15:]
plt.barh(range(15), f_scores[top_f_indices])
plt.yticks(range(15), X.columns[top_f_indices])
plt.title('Top 15 Features - F-test Scores')
plt.xlabel('F-test Score')

plt.subplot(1, 2, 2)
top_mi_indices = np.argsort(mi_scores)[-15:]
plt.barh(range(15), mi_scores[top_mi_indices])
plt.yticks(range(15), X.columns[top_mi_indices])
plt.title('Top 15 Features - Mutual Information')
plt.xlabel('Mutual Information Score')

plt.tight_layout()
plt.show()

# Feature scaling
print("\nFeature Scaling:")
print("Applying StandardScaler to normalize features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Alternative: MinMaxScaler (commented out)
# scaler_mm = MinMaxScaler()
# X_scaled_mm = scaler_mm.fit_transform(X)

print("✓ StandardScaler applied")

print("\n🔧 FEATURE ENGINEERING JUSTIFICATION:")
print("• StandardScaler chosen over MinMaxScaler because:")
print("  - Network traffic data often has different scales (bytes vs counts)")
print("  - StandardScaler handles outliers better than MinMaxScaler")
print("  - Logistic Regression and KNN benefit from standardized features")
print("• Feature selection helps reduce dimensionality and improve performance")
print("• F-test identifies features with strong linear relationships")
print("• Mutual Information captures non-linear relationships")
print("="*50)

# =============================================================================
# 4. MODEL BUILDING
# =============================================================================

print("4. MODEL BUILDING")
print("-" * 25)

# 4a. Dataset splitting
print("4a. Dataset Splitting:")

# Split 1: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"80-20 Split:")
print(f"  Training set: {X_train.shape}")
print(f"  Test set: {X_test.shape}")
print(f"  Training target distribution: {np.bincount(y_train)}")
print(f"  Test target distribution: {np.bincount(y_test)}")

# Split 2: Alternative split (70-30)
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\n70-30 Split (Alternative):")
print(f"  Training set: {X_train_alt.shape}")
print(f"  Test set: {X_test_alt.shape}")

print("\n📋 SPLIT JUSTIFICATION:")
print("• 80-20 split provides sufficient training data while maintaining test set size")
print("• Stratified split ensures balanced representation of both classes")
print("• Alternative 70-30 split for comparison and validation")

# 4b. Model Implementation and Hyperparameter Tuning
print("\n4b. Model Implementation and Hyperparameter Tuning:")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Define hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Decision Tree': {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# Train and tune models
trained_models = {}
best_params = {}
cv_scores = {}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Training and tuning models...")
for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(
        model, param_grids[name], cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Store results
    trained_models[name] = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_
    cv_scores[name] = grid_search.best_score_
    
    print(f"  ✓ Best CV F1-Score: {cv_scores[name]:.4f}")
    print(f"  ✓ Best Parameters: {best_params[name]}")

print("\n🎯 HYPERPARAMETER TUNING JUSTIFICATION:")
print("• GridSearchCV with StratifiedKFold ensures robust parameter selection")
print("• F1-score chosen as scoring metric due to potential class imbalance")
print("• 5-fold CV provides good balance between computational cost and reliability")
print("• Parameters selected cover key aspects:")
print("  - Logistic Regression: Regularization strength and penalty type")
print("  - Decision Tree: Tree complexity and splitting criteria")
print("  - KNN: Neighborhood size and distance metrics")
print("  - Random Forest: Ensemble size and tree parameters")
print("="*50)

# =============================================================================
# 5. PERFORMANCE EVALUATION
# =============================================================================

print("5. PERFORMANCE EVALUATION")
print("-" * 30)

# 5a. Model Performance Comparison
print("5a. Performance Comparison:")

# Evaluate all models
results = {}
predictions = {}

for name, model in trained_models.items():
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC (if probability predictions available)
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = roc_auc_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    }
    
    predictions[name] = {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Create results DataFrame
results_df = pd.DataFrame(results).T
print("Performance Metrics:")
print(results_df.round(4))

# Visualization of results
plt.figure(figsize=(15, 12))

# Performance comparison
plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(trained_models))
width = 0.15

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, results_df[metric], width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width*2, results_df.index, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# ROC Curves
plt.subplot(2, 3, 2)
for name, pred_data in predictions.items():
    if pred_data['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, pred_data['y_pred_proba'])
        auc_score = results[name]['AUC-ROC']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion matrices
for i, (name, pred_data) in enumerate(predictions.items()):
    plt.subplot(2, 3, 3+i)
    cm = confusion_matrix(y_test, pred_data['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name}\nConfusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Detailed classification reports
print("\nDetailed Classification Reports:")
for name, pred_data in predictions.items():
    print(f"\n{name}:")
    print(classification_report(y_test, pred_data['y_pred'], 
                              target_names=['Normal', 'Attack']))

# 5b. Best Model Selection
print("\n5b. Best Model Selection:")

# Calculate overall score (weighted combination of metrics)
overall_scores = {}
weights = {'Accuracy': 0.2, 'Precision': 0.2, 'Recall': 0.2, 'F1-Score': 0.3, 'AUC-ROC': 0.1}

for name in results.keys():
    score = sum(results[name][metric] * weight for metric, weight in weights.items())
    overall_scores[name] = score

best_model_name = max(overall_scores, key=overall_scores.get)
best_model = trained_models[best_model_name]

print(f"🏆 BEST MODEL: {best_model_name}")
print(f"Overall Score: {overall_scores[best_model_name]:.4f}")
print(f"\nPerformance Metrics:")
for metric, value in results[best_model_name].items():
    print(f"  {metric}: {value:.4f}")

print(f"\nBest Parameters:")
for param, value in best_params[best_model_name].items():
    print(f"  {param}: {value}")

# Feature importance for the best model (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(15), feature_importance['importance'][:15])
    plt.yticks(range(15), feature_importance['feature'][:15])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop 10 Most Important Features:")
    for i, (idx, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

print(f"\n🎯 WHY {best_model_name.upper()} IS THE BEST MODEL:")

if best_model_name == 'Random Forest':
    print("• Excellent balance of precision and recall for both classes")
    print("• High AUC-ROC indicates good separability between normal and attack traffic")
    print("• Ensemble method reduces overfitting and improves generalization")
    print("• Feature importance provides interpretability for network administrators")
    print("• Robust to outliers and handles mixed data types well")
elif best_model_name == 'Logistic Regression':
    print("• Fast training and prediction, suitable for real-time detection")
    print("• Good interpretability with coefficient analysis")
    print("• Probabilistic output helps in setting detection thresholds")
    print("• Regularization prevents overfitting")
elif best_model_name == 'Decision Tree':
    print("• Highly interpretable with clear decision rules")
    print("• Handles non-linear relationships well")
    print("• No need for feature scaling")
    print("• Can capture complex interactions between features")
elif best_model_name == 'K-Nearest Neighbors':
    print("• Non-parametric approach adapts to data distribution")
    print("• Good performance on local patterns in network traffic")
    print("• Simple yet effective for this type of classification problem")

print("\n📊 FINAL RECOMMENDATIONS:")
print(f"• Deploy {best_model_name} for network intrusion detection")
print(f"• Monitor model performance regularly and retrain with new data")
print(f"• Consider ensemble of top 2-3 models for production deployment")
print(f"• Implement real-time feature scaling pipeline")
print(f"• Set appropriate threshold based on false positive tolerance")

print("\n" + "="*50)
print("🎉 NETWORK INTRUSION DETECTION ANALYSIS COMPLETE!")
print("="*50)