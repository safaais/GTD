
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('terror.csv', encoding='ISO-8859-1')

# Preprocess the data
# Replace 0 and 1 in the 'success' column with 'no' and 'yes'
df['success'] = df['success'].map({0: 'no', 1: 'yes'})

# Drop rows where the target variable is missing
df = df.dropna(subset=['success'])

# Select a broader set of features
features = [
    'iyear', 'imonth', 'iday', 'extended', 'country_txt', 'region_txt', 
    'latitude', 'longitude', 'specificity', 'attacktype1_txt', 'targtype1_txt', 
    'targsubtype1_txt', 'weaptype1_txt', 'weapsubtype1_txt', 'nkill', 'nwound', 
    'property', 'ishostkid', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY'
]

# Filter the dataset to include only selected features
df = df[features + ['success']]

# Handle missing values
# Fill missing numerical values with the median
numerical_features = ['latitude', 'longitude', 'specificity', 'nkill', 'nwound', 'property']
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

# Fill missing categorical values with the mode
categorical_features = ['country_txt', 'region_txt', 'attacktype1_txt', 'targtype1_txt', 
                        'targsubtype1_txt', 'weaptype1_txt', 'weapsubtype1_txt']
df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

# Remove outliers in numerical features using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in numerical_features:
    df = remove_outliers(df, col)

# Encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Separate features and target
X = df.drop(columns=['success'])
y = df['success']

# Encode the target variable (y) into numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts 'no' -> 0, 'yes' -> 1

# Check for missing values in X
print("Missing values in X before imputation:")
print(X.isnull().sum())

# Impute any remaining missing values (if necessary)
imputer = SimpleImputer(strategy='median')  # Use 'median' for numerical features
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Check for missing values in X after imputation
print("Missing values in X after imputation:")
print(X.isnull().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Initialize individual models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train and evaluate individual models
models = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'AdaBoost': ada_model,
    'XGBoost': xgb_model
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))  # Use original class names
    print(f"Accuracy for {name}: {accuracy_score(y_test, y_pred)}")
    print("-" * 60)
