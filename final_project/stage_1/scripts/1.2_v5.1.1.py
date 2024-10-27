# Unsupervised Evaluation
# Method 5: Anomaly Detection with One-Class SVM

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import OneClassSVM

# Load data
train_data = pd.read_csv('./train_data.csv')
test_data = pd.read_csv('./same_season_test_data.csv')

# Update the target variable for test data using One-Class SVM
# Select Relevant Features
features = [
    'home_team_wins_mean',
    'away_team_wins_mean',
    'home_batting_batting_avg_mean',
    'away_batting_batting_avg_mean'
]
X = test_data[features].fillna(0)

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply One-Class SVM
oc_svm = OneClassSVM(gamma='auto')
test_data['anomaly'] = oc_svm.fit_predict(X_scaled)

# Assign Proxy Target Variable
# Assume anomalies represent away team wins (less frequent)
test_data['home_team_win'] = (test_data['anomaly'] == 1).astype(int)

# Drop date columns from the datasets
train_data.drop(columns=['date'], inplace=True, errors='ignore')

# Identify categorical columns
categorical_cols = ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season']
numeric_cols = [col for col in train_data.columns if col not in categorical_cols + ['home_team_win']]

# Creating pipelines for the preprocessing of numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Column transformer for applying different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split features and target
X_train = train_data.drop(columns=['home_team_win'])
y_train = train_data['home_team_win']
X_test = test_data.drop(columns=['home_team_win'])
y_test = test_data['home_team_win']

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Step 1: Outlier Detection with Modified Threshold
def christoffel_darboux_kernel_outliers(data, n_neighbors=5, threshold=7.0):
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, _ = nn.kneighbors(data)
    kernel_density_scores = np.mean(distances, axis=1)
    
    outliers = np.where(kernel_density_scores > threshold)[0]
    return outliers

# Detect and remove outliers from training data
outliers = christoffel_darboux_kernel_outliers(X_train_preprocessed)
print(f"Initial Outliers Detected: {len(outliers)}")

# Adjust outlier removal to prevent empty dataset
if len(outliers) < X_train_preprocessed.shape[0] * 0.8:  # Keep at least 20% of the data
    X_train_filtered = X_train_preprocessed[np.isin(range(len(X_train)), outliers, invert=True)]
    y_train_filtered = y_train.drop(outliers).reset_index(drop=True)
else:
    print("Too many outliers detected; skipping outlier removal.")
    X_train_filtered = X_train_preprocessed
    y_train_filtered = y_train

# Step 2: Feature Engineering with ARMA for Temporal Dependencies
win_ratios = train_data.groupby('home_team_abbr')['home_team_win'].apply(list)
forecasted_ratios = {}
for team, ratios in win_ratios.items():
    if len(ratios) > 5:
        try:
            model = ARIMA(ratios, order=(1, 0, 1))  # Simplified ARIMA model
            arma_model = model.fit()
            forecast = arma_model.forecast(steps=1)
            forecasted_ratios[team] = forecast[0]
        except Exception as e:
            print(f"ARIMA model failed for team {team}: {e}")

train_data['forecasted_win_ratio'] = train_data['home_team_abbr'].map(forecasted_ratios).fillna(0)

# Step 3: Baseline Models - Logistic Regression and Decision Tree
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_filtered, y_train_filtered)
logreg_pred = logreg.predict(X_test_preprocessed)
print("Logistic Regression - Accuracy:", accuracy_score(y_test, logreg_pred))

dt = DecisionTreeClassifier()
dt.fit(X_train_filtered, y_train_filtered)
dt_pred = dt.predict(X_test_preprocessed)
print("Decision Tree - Accuracy:", accuracy_score(y_test, dt_pred))

# Step 4: Custom Online Learning Model - Mondrian Forest Approximation with Incremental Updates
forest = RandomForestClassifier(n_estimators=10, random_state=42)
forest.fit(X_train_filtered, y_train_filtered)

# Function to incrementally update the model
def incremental_update(model, new_data, new_labels):
    model.fit(new_data, new_labels)
    return model

# Initial evaluation
forest_pred = forest.predict(X_test_preprocessed)
print("Initial Mondrian Forest Approximation - Accuracy:", accuracy_score(y_test, forest_pred))

# Step 5: Simulate Incremental Learning with Time-Based Data Batches
batch_size = 50  # Define batch size for incremental updates
num_batches = X_test_preprocessed.shape[0] // batch_size

for i in range(num_batches):
    start = i * batch_size
    end = (i + 1) * batch_size
    X_batch = X_test_preprocessed[start:end]
    y_batch = y_test.iloc[start:end]
    
    # Evaluate before update
    batch_pred = forest.predict(X_batch)
    batch_accuracy = accuracy_score(y_batch, batch_pred)
    print(f"Batch {i+1} Accuracy before update: {batch_accuracy:.2f}")
    
    # Incremental update
    forest = incremental_update(forest, X_batch, y_batch)

    # Evaluate after update
    batch_pred_updated = forest.predict(X_batch)
    batch_accuracy_updated = accuracy_score(y_batch, batch_pred_updated)
    print(f"Batch {i+1} Accuracy after update: {batch_accuracy_updated:.2f}")

# Step 6: Model Evaluation - Accuracy and 0/1 Error
models = {
    "Logistic Regression": logreg,
    "Decision Tree": dt,
    "Mondrian Forest Approximation": forest
}

for model_name, model in models.items():
    y_pred = model.predict(X_test_preprocessed)
    accuracy = accuracy_score(y_test, y_pred)
    error = zero_one_loss(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.2f}, 0/1 Error: {error:.2f}")
