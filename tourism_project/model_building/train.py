# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/nv185001/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/nv185001/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/nv185001/Tourism-Package-Prediction/ytrain.csv"
ytest_path = "hf://datasets/nv185001/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# List of numerical features in the dataset
numeric_features = [
    'Age',       # Age of the customer
    'CityTier',  # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
    'DurationOfPitch',   # Duration of the sales pitch delivered to the customer
    'NumberOfPersonVisiting', # Total number of people accompanying the customer on the trip.
    'NumberOfFollowups',      # Total number of follow-ups by the salesperson after the sales pitch.
    'PreferredPropertyStar',  # Preferred hotel rating by the customer.
    'NumberOfTrips', # Average number of trips the customer takes annually.
    'Passport', # Whether the customer holds a valid passport (0: No, 1: Yes).
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch.
    'OwnCar', # Whether the customer owns a car (0: No, 1: Yes)
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer.
    'MonthlyIncome' # Gross monthly income of the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',  # The method by which the customer was contacted (Company Invited or Self Inquiry) 
    'Occupation',     # Customer's occupation (e.g., Salaried, Freelancer) 
    'Gender',         # Gender of the customer (Male, Female).
    'ProductPitched', # The type of product pitched to the customer.
    'MaritalStatus',  # Marital status of the customer (Single, Married, Divorced).
    'Designation'     # Customer's designation in their current organization
]

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# Check the parameters of the best model
grid_search.best_params_

# Store the best model
best_model = grid_search.best_estimator_
best_model

# Set the classification threshold
classification_threshold = 0.45

# Make predictions on the training data
y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

# Make predictions on the test data
y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

# Generate a classification report to evaluate model performance on training set
print(classification_report(ytrain, y_pred_train))

# Generate a classification report to evaluate model performance on test set
print(classification_report(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "best_tourism_prediction_model.joblib")

# Upload to Hugging Face
repo_id = "nv185001/churn-model"
repo_type = "model"

api = HfApi(token=os.getenv("PRJ_HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("churn-model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="best_tourism_prediction_model.joblib",
    path_in_repo="best_tourism_prediction_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)



