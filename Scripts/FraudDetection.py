import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
import joblib

# Load data
transaction_data = pd.read_csv('data/fraud_data.csv')

# Convert 'flagged_transaction' to binary labels
transaction_data['fraud_label'] = (transaction_data['flagged_transaction'] == 'fraud').astype(int)

# Select features and target variable
features = ['amount', 'time_of_day', 'point_of_transaction', 'country']
X = transaction_data[features]
y = transaction_data['fraud_label']

# Preprocessing: Standardize numerical features and one-hot encode categorical features
numerical_features = ['amount', 'time_of_day']
categorical_features = ['point_of_transaction', 'country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create a pipeline that first preprocesses the data then fits the RandomForest classifier
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Create a pipeline for Logistic Regression for comparison
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CalibratedClassifierCV(LogisticRegression(random_state=42), cv=3))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest classifier within the pipeline
pipeline_rf.fit(X_train, y_train)

# Train the Logistic Regression within the pipeline
pipeline_lr.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_lr = pipeline_lr.predict(X_test)

# Evaluate the classifiers
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Plotting confusion matrix for both models
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", ax=ax[0])
ax[0].set_title('Random Forest Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", ax=ax[1])
ax[1].set_title('Logistic Regression Confusion Matrix')

# ROC Curve for both models
fpr_rf, tpr_rf, _ = roc_curve(y_test, pipeline_rf.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, pipeline_lr.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = %0.2f)' % roc_auc_rf)
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save the models into the /Models directory
joblib.dump(pipeline_rf, 'Models/random_forest_fraud_detection_model.pkl')
joblib.dump(pipeline_lr, 'Models/logistic_regression_fraud_detection_model.pkl')
