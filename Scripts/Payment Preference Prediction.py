import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns

# Load data
transaction_data = pd.read_csv('data/customer_payment_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
transaction_data['payment_method'] = imputer.fit_transform(transaction_data[['payment_method']]).ravel()

# One-hot encode 'payment_method' column
encoder = OneHotEncoder(drop='first')  
payment_method_encoded = encoder.fit_transform(transaction_data[['payment_method']])
payment_method_encoded_df = pd.DataFrame(payment_method_encoded.toarray(), columns=encoder.get_feature_names_out(['payment_method']))
transaction_data = pd.concat([transaction_data, payment_method_encoded_df], axis=1)

# Encode 'gender' column
gender_mapping = {'Male': 0, 'Female': 1}
transaction_data['gender'] = transaction_data['gender'].map(gender_mapping)

# Feature engineering
transaction_data['age_income_ratio'] = transaction_data['age'] / transaction_data['income']
transaction_data['amount_mean'] = transaction_data.groupby('customer_id')['amount'].transform('mean')

# Split data
features = transaction_data.drop(['transaction_id', 'customer_id', 'payment_method'] + list(payment_method_encoded_df.columns), axis=1)
label = transaction_data['payment_method']
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Train RandomForest model
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)

# Train GradientBoosting model for comparison
clf_gb = GradientBoostingClassifier(random_state=42)
clf_gb.fit(X_train, y_train)

# Evaluate RandomForest model
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Evaluate GradientBoosting model
y_pred_gb = clf_gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting Accuracy:", accuracy_gb)
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

# Feature Importance Visualization for RandomForest
feature_importances = pd.Series(clf_rf.feature_importances_, index=features.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features in RandomForest Model')
plt.show()

# Confusion Matrix Visualization
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Save the models into the /Models directory
joblib.dump(clf_rf, 'Models/RandomForest_Payment_Preference_Prediction.pkl')
joblib.dump(clf_gb, 'Models/GradientBoosting_Payment_Preference_Prediction.pkl')
