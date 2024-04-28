import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import seaborn as sns
import joblib

# Load data
transaction_data = pd.read_csv('data/transaction_data.csv')
fee_data = pd.read_csv('data/fee_structures.csv')

# Ensure 'amount' is numeric
transaction_data['amount'] = pd.to_numeric(transaction_data['amount'], errors='coerce')

# One-hot encode 'payment_method' and 'merchant_category' columns
encoder = OneHotEncoder(drop='first')
encoded_columns = encoder.fit_transform(transaction_data[['payment_method', 'merchant_category']])
encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(['payment_method', 'merchant_category']))

# Concatenate encoded columns with original dataframe
transaction_data = pd.concat([transaction_data, encoded_df], axis=1)
transaction_data.drop(['payment_method', 'merchant_category'], axis=1, inplace=True)

# Preprocess the 'amount' column
imputer = SimpleImputer(strategy='mean')
transaction_data['amount'] = imputer.fit_transform(transaction_data[['amount']].values)
scaler = MinMaxScaler()
transaction_data['amount_scaled'] = scaler.fit_transform(transaction_data[['amount']])

# Function to estimate fees
def estimate_fee(row):
    if fee_data.empty or not {'payment_method', 'fee_percentage'}.issubset(fee_data.columns):
        raise ValueError("fee_data must contain 'payment_method' and 'fee_percentage' columns with data.")
    payment_type_cols = ['payment_method_Credit Card', 'payment_method_Debit Card', 'payment_method_PayPal']
    for col in payment_type_cols:
        if col in row.index and row[col] == 1:
            payment_type = col.split('_')[-1]
            fee_percentage = fee_data.loc[fee_data['payment_method'] == payment_type, 'fee_percentage'].values[0]
            return fee_percentage * row['amount_scaled']
    return 0

# Add estimated fee feature
transaction_data['estimated_fee'] = transaction_data.apply(estimate_fee, axis=1)
transaction_data['estimated_fee'].fillna(0, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(transaction_data.drop(['estimated_fee'], axis=1), transaction_data['estimated_fee'], test_size=0.2)

# Train Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Train Ridge Regression model
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)

# Evaluate models
predicted_fees_lr = model_lr.predict(X_test)
predicted_fees_ridge = model_ridge.predict(X_test)

r_squared_lr = r2_score(y_test, predicted_fees_lr)
r_squared_ridge = r2_score(y_test, predicted_fees_ridge)
print(f"Linear Regression R-squared: {r_squared_lr:.2f}")
print(f"Ridge Regression R-squared: {r_squared_ridge:.2f}")

# Visualization of residuals for both models
plt.figure(figsize=(12, 6))
sns.residplot(x=predicted_fees_lr, y=y_test, lowess=True, color="g", label='Linear Regression')
sns.residplot(x=predicted_fees_ridge, y=y_test, lowess=True, color="b", label='Ridge Regression')
plt.title('Residual Plot')
plt.xlabel('Predicted Fees')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# Feature importance from Linear Regression
coefficients = pd.DataFrame(model_lr.coef_, X_train.columns, columns=['Coefficient'])
coefficients.plot(kind='barh')
plt.title('Feature Importance in Linear Regression Model')
plt.xlabel('Coefficient Value')
plt.show()

# Save models
joblib.dump(model_lr, 'Models/payment_fee_optimization_linear_model.pkl')
joblib.dump(model_ridge, 'Models/payment_fee_optimization_ridge_model.pkl')
