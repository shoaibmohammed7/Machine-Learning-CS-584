import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

# Function to return a list of bank holidays
def get_bank_holidays():
    return [
        '2023-01-01',  # New Year's Day
        '2023-04-07',  # Good Friday
        '2023-04-10',  # Easter Monday
        '2023-05-01',  # Early May bank holiday
        '2023-05-29',  # Spring bank holiday
        '2023-08-28',  # Summer bank holiday
        '2023-12-25',  # Christmas Day
        '2023-12-26',  # Boxing Day
        '2023-01-16',  # Martin Luther King Jr. Day
        '2023-02-20',  # Washington's Birthday
        '2023-05-29',  # Memorial Day
        '2023-07-04',  # Independence Day
        '2023-09-04',  # Labor Day
        '2023-11-23'   # Thanksgiving Day
    ]

# Load data
payment_data = pd.read_csv('data/payment_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
payment_data['due_date'] = pd.DataFrame(imputer.fit_transform(payment_data[['due_date']]))
payment_data['due_date'] = pd.to_datetime(payment_data['due_date'])

# Feature engineering
payment_data['day'] = payment_data['due_date'].dt.day
payment_data['month'] = payment_data['due_date'].dt.month
payment_data['year'] = payment_data['due_date'].dt.year
payment_data['is_weekend'] = payment_data['due_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
payment_data['is_holiday'] = payment_data['due_date'].dt.date.isin(get_bank_holidays()).astype(int)

# One-hot encode 'payment_method' column
encoder = OneHotEncoder()
encoded_payment_method = encoder.fit_transform(payment_data[['payment_method']]).toarray()
encoded_payment_method_df = pd.DataFrame(encoded_payment_method, columns=encoder.get_feature_names_out(['payment_method']))

# Concatenate encoded columns with original dataframe
payment_data = pd.concat([payment_data, encoded_payment_method_df], axis=1)
payment_data.drop(['due_date', 'payment_method'], axis=1, inplace=True)

# Split data
X = payment_data.drop(['payment_processing_time'], axis=1)
y = payment_data['payment_processing_time']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, predictions_lr)
mse_lr = mean_squared_error(y_test, predictions_lr)

# Train Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, predictions_rf)
mse_rf = mean_squared_error(y_test, predictions_rf)

# Comparison of evaluation metrics
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
metrics_lr = [mae_lr, mse_lr]
metrics_rf = [mae_rf, mse_rf]
index = ['MAE', 'MSE']
df_metrics = pd.DataFrame({'Linear Regression': metrics_lr, 'Random Forest': metrics_rf}, index=index)
df_metrics.plot.bar(ax=ax[0])
ax[0].set_title('Comparison of Evaluation Metrics')
ax[0].set_ylabel('Error Value')

# Residual plot
residuals_lr = y_test - predictions_lr
residuals_rf = y_test - predictions_rf
ax[1].scatter(predictions_lr, residuals_lr, color='blue', alpha=0.5, label='Linear Regression')
ax[1].scatter(predictions_rf, residuals_rf, color='green', alpha=0.5, label='Random Forest')
ax[1].hlines(y=0, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], colors='red')
ax[1].set_title('Residual Plot')
ax[1].set_xlabel('Predicted Values')
ax[1].set_ylabel('Residuals')
ax[1].legend()

plt.tight_layout()
plt.show()

# Save models
joblib.dump(model_lr, 'Models/linear_regression_payment_timing_model.pkl')
joblib.dump(model_rf, 'Models/random_forest_payment_timing_model.pkl')

print(f"Linear Regression - MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}")
print(f"Random Forest - MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}")
