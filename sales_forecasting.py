import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load dataset
df = pd.read_csv("sales_data.csv")
df['order_value_EUR'] = df['order_value_EUR'].astype(str).str.replace(',', '')
df['order_value_EUR'] = df['order_value_EUR'].astype(float)

# Convert date
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date')

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Extract date features
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Encode categorical columns
df['country'] = df['country'].astype('category').cat.codes
df['category'] = df['category'].astype('category').cat.codes
df['device_type'] = df['device_type'].astype('category').cat.codes
df['sales_manager'] = df['sales_manager'].astype('category').cat.codes
df['sales_rep'] = df['sales_rep'].astype('category').cat.codes

# -------------------------------
# FEATURES & TARGET
# -------------------------------
features = [
    'country', 'cost', 'category',
    'device_type', 'sales_manager',
    'sales_rep', 'day', 'month', 'year'
]

X = df[features]
y = df['order_value_EUR']

# -------------------------------
# TRAIN MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# PREDICTION ON TEST DATA
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# FUTURE SIMULATION (NEXT 30 ROWS STYLE)
# -------------------------------
future = df.tail(30).copy()

future['day'] = future['date'].dt.day
future['month'] = future['date'].dt.month
future['year'] = future['date'].dt.year

future_X = future[features]

future['predicted_sales'] = model.predict(future_X)

# Save output
future[['date', 'predicted_sales']].to_csv("forecast_output.csv", index=False)

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['order_value_EUR'], label="Actual Sales")
plt.plot(future['date'], future['predicted_sales'], label="Predicted Sales", color='red')

plt.title("Sales Forecasting (ML Model)")
plt.xlabel("Date")
plt.ylabel("Order Value (EUR)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ Forecast completed successfully!")