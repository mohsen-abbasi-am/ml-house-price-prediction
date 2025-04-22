import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

data_path = os.path.join('data', 'housing.csv')
df = pd.read_csv(data_path)

X = df[['area', 'bedrooms', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/linear_model.pkl')
