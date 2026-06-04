import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("car data.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Convert categorical columns
le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])
df['Car_Name'] = le.fit_transform(df['Car_Name'])

# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("\n==============================")
print("CAR PRICE PREDICTION")
print("==============================")
print("R2 Score :", round(score * 100, 2), "%")

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.savefig("car_price_graph.png")
plt.show()