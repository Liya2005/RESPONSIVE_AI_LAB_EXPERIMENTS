import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")

data = pd.read_csv("dataset.csv")

X = data[["fever", "cough", "headache"]]
y = data["disease"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "model/disease_model.pkl")

print("Model trained and saved successfully!")