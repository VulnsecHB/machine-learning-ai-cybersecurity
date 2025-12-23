import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

data = {
    "user": [
        "alice","alice","alice","bob","bob","bob","chris","chris","chris","dina","dina","dina"
    ],
    "hour": [9, 14, 3, 10, 18, 2, 14, 11, 22, 8, 17, 12], 
    "failed_attempts": [0, 0, 4, 0, 1, 0, 0, 0, 3, 0, 0, 0], 
}

df = pd.DataFrame(data)

X = df[["hour", "failed_attempts"]]

model = IsolationForest(contamination=0.25, random_state=42)
model.fit(X)

df["anomaly"] = model.predict(X)
df["anomaly"] = df["anomaly"].map({1:"normal", -1:"anomaly"})

print("\n Detection result: \n", df)

print("\n Flagged anomalies: ")
print(df[df["anomaly"]=="anomaly"])
