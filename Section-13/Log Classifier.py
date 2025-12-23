import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

data = {
    "log": [
        "System started successfully",          # INFO
        "User login successful",                # INFO
        "Disk error detected",                  # ERROR
        "Application crashed with code 500",    # ERROR
        "Multiple failed login attempts",       # SECURITY
        "Suspicious connection detected"        # SECURITY
    ],
    "label": ["INFO","INFO","ERROR","ERROR","SECURITY","SECURITY"]
}
df = pd.DataFrame(data)

X = df["log"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

vec = CountVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

samples = [
    "software started",
    "Unauthorized login attampt detected",
    "Critical application failure",

]

print("\n Demo Prediction: ")
for s in samples:
    print(s, "->", model.predict(vec.transform([s]))[0])
