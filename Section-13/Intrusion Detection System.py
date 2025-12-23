import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = {
    "packet_count": [20, 25, 30, 200, 220, 250],
    "unique_ports": [2, 3, 4, 50, 60, 70],
    "label": ["normal","normal","normal","scan","scan","scan"]
}

df = pd.DataFrame(data)

X = df[["packet_count", "unique_ports"]]
y = df[["label"]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print("Training: ", train_acc)
print("Testing: ", test_acc)


y_pred = model.predict(X_test)
print("Classification report: ", classification_report(y_test, y_pred))

sample = [[40,5], [180,45], [2000, 30000], [10,1]]
print("\nDemo Prediction: ")
for s, p in zip(sample, model.predict(sample)):
    print(f"Input={s} -> {p}")
