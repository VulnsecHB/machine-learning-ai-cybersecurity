import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "text": [
        "Win a free iPhone now",
        "Lowest price for your meds",
        "Meeting at 10am tomorrow",
        "Project deadline is next week",
        "Claim your lottery prize today",
        "Your invoice is attached",
        "Urgent: Reset your password",
        "Lunch at the canteen?",
        "Congratulations, you won money",
        "Reminder: doctor's appointment"
    ],
    "label": [
        "spam","spam","ham","ham","spam",
        "ham","spam","ham","spam","ham"
    ]
}
df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english")),
    ("nb", MultinomialNB())
])

pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n ", classification_report(y_test, preds))
print("Confusion Matrix: \n", confusion_matrix(y_test, preds, labels=["spam", "ham"]))


samples = [
    "Congratulations, you won free tickets",
    "Let's meet for lunch tomorrow",
    "Urgent: verify your account immediately",
    "Project meeting scheduled next week"
]

print("\n ------Demo------")
for s,p in zip(samples, pipe.predict(samples)):
    print(f"{p.upper():4s} | {s}")
