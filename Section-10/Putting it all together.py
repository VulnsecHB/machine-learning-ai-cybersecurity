from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

X = ["win free iphone", "meeting tomorrow at 10am", "reset password urgent"]
y = ["spam", "ham", "spam"]


model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X,y)

print(model.predict(["Lunch meeting at 10pm"]))
print(model.predict(["Claim your gift free tomorrow"]))
