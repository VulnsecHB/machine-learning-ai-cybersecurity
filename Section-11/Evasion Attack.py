from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = ["buy cheap meds", "earn money fast", "meeting at 10am", "see you tomorrow"]
y = ["spam", "spam", "ham", "ham"]

vec = CountVectorizer()
X_vec = vec.fit_transform(X)
model = MultinomialNB().fit(X_vec, y)

msg1 = "buy cheap iphone"
print(msg1, "->", model.predict(vec.transform([msg1]))[0])

msg2 = "b u y c h e a p iphone"
print(msg2, "->", model.predict(vec.transform([msg2]))[0])
