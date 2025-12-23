from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = ["free iphone", "win prize", "meeting tomorrow", "project deadline"]
y = ["spam", "spam", "ham", "ham"]

y_poison = ["ham", "ham", "spam", "spam"]

vec = CountVectorizer()
X_vec = vec.fit_transform(X)

m1 = MultinomialNB().fit(X_vec, y)
print("Clean: ", m1.predict(vec.transform(["free prize"]))[0])

m2 = MultinomialNB().fit(X_vec, y_poison)
print("Poisoned: ", m2.predict(vec.transform(["free prize"]))[0])
