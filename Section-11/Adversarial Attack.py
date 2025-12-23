from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = ["win a free iphone", "claim your prize", "meeting at 10am", "project deadline tomorrow"]
y = ['spam', "spam", "ham", "ham"]

vec = CountVectorizer()
X_vec = vec.fit_transform(X)
model = MultinomialNB().fit(X_vec, y)

msg1 = "you get a free"
print(msg1, "->", model.predict(vec.transform([msg1]))[0])

msg2 = "you get fr33 gift"
print(msg2, "->", model.predict(vec.transform([msg2]))[0])
