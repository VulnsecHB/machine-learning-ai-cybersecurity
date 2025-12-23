from sklearn.feature_extraction.text import CountVectorizer

docs = ["win a free iphone", "meeting tomorrow"]
vec = CountVectorizer()
X = vec.fit_transform(docs)

print(vec.get_feature_names_out())
print(X.toarray())
