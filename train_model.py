import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv", encoding="utf-8-sig")

X = data["message"]
y = data["label"].map({"ham":0,"spam":1})

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vector, y)

pickle.dump(model, open("spam_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model trained and saved")