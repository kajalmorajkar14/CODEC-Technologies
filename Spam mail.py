import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "label": [0, 1, 0, 1],
    "message": [
        "Hello how are you",
        "Win a free mobile now",
        "Let's meet tomorrow",
        "Congratulations you won cash prize"
    ]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

test = ["Free prize waiting for you"]
result = model.predict(vectorizer.transform(test))

print("Prediction:", "SPAM" if result[0] == 1 else "NOT SPAM")
