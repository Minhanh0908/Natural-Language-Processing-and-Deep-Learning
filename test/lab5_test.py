import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.models.text_classification import TextClassifier 
from sklearn.model_selection import train_test_split

texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

model = TextClassifier(vectorizer)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = model.evaluate(y_test, y_pred)
print("Evaluation metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


