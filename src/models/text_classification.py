from src.core.interfaces import Vectorizer
from sklearn.linear_model import LogisticRegression
from typing import List, Dict 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

class TextClassifier(): 
  def __init__(self, vectorizer: Vectorizer):
    self.vectorizer = vectorizer
    self._model = None 

  def fit(self, text: List[str], labels: List[str]):
    X = self.vectorizer.fit_transform(text)

    self._model = LogisticRegression(solver='liblinear', random_state=42)
    self._model.fit(X, labels)

  def predict(self, texts: List[str]): 
    if self._model is None:
      raise ValueError("Model hasn't been trained.")

    X = self.vectorizer.transform(texts)
    return self._model.predict(X)
  
  def evaluate(self, y_true: List[int], y_pred = List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
            }