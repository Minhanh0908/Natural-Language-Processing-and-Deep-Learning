from src.core.interfaces import Vectorizer, Tokenizer
from typing import List, Dict

class CountVectorizer(Vectorizer):
  """
  Represents documents as vectors of token counts. 
  """
  def __init__(self, tokenizer: Tokenizer):
    self.tokenizer = tokenizer
    self.vocabulary_: Dict[str, int] = {}
  
  def fit(self, corpus: List[str]) -> None:
    """
    Build the vocabulary from the corpus.
    """
    unique_tokens = set()

    for doc in corpus: 
      tokens = self.tokenizer.tokenize(doc)
      unique_tokens.update(tokens)

    self.vocabulary_ = {token: id for id, token in enumerate(sorted(unique_tokens))}

  def transform(self, documents: List[str]) -> List[List[int]]:
    """
    Transform documents to their vector representation.
    """
    if not self.vocabulary_:
      raise RuntimeError("Vectorizer has not been fitted yet. Call fit() first")

    vectors = []
    vocab_size = len(self.vocabulary_)

    for doc in documents: 
      vector = [0] * vocab_size
      tokens = self.tokenizer.tokenize(doc)

      for token in tokens: 
        if token in self.vocabulary_:
          id = self.vocabulary_[token]
          vector[id] += 1 
        
      vectors.append(vector)
      
    return vectors 
  
  def fit_transform(self, corpus):
    """
    Perform fit and then transform on the corpus. 
    """
    self.fit(corpus)
    return self.transform(corpus)