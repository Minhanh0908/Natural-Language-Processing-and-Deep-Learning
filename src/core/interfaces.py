from abc import ABC, abstractmethod
from typing import List 

class Tokenizer(ABC):
  """
  Abstract base class for a tokenizer.
  """

  @abstractmethod
  def tokenize(self, text: str) -> List[str]:
    """
    Splits a string into a list of tokens
    Args:
      text: The input string to tokenize 
    Returns: 
      A list of string tokens.
    """
    pass 
  
class Vectorizer(ABC):
  """
  Abstract base class for a vectorizer.
  """

  @abstractmethod
  def fit(self, corpus: List[str]) -> None:
    """
    Learn the vocabulary from a list of documents
    
    Args:
      corpus: A list of documents (strings).
    """
    pass

  @abstractmethod
  def transform(self, document: List[str]) -> List[List[int]]:
    """
    Transforms documents into a list of vectors. 

    Args:
      documents: A list of strings to transform 
    """
    pass 

  @abstractmethod
  def fit_transform(self, corpus: List[str]) -> List[List[int]]:
    """
    Perform fit and then transform on the same data.
    """
    pass 