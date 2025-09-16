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
  