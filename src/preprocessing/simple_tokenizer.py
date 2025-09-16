from src.core.interfaces import Tokenizer
from typing import List
import string 

class SimpleTokenizer(Tokenizer):
  def __init__(self):
    super().__init__()

  def tokenize(self, text: str) -> List[str]:
    # Replace punctuation with spaces
    for p in string.punctuation:
      text = text.replace(p, f" {p} ")
    
    return text.lower().split()
