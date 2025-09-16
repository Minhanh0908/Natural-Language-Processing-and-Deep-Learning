from core.interfaces import Tokenizer
from typing import List
import re

class RegexTokenizer(Tokenizer):
  def __init__(self, pattern: str = r'\w+|[^\w\s]'):
    super().__init__()
    self.pattern = pattern

  def tokenize(self, text: str) -> List[str]:
    return re.findall(self.pattern, text.lower())