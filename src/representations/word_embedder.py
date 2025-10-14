import gensim.downloader as api 
from src.preprocessing.regex_tokenizer import RegexTokenizer
import numpy as np 

class WordEmbedder():

  def __init__(self, model_name: str):
    """
     Load a pre-trained word embedding model from gensim-data.
    """
    try: 
      print(f"Loading model: {model_name}...")
      self.model = api.load(model_name)
      print("Model loaded successfully.")
    except ValueError as e:
      print(f"Model '{model_name}' not found in gensim-data.")
      available = list(api.info()['models'].keys())
      print(f"Available models are: {available}")
      raise ValueError(
        f"Model '{model_name}' not found. Choose from the list above."
      ) from e
    except Exception as e:
            raise RuntimeError(f"Error loading model '{model_name}': {e}") from e
    
  def get_vector(self, word: str):
    """
    Return the embedding vector for a given word. OOV cases return None
    """
    try: 
      return self.model[word]
    except KeyError:

      print(f'Word {word} not found in vocabulary.')
      return None 
  
  def get_similarity(self, word1: str, word2: str):
    """
    Return the cosine similarity between two words. OOV cases return None"""
    try:
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is None or vec2 is None:
            print("One or both words are out of vocabulary.")
            return None 
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return None
    
  def get_most_similar(self, word: str, top_n: int=10):
    """
    Return the top_n most similar words to a given word.
    """
    try:
        similar_words = self.model.most_similar(word, topn=top_n)
        return similar_words
    except KeyError:
        print(f'Word {word} not found in vocabulary.')
        return None
    except Exception as e:
        print(f"Error finding most similar words: {e}")
        return None

  def embed_document(self, document: str):
    """
        Represent a document as the mean of its word vectors.
        - Tokenize the document using RegexTokenizer.
        - Ignore OOV words.
        - Return zero vector if no valid words found.
    """
    tokenizer = RegexTokenizer()
    tokens = tokenizer.tokenize(document)
    valid_vectors = []
            
    for tok in tokens:
      vec = self.get_vector(tok)
      if vec is not None:
          valid_vectors.append(vec)
      if not valid_vectors:
          # Return zero vector with correct dimension
          dim = self.model.vector_size
          return np.zeros(dim)
    
    return np.mean(valid_vectors, axis=0)