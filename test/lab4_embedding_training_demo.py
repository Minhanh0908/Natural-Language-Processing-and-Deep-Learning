import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os 
from gensim.models import Word2Vec 
from gensim.utils import simple_preprocess

def load_raw_text_data(file_path: str): 
  """Load raw text data from a file."""

  print("Reading data from:", file_path)
  sentences = []
  with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      tokens = simple_preprocess(line)
      if tokens:
        sentences.append(tokens)
  return sentences

def train_word2vec(sentences, vector_size=100, window=5, min_count=2, epochs=10):
  print("Training Word2Vec model...")
  model = Word2Vec(
    sentences=sentences, 
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=os.cpu_count() - 1, 
    epochs=epochs
  )
  print("Model training completed.")
  return model
def save_model(model, save_path="results/word2vec_ewt.model"):
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  model.save(save_path)
  print(f"Model saved to {save_path}")

def test_model(model):
  try:
      print("Top 10 similar to 'computer':")
      print(model.wv.most_similar("computer", topn=10))
  except KeyError:
      print("Word 'computer' not in vocabulary.")
  try:
      print("\nAnalogy test: man → king :: woman → ?")
      result = model.wv.most_similar(positive=["woman", "king"], negative=["man"], topn=1)
      print(result)
  except KeyError as e:
      print(f"⚠️ Missing word in analogy: {e}")

if __name__ == "__main__":
    data_path = "data/UD_English-EWT/en_ewt-ud-train.txt"

    # Step 1: load data
    sentences = load_raw_text_data(data_path)

    # Step 2: train model
    model = train_word2vec(sentences)

    # Step 3: save model
    save_model(model)

    # Step 4: demonstrate usage
    test_model(model)