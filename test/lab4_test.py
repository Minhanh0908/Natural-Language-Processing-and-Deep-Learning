import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder

# Initialize the WordEmbedder with a pre-trained model
embedder = WordEmbedder("glove-wiki-gigaword-50")

# Vector for the word "king"
print("Vector for 'king':")
print(embedder.get_vector("king"))

# Similarity
print("\nSimilarity:")
print("king vs queen:", embedder.get_similarity("king", "queen"))
print("king vs man:", embedder.get_similarity("king", "man"))

# 3Top 10 most similar to “computer”
print("\nMost similar to 'computer':")
print(embedder.get_most_similar("computer"))

# Document embedding
doc_vec = embedder.embed_document("The queen rules the country.")
print("\nDocument vector:")
print(doc_vec)
print("Vector shape:", doc_vec.shape)
