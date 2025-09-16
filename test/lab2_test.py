import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.core.dataset_loaders import load_raw_text_data

logging.basicConfig(
    filename=os.path.join("logging", "lab2_test.log"),
    filemode="w",
    level=logging.INFO,
    format="%(message)s"
)

def log_and_print(msg: str):
    print(msg)
    logging.info(msg)

def main():
    log_and_print("---CountVectorizer Test---")

    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    log_and_print(f"\nSample corpus: {corpus}")

    # Initialize tokenizer and vectorizer
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    # Fit the vectorizer and transform the corpus
    vectors = vectorizer.fit_transform(corpus)

    log_and_print(f"\nVocabulary: {vectorizer.vocabulary_}")
    log_and_print("\nDocument Vectors:")
    for vec in vectors:
        log_and_print(f"Vector: {vec}")
    
    # Test with a real dataset
    log_and_print("\n---Count Vectorization with UD_English-EWT Dataset---")
    dataset_path = r"D:\NLP_DL\data\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    documents = raw_text.split("\n\n")[:5]  # Take first 5 documents for testing

    log_and_print(f"\nSample documents: {documents}")
    vectors = vectorizer.fit_transform(documents)

    log_and_print(f"\nVocabulary: {vectorizer.vocabulary_}")
    log_and_print("\nDocument Vectors:")
    for vec in vectors:
        log_and_print(f"Vector: {vec}")


if __name__ == "__main__":
    main()