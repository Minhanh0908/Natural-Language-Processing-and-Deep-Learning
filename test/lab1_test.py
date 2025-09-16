import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data


# Setup logger
logging.basicConfig(
    filename=os.path.join("logging", "lab1_test.log"),
    filemode="w",
    level=logging.INFO,
    format="%(message)s"
)

def log_and_print(msg: str):
    print(msg)
    logging.info(msg)


def test_sentences():
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    log_and_print("Simple Tokenizer Results:")
    for s in sentences:
        log_and_print(f"Original: {s}")
        log_and_print(f"Tokenized: {simple_tokenizer.tokenize(s)}")

    log_and_print("\nRegex Tokenizer Results:")
    for s in sentences:
        log_and_print(f"Original: {s}")
        log_and_print(f"Tokenized: {regex_tokenizer.tokenize(s)}")


def test_dataset_sample():
    dataset_path = r"D:\NLP_DL\data\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)

    sample_text = raw_text[:500] 

    log_and_print("\nTokenizing Sample Text from UD_English-EWT")
    log_and_print(f"Original Sample: {sample_text[:100]}")

    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    regex_tokens = regex_tokenizer.tokenize(sample_text)

    log_and_print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    log_and_print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")

if __name__ == "__main__":
   test_sentences()
   test_dataset_sample()
