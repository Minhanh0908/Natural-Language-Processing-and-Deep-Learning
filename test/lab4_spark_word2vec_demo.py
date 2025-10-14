import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("SparkWord2VecDemo") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Load Json file 
    df = spark.read.json("data/c4-train.00000-of-01024-30K.json")

    # Get the text column and preprocess
    df = df.select("text").na.drop()
    # Convert to lowercase
    df = df.withColumn("text", lower(col("text")))
    # Remove punctuation and special characters
    df_clean = df.withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", " "))
    # Split the text into an array of words 
    df_clean = df_clean.withColumn("words", split(col("text"), r"\s+"))

    # Configure and train the Word2Vec model
    word2Vec = Word2Vec(
        vectorSize=100,  
        minCount=5,      
        inputCol="words",
        outputCol="result"
    )

    model = word2Vec.fit(df_clean)

    # Find synonyms for the word "computer"
    synonyms = model.findSynonyms("computer", 5)
    print("Synonyms for 'computer':")
    synonyms.show()

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()