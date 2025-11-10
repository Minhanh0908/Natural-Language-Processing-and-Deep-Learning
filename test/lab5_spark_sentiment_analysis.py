from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session 
spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

# 2. Load data 
data_path = r"D:\NLP_DL\data\sentiments.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
# Convert -1/1 labels to 0/1: Normalize sentiment labels
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
# Drop rows with null sentiment values before processingl
initial_row_count = df.count()
df = df.dropna(subset=["text", "sentiment"])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 3. Preprocessing Pipeline 
tokenizer = Tokenizer(inputCol="text", outputCol="words")

stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

idf = IDF(inputCol="raw_features", outputCol="features")

# 4. Train the model 
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[
    tokenizer,
    stopwordsRemover,
    hashingTF,
    idf,
    lr
])

# Train model 
model = pipeline.fit(train_df)

# 5. Evaluate the model 
predictions = model.transform(test_df)

evaluator_acc = MulticlassClassificationEvaluator(
    metricName="accuracy",
    labelCol="label",
    predictionCol="prediction"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    metricName="f1",
    labelCol="label",
    predictionCol="prediction"
)

accuracy = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print("Model Evaluation")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

spark.stop()
