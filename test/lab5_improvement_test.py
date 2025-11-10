from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml.classification import GBTClassifier, MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Sentiment Analysis Improvement") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 2. Load Dataset
data_path = r"D:\NLP_DL\data\sentiments.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
df = df.dropna(subset=["text", "sentiment"])

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 3. Preprocessing
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

# 4. Evaluate model
def evaluate_model(model_name, pipeline):
    model = pipeline.fit(train_data)
    preds = model.transform(test_data)

    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1", labelCol="label")

    acc = evaluator_acc.evaluate(preds)
    f1 = evaluator_f1.evaluate(preds)

    print(f"\n{model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return acc, f1

# 5. TF-IDF Representation
hashingTF = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# 6. Word2Vec Representation
word2vec_train = Word2Vec(vectorSize=50, minCount=3, inputCol="filtered", outputCol="features")

num_classes = int(df.select("label").distinct().count())

# TF-IDF Models 
pipeline_tfidf_lr = Pipeline(stages=[tokenizer, remover, hashingTF, idf, LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")])
pipeline_tfidf_gbt = Pipeline(stages=[tokenizer, remover, hashingTF, idf, GBTClassifier(labelCol="label", featuresCol="features", maxIter=20)])
pipeline_tfidf_mlp = Pipeline(stages=[tokenizer, remover, hashingTF, idf, MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", maxIter=150, layers=[10000, 64, 32, num_classes], seed=42)])

# Word2Vec Models
pipeline_w2v_lr = Pipeline(stages=[tokenizer, remover, word2vec_train, LogisticRegression(maxIter=100, regParam=0.001, featuresCol="features", labelCol="label")])
pipeline_w2v_gbt = Pipeline(stages=[tokenizer, remover, word2vec_train, GBTClassifier(labelCol="label", featuresCol="features", maxIter=20)])
pipeline_w2v_mlp = Pipeline(stages=[tokenizer, remover, word2vec_train, MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", maxIter=150, layers=[50, 64, 32, num_classes], seed=42)])

# 7. Evaluate all models
results = {}
for name, pipeline in [
    ("TF-IDF + Logistic Regression", pipeline_tfidf_lr),
    ("TF-IDF + GBT", pipeline_tfidf_gbt),
    ("TF-IDF + MLP", pipeline_tfidf_mlp),
    ("Word2Vec + Logistic Regression", pipeline_w2v_lr),
    ("Word2Vec + GBT", pipeline_w2v_gbt),
    ("Word2Vec + MLP", pipeline_w2v_mlp),
]:
    results[name] = evaluate_model(name, pipeline)

# 8. Final Results
print("\nFinal Sentiment Analysis Results")
print(f"{'Model':<35}{'Accuracy':<15}{'F1 Score'}")
print("-" * 60)
for name, (acc, f1) in results.items():
    print(f"{name:<35}{acc:<15.4f}{f1:.4f}")

spark.stop()
