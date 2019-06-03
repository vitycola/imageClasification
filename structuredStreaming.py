from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import split
from pyspark.sql.functions import udf,col,from_json
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.types import StructType,StringType,StructField,ArrayType

import os
packages = "org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.1"

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages {0} pyspark-shell".format(packages)
)

spark = SparkSession.builder.appName("StreamingPrediction").getOrCreate()

model = RandomForestClassificationModel.load("/Users/victor/PycharmProjects/imagesPrediction/models/randomForest")

dataStream = spark.readStream.format("kafka")\
    .option("kafka.bootstrap.servers","localhost:9092")\
    .option("subscribe","topic1")\
    .load()

schema = StructType([
    StructField("name", StringType(), True),
    StructField("data", StringType(), True)]
)
df = dataStream.selectExpr("CAST(value AS STRING) as json")\
 .select(from_json(col("json"),schema).alias("message"))\
 .select("message.*")

parse_ = udf(lambda a: Vectors.dense(a), VectorUDT())
df2 = df.select("name",split(df["data"],",").cast("array<int>").alias("pixels"))\
    .withColumn("pixels", parse_("pixels"))

predict = model.transform(df2)\
.select("name","pixels","prediction")

query = predict.writeStream.format("console").start()
#query = assemmbledDF.writeStream.format("csv").outputMode("append")\
#    .option("path", "/Users/victor/PycharmProjects/imagesPrediction/results/")\
#   .option("checkpointLocation", "/Users/victor/PycharmProjects/imagesPrediction/checkpoint")\
#   .start()

query.awaitTermination()



