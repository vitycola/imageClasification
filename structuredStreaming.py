from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import split
from pyspark.sql.functions import udf
from pyspark.ml.classification import  RandomForestClassificationModel

import os
packages = "org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.1"

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages {0} pyspark-shell".format(packages)
)

spark=SparkSession.builder.appName("Pyspark").getOrCreate()



model = RandomForestClassificationModel.load("/Users/victor/PycharmProjects/imagesPrediction/models/randomForest")

dataStream = spark.readStream.format("kafka")\
    .option("kafka.bootstrap.servers","localhost:9092")\
    .option("subscribe","topic1")\
    .load()

decodificar = udf(lambda a: ','.join(str(e) for e in list(a)))
parse_ = udf(lambda a: Vectors.dense(a), VectorUDT())

df = dataStream.select(decodificar(dataStream["value"]).alias("values"))

df2 = df.select(split(df["values"],",").cast("array<int>").alias("pixels")).withColumn("pixels", parse_("pixels"))

pixelColumns = df2.columns


predict = model.transform(df2)\
.select("pixels","prediction","rawPrediction","probability")

query = predict.writeStream.format("console").start()
#query = assemmbledDF.writeStream.format("csv").outputMode("append")\
#    .option("path", "/Users/victor/PycharmProjects/imagesPrediction/results/")\
#   .option("checkpointLocation", "/Users/victor/PycharmProjects/imagesPrediction/checkpoint")\
#   .start()

query.awaitTermination()



