from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import udf,split
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator




from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Algortimo de clasificacion multiclase")\
        .master("local[2]")\
        .getOrCreate()

df = spark.read.format('com.databricks.spark.csv')\
    .options(header='true', inferschema='true',delimiter=",")\
    .load("/Users/victor/PycharmProjects/imagesPrediction/resources/fer2013.csv",header=True)

parse_ = udf(lambda a: Vectors.dense(a), VectorUDT())

df2 = df.select(split(df["pixels"]," ").cast("array<int>").alias("pixels"),"emotion").withColumn("pixels", parse_("pixels"))


dfSample = df2.sample(False,0.4,5)

train, test = dfSample.randomSplit([0.7, 0.3])

rf = RandomForestClassifier(labelCol="emotion", featuresCol="pixels", numTrees=20)

model = rf.fit(train)

predictions = model.transform(test)

#Precision del algoritmo
evaluator = MulticlassClassificationEvaluator(
    labelCol="emotion", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

#Guardar algoritmo que mejor resultado de para usarlo en el streaming
model.write().overwrite().save("/Users/victor/PycharmProjects/imagesPrediction/modelsTL/randomForest")

