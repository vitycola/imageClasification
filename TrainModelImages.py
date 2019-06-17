import logging

from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit,col,udf
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Algortimo de clasificacion multiclase").getOrCreate()

path = "./resources/"
angry_df = ImageSchema.readImages(path + "0/").withColumn ("label" , lit ( 0 ))
happy_df = ImageSchema.readImages(path + "3/" ).withColumn ( "label" , lit ( 1 ))
sad_df = ImageSchema.readImages(path + "4/" ).withColumn ( "label" , lit ( 2 ))


sc = spark.sparkContext

log4jLogger = sc._jvm.org.apache.log4j
log = log4jLogger.Logger.getLogger(__name__)

log.info("pyspark script logger initialized")

df1 = angry_df.union(happy_df).union(sad_df)

parse_ = udf(lambda a: Vectors.dense(a), VectorUDT())
df = df1.withColumn("features",parse_(df1["image.data"]))

train, test, _ = df.randomSplit([0.1, 0.05, 0.85])

lr = LogisticRegression(maxIter=100, regParam=0.05, elasticNetParam=0.3, featuresCol="features",labelCol="label")
train.cache()

p = Pipeline(stages=[lr])
p_model = p.fit(train)

predictions = p_model.transform(test)

#Precision del algoritmo
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
log.info("Test Error = %g" % (1.0 - accuracy))

#Guardar algoritmo que mejor resultado de para usarlo en el streaming
p_model.stages[0].write().overwrite().save("./models/LogisticRegression")
log.info("Saved model")
log.info("Acurraccy = %g" % (accuracy))