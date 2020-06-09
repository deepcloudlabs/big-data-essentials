from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

config = SparkConf().setMaster("local").setAppName("NaiveBayes")
sc = SparkContext(conf=config)
spark = SparkSession.builder.appName("naive-bayes").getOrCreate()

smsData = sc.textFile("data/sms-spam-collection.csv",2)
smsData.cache()
smsData.collect()

def TransformToVector(inputStr):
    attList=inputStr.split(",")
    smsType= 0.0 if attList[0] == "ham" else 1.0
    return [smsType, attList[1]]

smsXformed=smsData.map(TransformToVector)

smsDf= spark.createDataFrame(smsXformed,
                          ["label","message"])
smsDf.cache()
smsDf.select("label","message").show()

(trainingData, testData) = smsDf.randomSplit([0.9, 0.1])
trainingData.count()
testData.count()
testData.collect()

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

tokenizer = Tokenizer(inputCol="message", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="tempfeatures")
idf=IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
nbClassifier=NaiveBayes()

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nbClassifier])

#Build a model with a pipeline
nbModel=pipeline.fit(trainingData)
#Predict on test data (will automatically go through pipeline)
prediction=nbModel.transform(testData)

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy")
evaluator.evaluate(prediction)

#Draw confusion matrics
prediction.groupBy("label","prediction").count().show()
