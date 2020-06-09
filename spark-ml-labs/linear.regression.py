from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

config = SparkConf().setMaster("local").setAppName("SparkLinearRegression")
sc = SparkContext(conf=config)
spark = SparkSession.builder.appName("linear-regression").getOrCreate()

autoData = sc.textFile("data/auto-miles-per-gallon.csv")
autoData.cache()
autoData.take(5)
dataLines = autoData.filter(lambda x: "CYLINDERS" not in x)
dataLines.count()

from pyspark.sql import Row
avgHP = sc.broadcast(80.0)

def CleanupData( inputStr) :
     global avgHP
     attList=inputStr.split(",")
     hpValue = attList[3]
     if hpValue == "?":
         hpValue=avgHP.value
     values= Row(     MPG=float(attList[0]),\
                      CYLINDERS=float(attList[1]), \
                      DISPLACEMENT=float(attList[2]), 
                      HORSEPOWER=float(hpValue),\
                      WEIGHT=float(attList[4]), \
                      ACCELERATION=float(attList[5]), \
                      MODELYEAR=float(attList[6]),\
                      NAME=attList[7]  )   
     return values

 
autoMap = dataLines.map(CleanupData)
autoMap.cache()

autoDF = spark.createDataFrame(autoMap)

for i in autoDF.columns:
     if not( isinstance(autoDF.select(i).take(1)[0][0],str)):
             print("Correlation to MPG for",i,autoDF.stat.corr("MPG",i))


from pyspark.ml.linalg import Vectors
def transformToLabeledPoint(row) :
    lp = ( row["MPG"], Vectors.dense([row["ACCELERATION"],\
                        row["DISPLACEMENT"], \
                        row["WEIGHT"]]))
    return lp

    
autoLp = autoMap.map(transformToLabeledPoint)
autoDF = spark.createDataFrame(autoLp,["label", "features"])
autoDF.select("label","features").show(10)

(trainingData, testData) = autoDF.randomSplit([0.9, 0.1])
trainingData.count()
testData.count()

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=200)
lrModel = lr.fit(trainingData)

print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

predictions = lrModel.transform(testData)
predictions.select("prediction","label","features").show()

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
evaluator.evaluate(predictions)
