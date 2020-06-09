from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

config = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf=config)
spark = SparkSession.builder.appName("decision-tree").getOrCreate()

irisData = sc.textFile("data/iris.csv")
irisData.cache()
irisData.count()

dataLines = irisData.filter(lambda x: "Sepal" not in x)
dataLines.count()

from pyspark.sql import Row
parts = dataLines.map(lambda l: l.split(","))
irisMap = parts.map(lambda p: Row(SEPAL_LENGTH=float(p[0]),\
                                SEPAL_WIDTH=float(p[1]), \
                                PETAL_LENGTH=float(p[2]), \
                                PETAL_WIDTH=float(p[3]), \
                                SPECIES=p[4] ))
                                
irisDf = spark.createDataFrame(irisMap)
irisDf.cache()

from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="SPECIES", outputCol="IND_SPECIES")
si_model = stringIndexer.fit(irisDf)
irisNormDf = si_model.transform(irisDf)

irisNormDf.select("SPECIES","IND_SPECIES").distinct().show()
irisNormDf.cache()

irisNormDf.describe().show()

for i in irisNormDf.columns:
    if not( isinstance(irisNormDf.select(i).take(1)[0][0], str)) :
        print( "Correlation to Species for ", i, irisNormDf.stat.corr('IND_SPECIES',i))

from pyspark.ml.linalg import Vectors
def transformToLabeledPoint(row) :
    lp = ( row["SPECIES"], row["IND_SPECIES"], Vectors.dense([row["SEPAL_LENGTH"], row["SEPAL_WIDTH"], row["PETAL_LENGTH"], row["PETAL_WIDTH"]]))
    return lp
    

irisLp = irisNormDf.rdd.map(transformToLabeledPoint)
irisLpDf = spark.createDataFrame(irisLp,["species","label", "features"])
irisLpDf.select("species","label","features").show(10)
irisLpDf.cache()

(trainingData, testData) = irisLpDf.randomSplit([0.9, 0.1])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",  featuresCol="features")
dtModel = dtClassifer.fit(trainingData)

print(dtModel.numNodes)
print(dtModel.depth)
print(dtModel.toDebugString) 

predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

predictions.groupBy("label","prediction").count().show()
