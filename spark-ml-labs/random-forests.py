from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

config = SparkConf().setMaster("local").setAppName("RandomForests")
sc = SparkContext(conf=config)
spark = SparkSession.builder.appName("randomforests").getOrCreate()

#Load the CSV file into a RDD
bankData = sc.textFile("data/bank.csv")
bankData.cache()
bankData.count()

#Remove the first line (contains headers)
firstLine=bankData.first()
dataLines = bankData.filter(lambda x: x != firstLine)
dataLines.count()

# Change labels to numeric ones and build a Row object

import math
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

def transformToNumeric( inputStr) :
    
    attList=inputStr.replace("\"","").split(";")
    
    age=float(attList[0])
    #convert outcome to float    
    outcome = 0.0 if attList[16] == "no" else 1.0
    
    #create indicator variables for single/married    
    single= 1.0 if attList[2] == "single" else 0.0
    married = 1.0 if attList[2] == "married" else 0.0
    divorced = 1.0 if attList[2] == "divorced" else 0.0
    
    #create indicator variables for education
    primary = 1.0 if attList[3] == "primary" else 0.0
    secondary = 1.0 if attList[3] == "secondary" else 0.0
    tertiary = 1.0 if attList[3] == "tertiary" else 0.0
    
    #convert default to float
    default= 0.0 if attList[4] == "no" else 1.0
    #convert balance amount to float
    balance=float(attList[5])
    #convert loan to float
    loan= 0.0 if attList[7] == "no" else 1.0
    
    #Create a row with cleaned up and converted data
    values= Row(     OUTCOME=outcome ,\
                    AGE=age, \
                    SINGLE=single, \
                    MARRIED=married, \
                    DIVORCED=divorced, \
                    PRIMARY=primary, \
                    SECONDARY=secondary, \
                    TERTIARY=tertiary, \
                    DEFAULT=default, \
                    BALANCE=balance, \
                    LOAN=loan                    
                    ) 
    return values
    
#Change to a Vector
bankRows = dataLines.map(transformToNumeric)
bankRows.collect()[:15]

bankData = spark.createDataFrame(bankRows)

#See descriptive analytics.
bankData.describe().show()

#Find correlation between predictors and target
for i in bankData.columns:
    if not( isinstance(bankData.select(i).take(1)[0][0], str)) :
        print( "Correlation to OUTCOME for ", i, bankData.stat.corr('OUTCOME',i))

#Transform to a Data Frame for input to Machine Learing
def transformToLabeledPoint(row) :
    lp = ( row["OUTCOME"], \
            Vectors.dense([
                row["AGE"], \
                row["BALANCE"], \
                row["DEFAULT"], \
                row["DIVORCED"], \
                row["LOAN"], \
                row["MARRIED"], \
                row["PRIMARY"], \
                row["SECONDARY"], \
                row["SINGLE"], \
                row["TERTIARY"]
        ]))
    return lp
    
bankLp = bankData.rdd.map(transformToLabeledPoint)
bankLp.collect()
bankDF = spark.createDataFrame(bankLp,["label", "features"])
bankDF.select("label","features").show(10)

#Perform PCA
from pyspark.ml.feature import PCA
bankPCA = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
pcaModel = bankPCA.fit(bankDF)
pcaResult = pcaModel.transform(bankDF).select("label","pcaFeatures")
pcaResult.show(truncate=False)

#Indexing needed as pre-req for Decision Trees
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(pcaResult)
td = si_model.transform(pcaResult)
td.collect()

#Split into training and testing data
(trainingData, testData) = td.randomSplit([0.7, 0.3])
trainingData.count()
testData.count()
testData.collect()

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
rmClassifer = RandomForestClassifier(labelCol="indexed", featuresCol="pcaFeatures")
rmModel = rmClassifer.fit(trainingData)

#Predict on the test data
predictions = rmModel.transform(testData)
predictions.select("prediction","indexed","label","pcaFeatures").show()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="indexed",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("indexed","prediction").count().show()
