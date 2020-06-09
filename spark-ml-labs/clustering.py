from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

config = SparkConf().setMaster("local").setAppName("SparkClustering")
sc = SparkContext(conf=config)
spark = SparkSession.builder.appName("clustering").getOrCreate()

autoData = sc.textFile("data/auto-data.csv")
autoData.cache()

firstLine = autoData.first()
dataLines = autoData.filter(lambda x: x != firstLine)
dataLines.count()

from pyspark.sql import Row

import math
from pyspark.ml.linalg import Vectors

def transformToNumeric( inputStr) :
    attList=inputStr.split(",")
    doors = 1.0 if attList[3] =="two" else 2.0
    body = 1.0 if attList[4] == "sedan" else 2.0 
    values= Row(DOORS= doors, \
                     BODY=float(body),  \
                     HP=float(attList[7]),  \
                     RPM=float(attList[8]),  \
                     MPG=float(attList[9])  \
                     )
    return values

autoMap = dataLines.map(transformToNumeric)
autoMap.persist()
autoMap.collect()

autoDf = spark.createDataFrame(autoMap)
autoDf.show()

summStats=autoDf.describe().toPandas()
meanValues=summStats.iloc[1,1:5].values.tolist()
stdValues=summStats.iloc[2,1:5].values.tolist()

bcMeans=sc.broadcast(meanValues)
bcStdDev=sc.broadcast(stdValues)

def centerAndScale(inRow) :
    global bcMeans
    global bcStdDev
    meanArray=bcMeans.value
    stdArray=bcStdDev.value
    retArray=[]
    for i in range(len(meanArray)):
        retArray.append( (float(inRow[i]) - float(meanArray[i])) / float(stdArray[i]) )
    return Vectors.dense(retArray)
    
csAuto = autoDf.rdd.map(centerAndScale)
csAuto.collect()

autoRows=csAuto.map( lambda f:Row(features=f))
autoDf = spark.createDataFrame(autoRows)

autoDf.select("features").show(10)

from pyspark.ml.clustering import KMeans
kmeans = KMeans(k=5, seed=1)
model = kmeans.fit(autoDf)
predictions = model.transform(autoDf)
predictions.show()

import pandas as pd

def unstripData(instr) :
    return ( instr["prediction"], instr["features"][0], instr["features"][1],instr["features"][2],instr["features"][3])
    
unstripped=predictions.rdd.map(unstripData)
predList=unstripped.collect()
predPd = pd.DataFrame(predList)

import matplotlib.pylab as plt
plt.cla()
plt.scatter(predPd[1],predPd[2], c=predPd[0])
plt.show()
