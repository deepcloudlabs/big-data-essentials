from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Python Spark SQL basic example").getOrCreate()

# spark is an existing SparkSession
df = spark.read.json("file:///c:/tmp/world.json")

# Displays the content of the DataFrame to stdout
df.show()

# spark, df are from the previous example
# Print the schema in a tree format
df.printSchema()

# Select only the "name" column
df.select("name").show()

# Select everybody, but increment the age by 1
df.select(df['name'], df['population'] + 1).show()

# Select highly populated countries
df.filter(df['population'] > 10000000).show()

# Count countries by continent
df.groupBy("continent").count().show()

# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("countries")

sqlDF = spark.sql("SELECT name FROM countries,population where population > 10000000")
sqlDF.show()
