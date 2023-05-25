# Random Forest with Pyspark Introduction  

## Introduction

In this lesson, let's walk through on how to use PySpark for the classification of Iris flowers with Random Forest Classifier. The dataset is located under the `data` folder.

## Objectives  

* Learn how to implement random forest with PySpark 

> Before continuing, check the version of PySpark installed on the machine. It should be above 3.1.


```python
from pyspark.sql import SparkSession #entry point for pyspark

#instantiate spark instance
spark = SparkSession.builder.appName('Random Forest Iris').master("local[*]").getOrCreate()

```

After version 3.0, `SparkSession` is the main entry point for Spark. `SparkSession.builder` creates a spark session. Any thing can go into the `appName()` to specify which jobs you are running currently. Once the spark session is instantiated, if you are running on your local machine, you can access the Spark UI at `localhost:4040` to view jobs.


```python
df = spark.read.csv('./data/IRIS.csv', header=True, inferSchema=True)
df.printSchema() #to see the schema
```


```python
df.show() # or df.show(Truncate=false) if you'd like to see all the contents
```


```python
# do more analysis if necessary on the data, and feel free to use pandas library
# import pandas as pd
# pd.DataFrame(df.take(10), columns=df.columns).transpose()
```

Once the exploratory data analysis is done, we can start feature transforming to prepare for feataure engineering. Feature transforming means scaling, modifying features to be used for train/test validation, converting, etc. For this purpose, we can use `VectorAssembler` in PySpark.`


```python
from pyspark.ml.feature import VectorAssembler


numeric_cols = [] #insert numeric cols
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df = assembler.transform(df) #just use the same dataframe
df.show()
```

This should have created another column in your dataframe called `features` as we have denoted in `outputCol`. Now, we can use the `StringIndexer` to encode the string column of species to a label indicies. By default, the labels are assigned according to the frequencies (for imbalanced dataset). The most frequent species would get an index of 0. For balanced dataset, whichever string appears first will get 0, then so on.


```python
from pyspark.ml.feature import StringIndexer

labeler = StringIndexer(inputCol="features", outputCol="encoded")
df = labeler.fit(df).transform(df)
df.show()
```

You should be able to see the new column named `encoded` with new values populated.


```python
# try doing this if you've already imported pandas
# pd.DataFrame(df.take(10), columns=df.columns).transpose()
```

Now we have transformed the data as we needed, we can now split the data into train/test dataset.


```python
train, test = df.randomSplit([0.7, 0.3], seed=42) #feel free to change the numbers in the random split or seed
print(f"Train dataset count: {str(train.count())}")
print(f"Test dataset count: {str(test.count())}")
```

Let's instantiate the `RandomForestClassifier` and run the model. At this point, feel free to pull up the Spark UI from `localhost:4040` and examine the executors tab.


```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol="features", labelCol="encoded")
model = rf.fit(train)
predictions = model.transform(test)

```


```python
# if the columns names here are different, do a `printSchema` on top of predictions to see the correct column names
predictions.select('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'encoded', 'rawPrediction', 'prediction', 'probability')
```

`featuresCol` is the list of features of the dataframe, which means if you have more features you'd like to include, you could put in a list. We create a model by fitting the training dataset, then predict on using the test dataset. `model.transform(test)` will create new columns, like `rawPrediction`, `prediction`, and `probability`.

Now we've built a model, let's evaluate the model by using `MulticlassClassificationEvaluator`.


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol='encoded', predictionCol='prediction')
accuracy = evaluator.evaluate(prediction)
print(f"Accuracy: {accuracy}%")
test_error = 1.0 - accuracy
print(f"Test Error = {test_error}")
```

Question: How did we perform on the model? What other metrics can we use to present if the classification models performed well?


```python

```
