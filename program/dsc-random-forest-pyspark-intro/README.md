# Random Forest with Pyspark Introduction  

## Introduction

In this lesson, you will walk through how to use PySpark for the classification of Iris flowers with a Random Forest Classifier. The dataset is located under the `data` folder.

## Objectives  

* Read a dataset into a PySpark DataFrame
* Implement a random forest classifier with PySpark

> Before continuing, check the version of PySpark installed on the machine. It should be above 3.1.
> 
> You will run this notebook in a `pyspark-env` environment following [these setup instructions without docker](https://github.com/learn-co-curriculum/dsc-spark-docker-installation)


```python
from pyspark.sql import SparkSession  # entry point for pyspark

# instantiate spark instance
spark = (
    SparkSession.builder.appName("Random Forest Iris").master("local[*]").getOrCreate()
)
```

    WARNING: An illegal reflective access operation has occurred
    WARNING: Illegal reflective access by org.apache.spark.unsafe.Platform (file:/Users/pisel/opt/anaconda3/envs/spark-env/lib/python3.8/site-packages/pyspark/jars/spark-unsafe_2.12-3.0.0.jar) to constructor java.nio.DirectByteBuffer(long,int)
    WARNING: Please consider reporting this to the maintainers of org.apache.spark.unsafe.Platform
    WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
    WARNING: All illegal access operations will be denied in a future release
    23/09/07 11:44:58 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).


After version 3.0, `SparkSession` is the main entry point for Spark. `SparkSession.builder` creates a spark session. Any thing can go into the `appName()` to specify which jobs you are running currently. Once the spark session is instantiated, if you are running on your local machine, you can access the Spark UI at `localhost:4040` to view jobs.


```python
df = spark.read.csv("./data/IRIS.csv", header=True, inferSchema=True)
df.printSchema()  # to see the schema
```

                                                                                    

    root
     |-- sepal_length: double (nullable = true)
     |-- sepal_width: double (nullable = true)
     |-- petal_length: double (nullable = true)
     |-- petal_width: double (nullable = true)
     |-- species: string (nullable = true)
    



```python
df.show()  # or df.show(Truncate=false) if you'd like to see all the contents
```

    +------------+-----------+------------+-----------+-----------+
    |sepal_length|sepal_width|petal_length|petal_width|    species|
    +------------+-----------+------------+-----------+-----------+
    |         5.1|        3.5|         1.4|        0.2|Iris-setosa|
    |         4.9|        3.0|         1.4|        0.2|Iris-setosa|
    |         4.7|        3.2|         1.3|        0.2|Iris-setosa|
    |         4.6|        3.1|         1.5|        0.2|Iris-setosa|
    |         5.0|        3.6|         1.4|        0.2|Iris-setosa|
    |         5.4|        3.9|         1.7|        0.4|Iris-setosa|
    |         4.6|        3.4|         1.4|        0.3|Iris-setosa|
    |         5.0|        3.4|         1.5|        0.2|Iris-setosa|
    |         4.4|        2.9|         1.4|        0.2|Iris-setosa|
    |         4.9|        3.1|         1.5|        0.1|Iris-setosa|
    |         5.4|        3.7|         1.5|        0.2|Iris-setosa|
    |         4.8|        3.4|         1.6|        0.2|Iris-setosa|
    |         4.8|        3.0|         1.4|        0.1|Iris-setosa|
    |         4.3|        3.0|         1.1|        0.1|Iris-setosa|
    |         5.8|        4.0|         1.2|        0.2|Iris-setosa|
    |         5.7|        4.4|         1.5|        0.4|Iris-setosa|
    |         5.4|        3.9|         1.3|        0.4|Iris-setosa|
    |         5.1|        3.5|         1.4|        0.3|Iris-setosa|
    |         5.7|        3.8|         1.7|        0.3|Iris-setosa|
    |         5.1|        3.8|         1.5|        0.3|Iris-setosa|
    +------------+-----------+------------+-----------+-----------+
    only showing top 20 rows
    


Check to see what the type is for the DataFrame you have loaded.


```python
type(df)
```




    pyspark.sql.dataframe.DataFrame



Go ahead and run some exploratory data analysis on the dataset. You can easily turn the PySpark DataFrame into a Pandas DataFrame.


```python
import pandas as pd

pandas_df = pd.DataFrame(df.take(100), columns=df.columns)
pandas_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.471000</td>
      <td>3.094000</td>
      <td>2.862000</td>
      <td>0.785000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.641698</td>
      <td>0.476057</td>
      <td>1.448565</td>
      <td>0.566288</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.000000</td>
      <td>2.800000</td>
      <td>1.500000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.400000</td>
      <td>3.050000</td>
      <td>2.450000</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.900000</td>
      <td>3.400000</td>
      <td>4.325000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>4.400000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pandas_df.dtypes
```




    sepal_length    float64
    sepal_width     float64
    petal_length    float64
    petal_width     float64
    species          object
    dtype: object



Once the exploratory data analysis is done, you can start feature transforming to prepare for feataure engineering. Feature transforming means scaling, modifying features to be used for train/test validation, and converting. For this purpose, you will use the `VectorAssembler` in PySpark.


```python
from pyspark.ml.feature import VectorAssembler

numeric_cols = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]  # insert numeric cols
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df = assembler.transform(df)  # just use the same dataframe
df.show()
```

    +------------+-----------+------------+-----------+-----------+-----------------+
    |sepal_length|sepal_width|petal_length|petal_width|    species|         features|
    +------------+-----------+------------+-----------+-----------+-----------------+
    |         5.1|        3.5|         1.4|        0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|
    |         4.9|        3.0|         1.4|        0.2|Iris-setosa|[4.9,3.0,1.4,0.2]|
    |         4.7|        3.2|         1.3|        0.2|Iris-setosa|[4.7,3.2,1.3,0.2]|
    |         4.6|        3.1|         1.5|        0.2|Iris-setosa|[4.6,3.1,1.5,0.2]|
    |         5.0|        3.6|         1.4|        0.2|Iris-setosa|[5.0,3.6,1.4,0.2]|
    |         5.4|        3.9|         1.7|        0.4|Iris-setosa|[5.4,3.9,1.7,0.4]|
    |         4.6|        3.4|         1.4|        0.3|Iris-setosa|[4.6,3.4,1.4,0.3]|
    |         5.0|        3.4|         1.5|        0.2|Iris-setosa|[5.0,3.4,1.5,0.2]|
    |         4.4|        2.9|         1.4|        0.2|Iris-setosa|[4.4,2.9,1.4,0.2]|
    |         4.9|        3.1|         1.5|        0.1|Iris-setosa|[4.9,3.1,1.5,0.1]|
    |         5.4|        3.7|         1.5|        0.2|Iris-setosa|[5.4,3.7,1.5,0.2]|
    |         4.8|        3.4|         1.6|        0.2|Iris-setosa|[4.8,3.4,1.6,0.2]|
    |         4.8|        3.0|         1.4|        0.1|Iris-setosa|[4.8,3.0,1.4,0.1]|
    |         4.3|        3.0|         1.1|        0.1|Iris-setosa|[4.3,3.0,1.1,0.1]|
    |         5.8|        4.0|         1.2|        0.2|Iris-setosa|[5.8,4.0,1.2,0.2]|
    |         5.7|        4.4|         1.5|        0.4|Iris-setosa|[5.7,4.4,1.5,0.4]|
    |         5.4|        3.9|         1.3|        0.4|Iris-setosa|[5.4,3.9,1.3,0.4]|
    |         5.1|        3.5|         1.4|        0.3|Iris-setosa|[5.1,3.5,1.4,0.3]|
    |         5.7|        3.8|         1.7|        0.3|Iris-setosa|[5.7,3.8,1.7,0.3]|
    |         5.1|        3.8|         1.5|        0.3|Iris-setosa|[5.1,3.8,1.5,0.3]|
    +------------+-----------+------------+-----------+-----------+-----------------+
    only showing top 20 rows
    


This should have created another column in your dataframe called `features` as you have denoted in `outputCol`. You can use the `StringIndexer` to encode the string column of species to a label index. By default, the labels are assigned according to the frequencies (for imbalanced dataset). The most frequent species would get an index of 0. For a balanced dataset, whichever string appears first will get 0, then so on.


```python
from pyspark.ml.feature import StringIndexer

labeler = StringIndexer(inputCol="species", outputCol="encoded")
df = labeler.fit(df).transform(df)
df.show()
```

    +------------+-----------+------------+-----------+-----------+-----------------+-------+
    |sepal_length|sepal_width|petal_length|petal_width|    species|         features|encoded|
    +------------+-----------+------------+-----------+-----------+-----------------+-------+
    |         5.1|        3.5|         1.4|        0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|    0.0|
    |         4.9|        3.0|         1.4|        0.2|Iris-setosa|[4.9,3.0,1.4,0.2]|    0.0|
    |         4.7|        3.2|         1.3|        0.2|Iris-setosa|[4.7,3.2,1.3,0.2]|    0.0|
    |         4.6|        3.1|         1.5|        0.2|Iris-setosa|[4.6,3.1,1.5,0.2]|    0.0|
    |         5.0|        3.6|         1.4|        0.2|Iris-setosa|[5.0,3.6,1.4,0.2]|    0.0|
    |         5.4|        3.9|         1.7|        0.4|Iris-setosa|[5.4,3.9,1.7,0.4]|    0.0|
    |         4.6|        3.4|         1.4|        0.3|Iris-setosa|[4.6,3.4,1.4,0.3]|    0.0|
    |         5.0|        3.4|         1.5|        0.2|Iris-setosa|[5.0,3.4,1.5,0.2]|    0.0|
    |         4.4|        2.9|         1.4|        0.2|Iris-setosa|[4.4,2.9,1.4,0.2]|    0.0|
    |         4.9|        3.1|         1.5|        0.1|Iris-setosa|[4.9,3.1,1.5,0.1]|    0.0|
    |         5.4|        3.7|         1.5|        0.2|Iris-setosa|[5.4,3.7,1.5,0.2]|    0.0|
    |         4.8|        3.4|         1.6|        0.2|Iris-setosa|[4.8,3.4,1.6,0.2]|    0.0|
    |         4.8|        3.0|         1.4|        0.1|Iris-setosa|[4.8,3.0,1.4,0.1]|    0.0|
    |         4.3|        3.0|         1.1|        0.1|Iris-setosa|[4.3,3.0,1.1,0.1]|    0.0|
    |         5.8|        4.0|         1.2|        0.2|Iris-setosa|[5.8,4.0,1.2,0.2]|    0.0|
    |         5.7|        4.4|         1.5|        0.4|Iris-setosa|[5.7,4.4,1.5,0.4]|    0.0|
    |         5.4|        3.9|         1.3|        0.4|Iris-setosa|[5.4,3.9,1.3,0.4]|    0.0|
    |         5.1|        3.5|         1.4|        0.3|Iris-setosa|[5.1,3.5,1.4,0.3]|    0.0|
    |         5.7|        3.8|         1.7|        0.3|Iris-setosa|[5.7,3.8,1.7,0.3]|    0.0|
    |         5.1|        3.8|         1.5|        0.3|Iris-setosa|[5.1,3.8,1.5,0.3]|    0.0|
    +------------+-----------+------------+-----------+-----------+-----------------+-------+
    only showing top 20 rows
    


The DataFrame now has a new column named `encoded` with new values populated. You can check the new columns have been added to the PySpark DataFrame by creating a new Pandas DataFrame


```python
pd.DataFrame(df.take(10), columns=df.columns)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
      <th>features</th>
      <th>encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>[4.7, 3.2, 1.3, 0.2]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>[4.6, 3.1, 1.5, 0.2]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>[5.0, 3.6, 1.4, 0.2]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
      <td>[5.4, 3.9, 1.7, 0.4]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
      <td>[4.6, 3.4, 1.4, 0.3]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>[5.0, 3.4, 1.5, 0.2]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
      <td>[4.4, 2.9, 1.4, 0.2]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
      <td>[4.9, 3.1, 1.5, 0.1]</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Now you have transformed the data as needed. To begin building your model, you need to split the data into a train/test dataset.


```python
train, test = df.randomSplit(
    [0.7, 0.3], seed=42
)
print(f"Train dataset count: {str(train.count())}")
print(f"Test dataset count: {str(test.count())}")
```

    Train dataset count: 104
    Test dataset count: 46


Next you will need to instantiate the `RandomForestClassifier` and train the model. At this point before you run the next cell, open up the Spark UI by typing `localhost:4040` into your browser, then navigating to the executors tab.


```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol="features", labelCol="encoded")
model = rf.fit(train)
predictions = model.transform(test)
```

`featuresCol` is the list of features of the dataframe, which means if you have more features you'd like to include, you could put in a list. You create the model by fitting on the training dataset, then validate it by making predictions on the test dataset. `model.transform(test)` will create new columns, like `rawPrediction`, `prediction`, and `probability`.


```python
# if the columns names here are different, do a `printSchema` on top of predictions to see the correct column names
predictions.select(
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "encoded",
    "rawPrediction",
    "prediction",
    "probability",
)
```




    DataFrame[sepal_length: double, sepal_width: double, petal_length: double, petal_width: double, encoded: double, rawPrediction: vector, prediction: double, probability: vector]



You have a trained model, go ahead and evaluate the model by using the `MulticlassClassificationEvaluator`.


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="encoded", predictionCol="prediction"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}%")
test_error = 1.0 - accuracy
print(f"Test Error = {test_error}")
```

    Accuracy: 0.9571428571428571%
    Test Error = 0.04285714285714293


As you can see, the model performs with 97.8% accuracy and has a test error of 0.021. 
