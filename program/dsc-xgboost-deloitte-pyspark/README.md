## Introduction  
Now that we've run the model locally with one month of data, we'd like to build the model using multiple months. The total data *zipped* is about ~10GB, but unzipped it will be much more. We can serialize the data to a Pandas dataframe but most likely it will throw memory issues depending on the machine you have. We want to write code for one month, locally, using PySpark then migrate the code to run on EMR, and take multiple unzipped files.

## Objectives  
* Migrate the model using PySpark to fully utilize distributed computing resource

First, use the boto3 client to set up the s3 resource then check if the file exists in your bucket. If it doesn't exist, you might have to upload it. You can skip this step for now, but will be helpful for the next lab, where you'll be pulling the data from the S3 bucket.


```python
import boto3
from pyspark import SparkSession
#solution

```


```python
spark = SparkSession \
    .builder \
    .appName("XGBoost") \
    .getOrCreate()
```


```python
# path could be local or boto3
path = ""
df = spark.read.csv(path=path, header="true", inferSchema="true")
```


```python
display(df)
```


```python
#df.cache()
```

### How many unique customers?


```python
# solution
```

### Preprocess the data

Using the logic from the previous lab, use pyspark functions to explore the dataset.


```python

```

# Modeling: Cart Abandonment

The model will be similar - let's build out the new features then start building the model.


```python
# solution for additional columns/features
```


```python
# df should be the dataframe with additional columns/features
train, test = df.randomSplit([0.7, 0.3], seed = 42)
print("There are %d training examples and %d test examples." % (train.count(), test.count()))
```

Most MLlib algorithms require a single input column containing a vector of features and a single target column. The DataFrame currently has one column for each feature. MLlib provides functions to help you prepare the dataset in the required format.

MLlib pipelines combine multiple steps into a single workflow, making it easier to iterate as you develop the model.

In this example, you create a pipeline using the following functions:

- VectorAssembler: Assembles the feature columns into a feature vector.
- VectorIndexer: Identifies columns that should be treated as categorical. This is done heuristically, identifying any column with a small number of distinct values as categorical. In this example, the cart abandonment feature would be categorical (0 or 1)
- XgboostRegressor: Uses the XgboostRegressor estimator to learn how to predict rental counts from the feature vectors.
- CrossValidator: The XGBoost regression algorithm has several hyperparameters. This notebook illustrates how to use hyperparameter tuning in Spark. This capability automatically tests a grid of hyperparameters and chooses the best resulting model.


```python
from pyspark.ml.feature import VectorAssembler, VectorIndexer
 
# Remove the target column from the input feature set.
featuresCols = df.columns
# featuresCols.remove('your target column')
 
# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
 
# vectorIndexer identifies categorical features and indexes them, and creates a new column "features". 
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)
```


```python
from sparkdl.xgboost import XgboostRegressor
 
xgb_regressor = XgboostRegressor(num_workers=3, labelCol="your_label_column", missing=0.0)
```


```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
 
# Define a grid of hyperparameters to test:
#  - maxDepth: maximum depth of each decision tree 
#  - maxIter: iterations, or the total number of trees 
paramGrid = ParamGridBuilder()\
  .addGrid(xgb_regressor.max_depth, [2, 5])\
  .addGrid(xgb_regressor.n_estimators, [10, 100])\
  .build()
 
# Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol=xgb_regressor.getLabelCol(),
                                predictionCol=xgb_regressor.getPredictionCol())
 
# Declare the CrossValidator, which performs the model tuning.
cv = CrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid)
```

### Create the pipeline


```python
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])
```


```python

```
