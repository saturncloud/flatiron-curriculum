# Random Forest Classifier in PySpark - Lab

## Introduction  

In this lab, you will build a Random Forest Classifier model to study the ecommerce behavior of consumers from a multi-category store. First, you will need to download the data to your local machine, then you will load the data from the local machine onto a Pandas Dataframe.

## Objectives  

* Use the kaggle eCommerce dataset in PySpark
* Build and train a random forest classifier in PySpark

## Instruction
* Accept the Kaggle policy and download the data from [Kaggle](https://www.kaggle.com/code/tshephisho/ecommerce-behaviour-using-xgboost/data)
* For the first model you will only use the 2019-Nov csv data (which is still around ~2gb zipped)
* You will run this notebook in a new `pyspark-env` environment following [these setup instructions without docker](https://github.com/learn-co-curriculum/dsc-spark-docker-installation)


```python
!pip install pandas
```


```python
# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime
```


```python
from pyspark.sql import SparkSession  # entry point for pyspark

# instantiate spark instance
spark = (
    SparkSession.builder.appName("Random Forest eCommerce")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .master("local[*]")
    .getOrCreate()
)
```


```python
path = "../archive/2019-Nov.csv"  # wherever path you saved the kaggle file to
df = spark.read.csv(path, header=True, inferSchema=True)
df.printSchema()  # to see the schema
```

If you want to use Pandas to explore the dataset instead of Pyspark, you have to use the `action` functions, which then means there will be a network shuffle. For smaller dataset such as the Iris dataset which is about ~1KB this is no problem. The current dataset may be too large, and may throw an `OutOfMemory` error if you attempt to load the data into a Pandas dataframe. You should only take a few rows for exploratory analysis if you are more comfortable with Pandas. Otherwise, stick with native PySpark functions. 


```python
pd.DataFrame(df.take(10), columns=df.columns).transpose()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>event_time</th>
      <td>2019-11-01 00:00:00 UTC</td>
      <td>2019-11-01 00:00:00 UTC</td>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>2019-11-01 00:00:01 UTC</td>
      <td>2019-11-01 00:00:02 UTC</td>
      <td>2019-11-01 00:00:02 UTC</td>
      <td>2019-11-01 00:00:02 UTC</td>
    </tr>
    <tr>
      <th>event_type</th>
      <td>view</td>
      <td>view</td>
      <td>view</td>
      <td>view</td>
      <td>view</td>
      <td>view</td>
      <td>view</td>
      <td>view</td>
      <td>view</td>
      <td>view</td>
    </tr>
    <tr>
      <th>product_id</th>
      <td>1003461</td>
      <td>5000088</td>
      <td>17302664</td>
      <td>3601530</td>
      <td>1004775</td>
      <td>1306894</td>
      <td>1306421</td>
      <td>15900065</td>
      <td>12708937</td>
      <td>1004258</td>
    </tr>
    <tr>
      <th>category_id</th>
      <td>2053013555631882655</td>
      <td>2053013566100866035</td>
      <td>2053013553853497655</td>
      <td>2053013563810775923</td>
      <td>2053013555631882655</td>
      <td>2053013558920217191</td>
      <td>2053013558920217191</td>
      <td>2053013558190408249</td>
      <td>2053013553559896355</td>
      <td>2053013555631882655</td>
    </tr>
    <tr>
      <th>category_code</th>
      <td>electronics.smartphone</td>
      <td>appliances.sewing_machine</td>
      <td>None</td>
      <td>appliances.kitchen.washer</td>
      <td>electronics.smartphone</td>
      <td>computers.notebook</td>
      <td>computers.notebook</td>
      <td>None</td>
      <td>None</td>
      <td>electronics.smartphone</td>
    </tr>
    <tr>
      <th>brand</th>
      <td>xiaomi</td>
      <td>janome</td>
      <td>creed</td>
      <td>lg</td>
      <td>xiaomi</td>
      <td>hp</td>
      <td>hp</td>
      <td>rondell</td>
      <td>michelin</td>
      <td>apple</td>
    </tr>
    <tr>
      <th>price</th>
      <td>489.07</td>
      <td>293.65</td>
      <td>28.31</td>
      <td>712.87</td>
      <td>183.27</td>
      <td>360.09</td>
      <td>514.56</td>
      <td>30.86</td>
      <td>72.72</td>
      <td>732.07</td>
    </tr>
    <tr>
      <th>user_id</th>
      <td>520088904</td>
      <td>530496790</td>
      <td>561587266</td>
      <td>518085591</td>
      <td>558856683</td>
      <td>520772685</td>
      <td>514028527</td>
      <td>518574284</td>
      <td>532364121</td>
      <td>532647354</td>
    </tr>
    <tr>
      <th>user_session</th>
      <td>4d3b30da-a5e4-49df-b1a8-ba5943f1dd33</td>
      <td>8e5f4f83-366c-4f70-860e-ca7417414283</td>
      <td>755422e7-9040-477b-9bd2-6a6e8fd97387</td>
      <td>3bfb58cd-7892-48cc-8020-2f17e6de6e7f</td>
      <td>313628f1-68b8-460d-84f6-cec7a8796ef2</td>
      <td>816a59f3-f5ae-4ccd-9b23-82aa8c23d33c</td>
      <td>df8184cc-3694-4549-8c8c-6b5171877376</td>
      <td>5e6ef132-4d7c-4730-8c7f-85aa4082588f</td>
      <td>0a899268-31eb-46de-898d-09b2da950b24</td>
      <td>d2d3d2c6-631d-489e-9fb5-06f340b85be0</td>
    </tr>
  </tbody>
</table>
</div>



### Know your Customers

How many unique customers visit the site?


```python
# using native pyspark
from pyspark.sql.functions import countDistinct

df.select(countDistinct("user_id")).show()
```

                                                                                    

    +-----------------------+
    |count(DISTINCT user_id)|
    +-----------------------+
    |                3696117|
    +-----------------------+
    


Did you notice the spark progress bar when you triggered the `action` function? The `show()` function is the `action` function which means the lazy evaluation of Spark was triggered and completed a certain job. `read.csv` should have been another job. If you go to `localhost:4040` you should be able to see 2 completed jobs under the `Jobs` tab, which are `csv` and `showString`. While a heavy job is getting executed, you can take a look at the `Executors` tab to examine the executors completing the tasks in parellel. Now, you may not see if we run this on a local machine, but this behavior should definitely be visible if you're on a cloud system, such as EMR.

### (Optional) Visitors Daily Trend

Does traffic flunctuate by date? Try using the event_time to see traffic, and draw the plots for visualization.


```python
# for event_time you should use a window and groupby a time period
from pyspark.sql.functions import window
```

Question: You would still like to see the cart abandonment rate using the dataset. What relevant features can we use for modeling?


```python
# your answer
```

Now, you will start building the model. Add the columns you would like to use for predictor features in the model to the `feature_cols` list


```python
from pyspark.ml.feature import VectorAssembler

feature_cols = []  # columns you'd like to use
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)
df.show()
```

To use a string column, you can use the `StringIndexer` to encode the column. Update the `inputCol` keyword argument so that you can encode the target feature.


```python
from pyspark.ml.feature import StringIndexer

labeler = StringIndexer(
    inputCol="", outputCol="encoded"
)  # what should we use for the inputCol here?
df = labeler.fit(df).transform(df)
df.show()
```

Now build the train/test dataset with a 70/30 `randomSplit` and a random seed set to 42


```python
train, test = df.randomSplit()
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))
```

Next you need to add in the name of the feature column and the name of the `labelCol` you previously encoded for training the model.


```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol="", labelCol="")
model = rf.fit(train)
predictions = model.transform(test)
# what goes in the select() function?
predictions.select().show(25)
```

Once the job execution is done, evaluate the model's performance. Add in the `labelCol` below.


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))
```

### Extra: Use the confusion matrix to see the other metrics


```python
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

preds_and_labels = (
    predictions.select(["prediction", "encoded"])
    .withColumn("encoded", F.col("encoded").cast(FloatType()))
    .orderBy("prediction")
)
preds_and_labels = preds_and_labels.select(["prediction", "encoded"])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())
```


```python

```
