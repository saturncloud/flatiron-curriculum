# Exploratory Data Analysis on Housing Dataset

## Congratulations!!

<img src="https://curriculum-content.s3.amazonaws.com/data-science/images/awesome.gif" alt='image of a man motioning in celebration of your progress'>

## Introduction

In this lesson you will use all of the information you have learned and do a short project. You will take a look at the provided dataset and gather some insight that you will share in a short presentation. You have been provided with some starter code, but please explore different features and look for more meaningful insight.

The ultimate purpose of exploratory analysis is not just to learn about the data, but to help an organization perform better. Explicitly relate your findings to business needs by recommending actions that you think an investor should take if they were shopping for property in this area.


```python
import pandas as pd

# Load the dataset
with open('housing_data.csv', 'w')as f:
    data = f.read()

df = # add the code here to load 'data' into a pandas dataframe

# Display the first few rows of the dataset


```

## Data Cleaning
Before you dive into the analysis, it's essential to ensure the data is clean and ready for exploration. Take some time to look inspect your data and handle missing values and anomalies. Use the following code cell for any data cleaning processes.


```python
# Data cleaning
# Handle missing values and anomalies

```

## Descriptive Statistics
Now, let's calculate some descriptive statistics to understand the distribution of numerical columns in the dataset. Use the following code cell to get some basic information about your data.


```python
# Descriptive statistics
# Calculate mean, median, standard deviation, percentiles, etc.

```

## Categorical Insights
Exploring categorical features can provide valuable insights into property distribution across different categories. Start by analyzing the distribution of property conditions.


```python
# Categorical insights
# Count occurrences of different categories.


```

## Visualization: Property Price Distribution Histogram
Visualizing the distribution of property prices can help us understand the price range that is most common in the market.


```python
import matplotlib.pyplot as plt

# Visualization: Property Price Distribution Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=20, color='blue', alpha=0.7)
plt.title('Property Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

```

## Visualization: Property Size vs. Price Scatter Plot
Let's create a scatter plot to explore the relationship between property size and price.


```python
# Visualization: Property Size vs. Price Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['sqft_living'], df['price'], alpha=0.5)
plt.title('Property Size vs. Price')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.show()

```

## Visualization: Geographic Distribution using Latitude and Longitude
This scatter plot will visualize the geographic distribution of properties using latitude and longitude, with color indicating property prices.


```python
# Visualization: Geographic Distribution using Latitude and Longitude
plt.figure(figsize=(12, 8))
plt.scatter(df['long'], df['lat'], alpha=0.5, c=df['price'], cmap='coolwarm')
plt.title('Geographic Distribution of Properties')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Price')
plt.show()

```

## Visualization: Property Condition Bar Chart
A bar chart can provide insights into the distribution of properties across different conditions.


```python
# Visualization: Property Condition Bar Chart
plt.figure(figsize=(10, 6))
condition_counts.plot(kind='bar', color='green', alpha=0.7)
plt.title('Property Condition Distribution')
plt.xlabel('Condition')
plt.ylabel('Number of Properties')
plt.xticks(rotation=0)
plt.show()

```

Visualization: Number of Bedrooms and Bathrooms Correlation
Finally, let's explore the correlation between the number of bedrooms and bathrooms in properties using a scatter plot.


```python
# Visualization: Number of Bedrooms and Bathrooms Correlation
plt.figure(figsize=(10, 6))
plt.scatter(df['bedrooms'], df['bathrooms'], alpha=0.5, color='purple')
plt.title('Bedrooms vs. Bathrooms Correlation')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Number of Bathrooms')
plt.show()

```

## Presentation Prompt
Now that we have completed the analysis and visualizations, your task is to prepare a short presentation describing the insights obtained from the analysis. Focus on the key findings, trends, and patterns you observed. Your presentation should be concise and informative, highlighting the investor relevant insights drawn from the data.
