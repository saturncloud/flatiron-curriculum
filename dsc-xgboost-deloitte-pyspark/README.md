# Migrate XGBoost Model to PySpark


### Objectives

- Migrate the model using PySpark to fully utilize distributed computing resource


Now we've developed an XGBoost model to study user behavior for the eCommerce, we need to think about how we can run heavy sets of data. At the moment, the model was running on a local computer, using a CPU resource. Think about when the data grows, from terabytes to petabytes because your business grew. How can we fully utilize the distributing computing resources that we have allocated?

One simple solution is to use the distributed computing algorithms, like MapReduce. Here, we would like to use PySpark because it will use the memory resources to compute, and will be much faster.
