## Objectives
- To create a mock pipeline based on the business model that students choose

## Introduction

Based on the previous readings, we will simulate how we can best determine
- The state of the current system (legacy)
- Whether we should migrate the system to cloud
- (Optional) Designing the architecture of the cloud system


### Problem Statement
Suppose you're a decision maker in the engineering team at an eCommerce company which currently generates millions of dollars in net revenue. You have a legacy system that sits on company's data centers in remote locations, away from the headquarter where your office is. The engineering team and the data analytics team have made the decision that they would like to build a search engine, recommendation engine, and few other engines that require heavy machine learning models to increase the revenue for the company. As a decision maker, you need to assess the current system, and the engineering requirement for the future fiscal years. So you decide to answer some questions and prepare answers for the next leadership meeting.


Currently, the capability of the data center is limited based on the square footage of the location, as well as the hardware that can be stored in the square footage. The data center resource is not only shared in the engineering team, but also within the supply chain team, logistics, HR's IT platform, and also with some other internal operations, like marketing, leadership, and also analytics team. The data centers have maxed out in terms of the hardware that they can support, and may be in need of expanding and opening up another data center.

#### Assumptions when solving this problem

- The problem statement merely states the current business problem you're facing. The design of the hypothetical data center, like square footage, hardware capabilities, human resources are all something you can create! The limitation is one thing to keep in mind, that in any data center(s), you have ran out of spaces to add more hardware.
- You can assume based on the revenue the company is making how much data is being generated per day.
- Be creative! If you're using third party data, i.e. Adobe Target for audience targeting, how are we receiving that data? Will that data sit on our data center and will it take up the resources? These are free constraints that you can create, but be realistic.

1. What is the current cost of maintaining the remote data center location?


```python
# your solution
```

2. What are the new business requirements from your engineering team? 


```python
# your solution, but be creative! It doesn't have to limit to the requirement from the problem statement
```

3. Based on the new business requirements, you decide that you need to start migrating (at least) your resources to the cloud system. What are some things that you may need to look into when considering migrating your legacy system to cloud?


```python
# your solution
```

4. In the end, you'll need to persuade leadership why you need to migrate a system to another place, i.e. cloud, when in the eyes of leadership, everything is working fine. Write out cons/pros based on the shared categories, like cost, human resources, etc.


```python
# your solution
```

(OPTIONAL) This may require an engineering/software background. You have the same hypothetical machine learning architecture, but one on the legacy system and one on the cloud. Moving away from the business logic, what is the benefit on the engineering stack if we use the cloud? What can be problematic if we use the cloud?


```python
# your solution, attach a screenshot of a hypothetical architecture at a high level
```


```python

```
