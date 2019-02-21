# Databricks notebook source
# MAGIC %md # Caravan EDA - MA755

# COMMAND ----------

# MAGIC %md ## Introduction

# COMMAND ----------

# MAGIC %md ## Dataset Description

# COMMAND ----------

# MAGIC %md Documentation on the dataset can be found at:
# MAGIC - https://www.kaggle.com/uciml/caravan-insurance-challenge/data

# COMMAND ----------

# MAGIC %md 
# MAGIC Each row of the dataset represents a group of customers. The following set of variables describe this group of customers:
# MAGIC - `MAANTHUI`: Number of houses 1 - 10 (in the group of customers)
# MAGIC - `MGEMOMV`: Avg size household 1 - 6 (in the group of customers)
# MAGIC - `MOSHOOFD`: Customer main type; see L2
# MAGIC - `MOSTYPE`: Customer Subtype; see L0
# MAGIC - `MGEMLEEF`: Avg age; see L1
# MAGIC 
# MAGIC See the documentation for the meaning of L0, L1 and L2. 
# MAGIC 
# MAGIC The remaining variables that start with "M" provide demographic data and contain integers between `0` and `9` that represent ranges of percentages with `0` representing 0% and `9` representing 100%. 
# MAGIC The other integers represent ranges of about 13% each. 
# MAGIC See the documentation for details. 
# MAGIC 
# MAGIC The variables that start with "A" or "P" provide information about the customers in that postal code. 
# MAGIC They contain integers between `0` and `9` that represent ranges of counts for that variable.
# MAGIC See the documentation for details. 
# MAGIC 
# MAGIC The `CARAVAN` variable is the target variable.
# MAGIC It is also a _count_ variable (as above) recording the number of mobile home policies in that postal code.

# COMMAND ----------

# MAGIC %md ## Load Libraries

# COMMAND ----------

# MAGIC %md Load the required libraries and check version numbers. 

# COMMAND ----------

import numpy             as np
import pandas            as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt
import seaborn as sns
np.__version__, pd.__version__, mpl.__version__, sns.__version__

# COMMAND ----------

# MAGIC %md ## Read Dataset

# COMMAND ----------

# MAGIC %md First check that the files exists where we expect it.

# COMMAND ----------

# MAGIC %sh ls /dbfs/mnt/group-ma755/data

# COMMAND ----------

# MAGIC %md Check that the file has a header and looks reasonable. 

# COMMAND ----------

# MAGIC %sh head /dbfs/mnt/group-ma755/data/caravan-insurance-challenge.csv

# COMMAND ----------

caravan_df = pd.read_csv('/dbfs/mnt/group-ma755/data/caravan-insurance-challenge.csv')
caravan_df.head()

# COMMAND ----------

# MAGIC %md ## Check Dataset

# COMMAND ----------

caravan_df.info()

# COMMAND ----------

caravan_df.columns

# COMMAND ----------

# MAGIC %md ## Explore Single Variables

# COMMAND ----------



# COMMAND ----------

# MAGIC %md __[REMOVE FROM REPORT]__ Choose 10 variables to analyze, including the `CARAVAN` variable. For each create a summary and a histogram with pandas. Briefly introduce each code block and describe the output. These summaries are important to check for anomalies and to verify that variables distributions. 

# COMMAND ----------

# MAGIC %md The following cells analyze the 'MGEMLEEF' variable using the .describe() function. This variable gives us the customer average age categories . There are 6 categories in all (1: 20-30 years 2: 30-40 years 3: 40-50 years 4: 50-60 years 5: 60-70 years 6: 70-80 years). The major customer subgroup in the dataset is in 40-50 years pld category. The histogram reinforces this.

# COMMAND ----------

caravan_df[['MGEMLEEF']].describe()

# COMMAND ----------

caravan_df[['MGEMLEEF']].plot(kind='hist',edgecolor='black')
display(plt.show())

# COMMAND ----------

# MAGIC %md The following cells analyze the 'MINKGEM' variable using the .describe() function. This variable gives us the level of the percentage of customers whose yearly income is above average in that postal area. The categories are 0: 0%, 1: 1 - 10%, 2: 11 - 23%, 3: 24 - 36%, 4: 37 - 49%, 5: 50 - 62%, 6: 63 - 75%, 7: 76 - 88%, 8: 89 - 99%, 9: 100%. The histogram shows the majority of customer groups have 24-49% of customers with above average income.

# COMMAND ----------

caravan_df[['MINKGEM']].describe()

# COMMAND ----------

caravan_df[['MINKGEM']].plot(kind='hist',edgecolor='black')
display(plt.show())

# COMMAND ----------

# MAGIC %md The following cells analyze the 'MFALLEEN' variable using the .describe() function. This variable gives us the percentage of customers who are in single status in each postal area. The categories are 0: 0%, 1: 1 - 10%, 2: 11 - 23%, 3: 24 - 36%, 4: 37 - 49%, 5: 50 - 62%, 6: 63 - 75%, 7: 76 - 88%, 8: 89 - 99%, 9: 100%. The histogram shows the largest group we have in dataset has no single customers, and about 75% of the observations have single customers with a percentage of below 36%. 

# COMMAND ----------

caravan_df[['MFALLEEN']].describe()

# COMMAND ----------

caravan_df[['MFALLEEN']].plot(kind='hist',edgecolor='black')
display(plt.show())

# COMMAND ----------

# MAGIC %md The following cells analyze the 'MZFONDS' variable using the .describe() function. This variable gives us the percentage of customers who purchase national health insurance in each postal area. The categories are 0: 0%, 1: 1 - 10%, 2: 11 - 23%, 3: 24 - 36%, 4: 37 - 49%, 5: 50 - 62%, 6: 63 - 75%, 7: 76 - 88%, 8: 89 - 99%, 9: 100%. The histogram shows above 75% of the observations have more than half customers in that postal area purchasing national health insurance. 

# COMMAND ----------

caravan_df[['MZFONDS']].describe()

# COMMAND ----------

caravan_df[['MZFONDS']].plot(kind='hist',edgecolor='black')
display(plt.show())

# COMMAND ----------

# MAGIC %md ## Explore Multiple Variables

# COMMAND ----------

# MAGIC %md __[REMOVE FROM REPORT]__ For _interesting_ pairs or triples of variables use the `groupby` and `pivot_table` functions to analyze multiple variables. Often this will involve a single numeric variable and one or two categorical variables. 

# COMMAND ----------

# MAGIC %md Below we look at the households yearly income below 30,000(MINKM30),from 30,000 to 45,000(MINK3045), from 45,000 to 75,000(MINK4575), from 75,000 to 122,000(MINK7512), above 123,000(MINK123M), grouped by the average households size(MGEMOMV). Averagely, as the household size grows, the percentage of low-income(below 30,000 annually) customers decrease distinctly, while the percentage of middle-income(from 45,000 to 75,000 annually) customers increase steadily. The highest income level(above 123,000) also shows an uptrend as the household size grows, but with relatively small number.

# COMMAND ----------

caravan_df[['MINKM30','MINK3045','MINK4575','MINK7512','MINK123M','MGEMOMV']].groupby('MGEMOMV',sort=True).mean()

# COMMAND ----------

caravan_df[['MINKM30','MINK3045','MINK4575','MINK7512','MINK123M','MGEMOMV']].groupby('MGEMOMV',sort=True).max()

# COMMAND ----------

# MAGIC %md The group below shows the percentage of customer with different religion grouped by household size. There is an interesting point in the largest household size level. On average customers in the largest size of household take a majority in Roman catholic as well as Protestant, but customers in the largest household size take up relatively very small part among other religion and no religion people.

# COMMAND ----------

caravan_df[['MGODRK','MGODPR','MGODOV','MGODGE','MGEMOMV']].groupby('MGEMOMV',sort=True).mean()

# COMMAND ----------



# COMMAND ----------

caravan_df[['MHHUUR','MHKOOP','MGEMOMV']].groupby('MGEMOMV',sort=True).mean()

# COMMAND ----------

# MAGIC %md ### Conclusion