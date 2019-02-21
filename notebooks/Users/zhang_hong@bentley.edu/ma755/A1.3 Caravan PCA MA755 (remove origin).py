# Databricks notebook source
# MAGIC %md # Caravan PCA - MA755

# COMMAND ----------

# MAGIC %md ## Introduction

# COMMAND ----------

# MAGIC %md In this notebook we will use PCA (Principal Component Analysis) to convert a set of observations of possibly correlated variables into a set of uncorrelated values called principal components.  

# COMMAND ----------

# MAGIC %md ## Dataset Description

# COMMAND ----------

# MAGIC %md 
# MAGIC Each row of the dataset represents a group of customers. The variables that start with "M" provide demographic data and contain integers between `0` and `9` that represent ranges of percentages with `0` representing 0% and `9` representing 100%. 
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

# MAGIC %md In the following cell we load the appropriate libraries into the notebook, and check the version numbers.

# COMMAND ----------

import numpy   as np
import pandas  as pd
import sklearn as sk
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import gen_features
import sklearn.preprocessing, sklearn.decomposition, \
       sklearn.linear_model,  sklearn.pipeline, \
       sklearn.metrics
np.__version__, pd.__version__, sk.__version__

# COMMAND ----------

# MAGIC %md ## Read Dataset

# COMMAND ----------

# MAGIC %md The following cell reads in the dataset and assigns it to the `caravan_df` variable.  We then run the `.head()` commande to verify the data looks as expected.

# COMMAND ----------

caravan_df = pd.read_csv('/dbfs/mnt/group-ma755/data/caravan-insurance-challenge.csv')
caravan_df.head()

# COMMAND ----------

# MAGIC %md The target variable `CARAVAN` should not be considered in the process of building model so that our model will not bring bias. Therefore, we remove the `CARAVAN` variable in the feature dataset.

# COMMAND ----------

Y = ['CARAVAN']
caravan_df_features=caravan_df.loc[:, ~caravan_df.columns.isin(Y)]
caravan_df_features

# COMMAND ----------

train=caravan_df_features[caravan_df_features.ORIGIN=="train"]
train

# COMMAND ----------

# MAGIC %md In the next 2 cells we run the `.info` commmand on the `caravan_df_features` variable to make sure the data structure is as expected, and also to find the total number of rows in the dataframe, which we then assign to the `caravan_rows` variable.

# COMMAND ----------

caravan_df_features.info()

# COMMAND ----------

# MAGIC %md In this cell we create the lists we will use in order to binarize the dataframe.  We create the variable `feature_list`, and use this variable to create two lists, `binarizer_list` and `scaler_list`.  The `binarizer_list` will include all the variables we want to binarize (make either a 1 or 0).  The `scalar_list` will include all the other variables in our dataframe. We accomplish this in two steps.  For the `binarizer_list`, we include any columns for which the name is present in `feature_list`.  For the `scalar_list`, we include any columns for which the name is not present in `feature_list`.  We then print out both lists to verify that the code has worked properly.

# COMMAND ----------

feature_list = ['MOSHOOFD','MOSTYPE']
binarizer_list = [[name] for name in list(caravan_df_features.columns) if name     in feature_list]
scaler_list    = [[name] for name in list(caravan_df_features.columns) if name not in feature_list]
binarizer_list, scaler_list

# COMMAND ----------

# MAGIC %md Next we create the pipeline, passing both the `binarizer_list` and the `scalar_list` into the pipeline using `FeatureUnion`, and transform our dataframe into a numpy array.   We then use .fit_transform() command on the dataframe to verify that the mapper behaves as expected.  The `np.round` command simply rounds the values to a number of decimals, given the `decimals` criteria at the end of the command.

# COMMAND ----------

pipe = sklearn.pipeline.FeatureUnion([('binarizer', DataFrameMapper(gen_features(columns=binarizer_list,classes=[{'class': sklearn.preprocessing.LabelBinarizer}]))), 
                                      ('scaler', DataFrameMapper(gen_features(columns=scaler_list,classes=[{'class': sklearn.preprocessing.MinMaxScaler}])))])
Binarized_Caravan = np.round(pipe.fit_transform(caravan_df_features), decimals=2)

# COMMAND ----------

# MAGIC %md ## PCA Analysis

# COMMAND ----------

# MAGIC %md The following cell defines the `MyTransformer` class that we will use the in pipeline.  `MyTransformer` subtracts the column mean from each value in the column.

# COMMAND ----------

from sklearn.base import BaseEstimator, TransformerMixin
class MyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,my_var=False):
    self.my_var = my_var
  def fit(self, X, y=None):
    self.col_mean_ = np.mean(X,axis=0)
    return self
  def transform(self, X):
    ret_df = X - self.col_mean_
    return ret_df

# COMMAND ----------

# MAGIC %md The following cell defines the `PCATransformer` class that we will use the in pipeline.  `PCATransformer` creates, then sorts, the eigenvalues and eigenvectors.  Then creates a covariance matrix using the dot product of the now mean adjusted dataframe and the eigenvectors.

# COMMAND ----------

from sklearn.base import BaseEstimator, TransformerMixin
class PCATransformer(BaseEstimator, TransformerMixin):
  def __init__(self,my_var=False):
    self.my_var = my_var
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    eig_val, eig_vec = np.linalg.eig(X.T.dot(X)/len(X))
    idx = eig_val.argsort()[::-1]
    explained_variance_eigenValues = eig_val[idx]
    explained_variance_eigenVectors = eig_vec[:,idx]
    AQ_cov = np.cov(m=X.dot(explained_variance_eigenVectors),rowvar=False,bias=True)
    final_data=X.dot(explained_variance_eigenVectors)
    return AQ_cov, final_data

# COMMAND ----------

# MAGIC %md Here we now create the pipeline using the pipe we created previously that binarizes the dataframe.  The pipeline then passes this first into the `MyTransformer` class, then into the `PCATransformer` class created in the cells directly above.  We then pass the `caravan_df` dataframe into the pipeline using the `np.round` command in order to round the results to 4 decimal places.  

# COMMAND ----------

from sklearn import pipeline
from sklearn import preprocessing

ml_pipeline=pipeline.Pipeline([('Binarizer', pipe),
                              ('std', MyTransformer()),
                              ('pca', PCATransformer())
                              ])
AQ_cov, final_data =ml_pipeline.fit_transform(caravan_df_features)
B=np.round(AQ_cov.astype('float64'), decimals=4)
B

# COMMAND ----------

# MAGIC %md We put the value of target variable `CARAVAN` as label for our outcome. To visualize our dimensionally reduced features, we make the 86 features appear in the 2 mixed features using the eigen vectors matrix we generated during PCA transform process. The plot shows how observations are distributed in a 2-dimensional way. The red dots represent having purchased a caravan policy and the green dots represent those who have not purchased a policy.

# COMMAND ----------

import matplotlib.pyplot as plt
label1=caravan_df[caravan_df.CARAVAN==1]
label0=caravan_df[caravan_df.CARAVAN==0]
plt.scatter(final_data[label0.index,0], final_data[label0.index,1], alpha=0.5, color='green')
display(plt.show())
plt.scatter(final_data[label1.index,0], final_data[label1.index,1], alpha=0.5, color='red')
display(plt.show())
plt.title('Data transformed with 2 eigenvectors')

# COMMAND ----------

# MAGIC %md ### Conclusion

# COMMAND ----------

# MAGIC %md In this notebook we create readin the `caravan` dataset and assign it to a dataframe labeled `caravan_df`.  We then binarize the data using the `LabelBinarizer` function for categorical variables, and `MinMaxScaler` for numerical variables.  We created a class called `PCATransformer` which calculates the eigenvalues & eigenvectors for the dataset, sorts them, then creates a covariance matrix based on the dataframe and the eigenvectors. It also returns a dot product of dataframe and eigenvectors.  Finally, we create a pipeline (shown below) that passes a dataframe into the binarizer function called `pipe`, which in turn is passed into the class `MyTransformer`, which subtracts the column mean from each value in that column.  Finally, this is passed into `PCATransformer`.   
# MAGIC 
# MAGIC `ml_pipeline=pipeline.Pipeline([('Binarizer', pipe),
# MAGIC                               ('std', MyTransformer()),
# MAGIC                               ('pca', PCATransformer())
# MAGIC                               ])`
# MAGIC                               
# MAGIC As expected, the pipeline returns a diagonal matrix where the only non-zero values reside in the diagonal of the matrix (similar to an identity matrix).  The values decrease as we move from left to right, which is also as expected since the eigenvectors were sorted in a descending order.