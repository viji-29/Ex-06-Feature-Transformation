# Ex-06-Feature-Transformation
AIM

To read the given data and perform Feature Transformation process and save the data to a file.

EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM

STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Transformation techniques to all the features of the data set

STEP 4 Save the data to the file CODE:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

df=pd.read_csv("/content/data.csv")

print(df)

df.head()

df.isnull().sum()

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

sm.qqplot(df.HighlyNegativSkew,fit=True,line='45')

sm.qqplot(df.ModeratPositiveSkew,fit=True,line='45')

plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df4=df.copy()

df4['ModerateNegativeSkew_1'],parameters=stats.yeojohnson(df4.ModerateNegativeSkew)

sm.qqplot(df4.ModerateNegativeSkew_1,fit=True,line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df4['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df4[['ModerateNegativeSkew']]))

sm.qqplot(df4['ModerateNegativeSkew_2'],fit=True,line='45')

plt.show()

OUTPUT:

