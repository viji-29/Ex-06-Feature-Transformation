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

![image](https://user-images.githubusercontent.com/95408674/197521484-2f942e07-1bdb-4091-8980-160562ab90e9.png)

![image](https://user-images.githubusercontent.com/95408674/197521590-eb255be9-d084-4622-81e0-e10376ded7f2.png)

![image](https://user-images.githubusercontent.com/95408674/197521656-3760b1a6-eb55-4b94-88e7-82e869b69682.png)

![image](https://user-images.githubusercontent.com/95408674/197521737-834c40b3-4eeb-40eb-ae7a-de7671e7f069.png)

![image](https://user-images.githubusercontent.com/95408674/197521864-b883ecb3-572b-4038-8037-58f763d025d7.png)

![image](https://user-images.githubusercontent.com/95408674/197521978-97f85caf-6f47-4166-8958-96d1f40b216a.png)

![image](https://user-images.githubusercontent.com/95408674/197522051-99e9a3f5-97a6-49b5-b67f-a2401bc3bc3b.png)

![image](https://user-images.githubusercontent.com/95408674/197522670-fc658329-720a-4627-ba6f-0a0f720c6139.png)

![image](https://user-images.githubusercontent.com/95408674/197522151-efc24b00-acde-40b7-a434-4657a7972ac9.png)

![image](https://user-images.githubusercontent.com/95408674/197522211-3d90866b-adfd-44a7-b340-88710b0dd2e5.png)

![image](https://user-images.githubusercontent.com/95408674/197522437-539e570c-5faa-4a87-9ad7-62b01762aab1.png)

![image](https://user-images.githubusercontent.com/95408674/197522506-06f0d5bf-3a0c-436e-844b-8a46f081616b.png)

![image](https://user-images.githubusercontent.com/95408674/197522569-3a0a1d4d-7a7c-4624-be89-e5a37c7cd528.png)

![image](https://user-images.githubusercontent.com/95408674/197522818-236ac430-5f5b-42bb-b747-4cf3dd2c51fe.png)

