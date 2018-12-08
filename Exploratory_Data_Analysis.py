import pandas as pd
import numpy as np
path='https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'

df = pd.read_csv(path)
df.head()

arr=df['horsepower']
arr1=np.array(arr)
arr1

import matplotlib.pyplot as plt
import seaborn as sns

# list the data types for each column
df.dtypes

df.corr()

df[['bore','stroke' ,'compression-ratio','horsepower']].corr()

'''Positive linear relationship'''
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

df[["engine-size", "price"]].corr()

'''Negative linear relationship'''
sns.regplot(x="highway-mpg", y="price", data=df)

'''Weak Linear Relationship'''
sns.regplot(x="peak-rpm", y="price", data=df)

df[['peak-rpm','price']].corr()

#correlation between x="stroke", y="price"
sns.regplot(x="stroke", y="price", data=df)
df[['stroke','price']].corr()

'''Categorical variables'''

#relationship between "body-style" and "price"
sns.boxplot(x="body-style", y="price", data=df)

#examine engine "engine-location" and "price"
sns.boxplot(x="engine-location", y="price", data=df)

# drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)

df.describe()

df['drive-wheels'].value_counts()

df['drive-wheels'].value_counts().to_frame()

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

'''Basic of Grouping'''
df['drive-wheels'].unique()

df_group_one=df[['drive-wheels','body-style','price']]

# grouping results
df_group_one=df_group_one.groupby(['drive-wheels'],as_index= False).mean()
df_group_one

# grouping results
df_gptest=df[['drive-wheels','body-style','price']]
grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
grouped_test1

grouped_pivot=grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

grouped_pivot=grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot

#"groupby" function to find the average "price" of each car based on "body-style"
grouped_test2=df_gptest.groupby(['body-style'],as_index= False).mean()
grouped_test2

'''Heat Map'''
#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

fig, ax=plt.subplots()
im=ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels=grouped_pivot.columns.levels[1]
col_labels=grouped_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0])+0.5, minor=False)
#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

'''Pearson Co-relation'''
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

'''ANOVA'''

grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
grouped_test2.head(2)
grouped_test2.get_group('4wd')['price']
# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   
#Separately: fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )

#4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)  

#4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)   

