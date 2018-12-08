import pandas as pd
filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(filename, names = headers)
print("Done")

# To see what the data set looks like, we'll use the head() method.
df.head()

import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)

#"True" stands for missing value, while "False" stands for not missing value
missing_data = df.isnull()
missing_data.head(5)

#Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")   
    
#Calculate the average of the column
avg_1 = df["normalized-losses"].astype("float").mean(axis = 0)

#Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_1, inplace = True)

#Calculate the mean value for 'bore' column
avg_2=df['bore'].astype('float').mean(axis=0)

#Replace NaN by mean value
df['bore'].replace(np.nan, avg_2, inplace= True)

#Calculate the mean value for 'stroke' column
avg_3 = df["stroke"].astype("float").mean(axis = 0)
# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_3, inplace = True)

#Calculate the mean value for the 'horsepower' column
avg_4=df['horsepower'].astype('float').mean(axis=0)
#Replace "NaN" by mean value
df['horsepower'].replace(np.nan, avg_4, inplace= True)

#Calculate the mean value for 'peak-rpm' column
avg_5=df['peak-rpm'].astype('float').mean(axis=0)
#Replace NaN by mean value
df['peak-rpm'].replace(np.nan, avg_5, inplace= True)

#To see which values are present in a particular column, we can use the ".value_counts()" method
df['num-of-doors'].value_counts()

#We can also use the ".idxmax()" method to calculate for us the most common type automatically
df['num-of-doors'].value_counts().idxmax()


#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace = True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace = True)

# reset index, because we droped two rows
df.reset_index(drop = True, inplace = True)

df.head()

#Convert data types to proper format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print("Done")
df.dtypes

'''Data Standrize'''

# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)

df.head()

''' Data Normalization '''
# replace (origianl value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

df['height'] = df['height']/df['height'].max() 
# show the scaled columns
df[["length","width","height"]].head()

'''Binning'''
df["horsepower"]=df["horsepower"].astype(float, copy=True)
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4
bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
bins
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)
df.head()

'''Indicator variable (or dummy variable)'''

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
#change column names for clarity
dummy_variable_1.rename(columns={'diesel':'fuel-type-diesel', 'gas':'fuel-type-gas'}, inplace=True)
dummy_variable_1.head()

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

#merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)


