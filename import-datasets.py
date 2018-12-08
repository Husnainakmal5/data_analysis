# import pandas library
import pandas as pd
# read the online file by the URL provides above, and assign it to variable "df"
path="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(path,header=None)
print("Done")

# show the first 5 rows using dataframe.head() method
df.head(5)

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
headers

#Adding header to dataframe
df.columns = headers
df.head(10)

#Droping missing values from columns
df.dropna(subset=["price"], axis=0)

# check the data type of data frame "df" by .dtypes
df.dtypes

#describe all the columns
df.describe()

# describe all the columns in "df" 
df.describe(include = "all")

#describe selected columns
df[['length','compression-ratio']].describe()
x=df['length']
y=df[['length']]

# look at the info of "df"
df.info