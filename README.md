##1. Impoting DataSet

Data Acquisition from [Here](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data)
Reading Data
Adding Header
Save Dataset using **df.to_csv()**

##2. Pre-Processing

### Missing Data
**Steps for working with missing data:**

identify missing data
deal with missing data
correct data format

**Deal with missing data**
How to deal with missing data?

1. drop data 
    a. drop the whole row
    b. drop the whole column
2. replace data
    a. replace it by mean
    b. replace it by frequency
    c. replace it based on other functions
    
### Data Standardization
Standardization is the process of transforming data into a common format which allows the researcher to make the meaningful comparison.

### Data Normalization
Normalization is the process of transforming values of several variables into a similar range.

There are several different techniques for Normalization e.g.
'Simple feature scaling'  **X**new=**X**old/**X**max 
'Min-Max'                 **X**new=(**X**old-**X**min)/(**X**max--**X**min)
'Z-score'                 **X**new=(**X**old-μ)/σ

### Binning
Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.

### Indicator variable (or dummy variable)
An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves do not have inherent meaning.

##3. Exploratory Data Analysis
Data Acquisition from [Here](https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv)

### Continuous numerical variables
Continuous numerical variables are variables that may contain any value within some range. Continuous numerical variables can have the type "int64" or "float64". A great way to visualize these variables is by using scatterplots with fitted lines. 

### Correlation and Causation

**Correlation:** a measure of the extent of interdependence between variables.

**Causation:** the relationship between cause and effect between two variables.

#### Pearson Correlation
The Pearson Correlation measures the linear dependence between two variables X and Y. The resulting coefficient is a value between -1 and 1 inclusive, where:

 1: total positive linear correlation,
 0: no linear correlation, the two variables most likely do not affect each other
-1: total negative linear correlation.


**P-value:** What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.

By convention, when the

p-value is < 0.001 we say there is strong evidence that the correlation is significant,
the p-value is < 0.05; there is moderate evidence that the correlation is significant,
the p-value is < 0.1; there is weak evidence that the correlation is significant, and
the p-value is > 0.1; there is no evidence that the correlation is significant.

### ANOVA

#### ANOVA: Analysis of Variance
The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant differences between the means of two or more groups. ANOVA returns two parameters:

**F-test score:** ANOVA assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption, and reports it as the F-test score. A larger score means there is a larger difference between the means.

**P-value:** P-value tells how statistically significant is our calculated score value

If our price variable is strongly correlated with the variable we are analyzing, expect ANOVA to return a sizeable F-test score and a small p-value.

##4. Model Development

### Linear Regression
One example of a Data Model that we will be using is Simple Linear Regression. Simple Linear Regression is a method to help us understand the relationship between two variables:

The predictor/independent variable (X)
The response/dependent variable (that we want to predict)(Y)
The result of Linear Regression is a linear function that predicts the response (dependent) variable as a function of the predictor (independent) variable.

𝑌:𝑅𝑒𝑠𝑝𝑜𝑛𝑠𝑒 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒
𝑋:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠

**Linear function:**

**𝑌=a+bX**

a refers to the intercept of the regression, in other words: the value of Y when X is 0
b refers to the slope of the regression line, in other words: the value with which Y changes when X increases by 1.

### Multiple Linear Regression
** 𝑌=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4 **
𝑌:𝑅𝑒𝑠𝑝𝑜𝑛𝑠𝑒 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒
𝑋1:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 1
𝑋2:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 2
𝑋3:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 3
𝑋4:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 4

𝑎:𝑖𝑛𝑡𝑒𝑟𝑐𝑒𝑝𝑡
𝑏1:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 1
𝑏2:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 2
𝑏3:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 3
𝑏4:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 4

### Polynomial Regression and Pipelines

**Quadratic - 2nd order or higher order**

#### Pipeline

Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.

### Measures for In-Sample Evaluation

Two very important measures that are often used in Statistics to determine the accuracy of a model are:

**R^2 / R-squared**
**Mean Squared Error (MSE)**

**R-squared**

R squared, also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line. The value of the R-squared is the percentage of variation of the response variable (y) that is explained by a linear model.

**Mean Squared Error (MSE)**

The Mean Squared Error measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).

**the model with the higher R-squared value is a better fit**
**the model with the smallest MSE value is a better fit**

