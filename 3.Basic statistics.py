#Basic Statistics, Graphs and Reports
#Taking a random sample 
import pandas as pd
#view all the names(functions) in a module on pd
dir(pd)
####################Sampling in R#############################
#Taking a random sample 
import pandas as pd
Online_Retail=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Online Retail.csv", encoding = "ISO-8859-1")
Online_Retail.columns.values
# is there any duplicates at invoice date
sum(Online_Retail.InvoiceDate.duplicated())
# is there any duplicates at Online retail dataset level
sum(Online_Retail.duplicated())

sample_data=Online_Retail.sample(n=500)
sample_data=Online_Retail.sample(n=1000,replace="True")
sample_data.shape
sum(sample_data.InvoiceNo.duplicated())
sum(sample_data.duplicated())

# lets check the varaible names in Online retail data
list(Online_Retail.columns)
#check the type of each and every varaible
Online_Retail.dtypes
#Perform the basic statstics to numerical variables
#calculate the mean, median and mode
Online_Retail.UnitPrice.mean()
Online_Retail.UnitPrice.median()
Online_Retail.UnitPrice.mode()

# Calculate the min value and maximum value
Online_Retail.UnitPrice.min()
Online_Retail.UnitPrice.max()
# calculate the variance and std
Online_Retail.UnitPrice.var()
Online_Retail.UnitPrice.std()










#####################LAB: Sampling in python#############################

#Import “Census Income Data/Income_data.csv”
Income=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Income_data.csv")
Income.shape
Income.head(10)
Income.tail(3)
sum(Income.duplicated())
 #Sample size 5000
Sample_income=Income.sample(n=5000)
Sample_income.shape
sum(Sample_income.duplicated())
# columns names in income data
Income.columns
# finding out data type each and every object
Income.dtypes
Income.info()
#####################Descriptive statistics#####################
#Import “Census Income Data/Income_data.csv”
Income=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Income_data.csv")

Income.columns.values

#Mean and Median on python
gain_mean=Income["capital-gain"].mean()
gain_mean

gain_median=Income["capital-gain"].median()
gain_median

gain_std=Income["capital-gain"].std()
gain_std




#####################LAB: Mean and Median on python#####################
Online_Retail=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Online Retail.csv", encoding = "ISO-8859-1")
Online_Retail.columns.values
Online_Retail.dtypes
#Mean and median of 'UnitPrice' in Online Retail data
up_mean=Online_Retail['UnitPrice'].mean()
up_mean

up_median=Online_Retail['UnitPrice'].median()
up_median

#Mean of "Quantity" in Online Retail data
Quantity_mean=Online_Retail['Quantity'].mean()
Quantity_mean

Quantity_median=Online_Retail['Quantity'].median()
Quantity_median

#####################Dispersion Measures#####################

Income["native-country"].value_counts()

#####################Variance and Standard deviation#####################
usa_income=Income[Income["native-country"]==' United-States']
usa_income.shape
Income.columns.values
other_income=Income[Income["native-country"]!=' United-States']
other_income.shape

#Var and SD for USA
var_usa=usa_income["education-num"].var()
var_usa

std_usa=usa_income["education-num"].std()
std_usa

var_other=other_income["education-num"].var()
var_other

std_other=other_income["education-num"].std()
std_other 

range=(other_income["education-num"].max()-other_income["education-num"].min())
max_other 
min_other=other_income["education-num"].min()
min_other

#####################LAB: Variance and Standard deviation#####################
##var and sd UnitPrice
var_UnitPrice=Online_Retail['UnitPrice'].var()
var_UnitPrice

std_UnitPrice=Online_Retail['UnitPrice'].std()
std_UnitPrice 

#variance and sd of Quantity
var_UnitPrice=Online_Retail['Quantity'].var()
var_UnitPrice

std_UnitPrice=Online_Retail['Quantity'].std()
std_UnitPrice 

######################Percentiles & Quartiles #####################

Income["capital-gain"].describe()

#Finding the percentile & quantile by using .quantile()
Income['capital-gain'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Income['capital-loss'].quantile([0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
Income['hours-per-week'].quantile([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.997,1])

######################LAB: Percentiles & quartiles in python######################
bank=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\bank_market.csv",encoding = "ISO-8859-1")
bank.shape

#Get the summary of the balance variable
#we can find the summary of the balance variable by using .describe()
summary_bala=bank["balance"].describe()
summary_bala

#Get relevant percentiles and see their distribution.
bank['balance'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

#Get the summary of the age variable
summary_age=bank['age'].describe()
summary_age

#Get relevant percentiles and see their distribution
bank['age'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

######################LAB: Box plots and outlier detection######################
#Do you suspect any outliers in balance
bank=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\bank_market.csv",encoding = "ISO-8859-1")

import matplotlib.pyplot as plt
dir(plt)
#Basic plot of boxplot by importing the matplot.pyplot as plt ("plt.boxplot())
plt.boxplot(bank.balance)

#Get relevant percentiles and see their distribution
bank['balance'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95,0.997, 1])
#Do you suspect any outliers in balance
# outlier are present in balance variable

#Do you suspect any outliers in age
#detect the ouliers in age variable by plt.boxplot()
plt.boxplot(bank.age)
#No outliers are present
#Get relevant percentiles and see their distribution
bank['age'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,0.997,1])
#Do you suspect any outliers in age
#outliers are not present in age variable


######################Creating Graphs ################################

##Scatter Plot:

cars=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\AutoDataset.csv",encoding = "ISO-8859-1")
cars.shape
# variable names of cars data
list(cars.columns)

cars[' horsepower'].describe()
cars[' city-mpg'].describe()

import matplotlib.pyplot as plt
dir(plt)
plt.plot(cars[' horsepower'],cars[' city-mpg'])
plt.scatter(cars[' horsepower'],cars[' city-mpg'])
plt.hist(Online_Retail['UnitPrice'])
######################LAB:Creating Graphs ################################

import matplotlib.pyplot as plt


#Sports data
sports_data=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Sporting_goods_sales.csv")
sports_data.head(10)

#Draw a scatter plot between Average_Income and Sales. Is there any relation between two variables
plt.scatter(sports_data.Average_Income,sports_data.Sales)

#Draw a scatter plot between Under35_Population_pect and Sales. Is there any relation between two

plt.scatter(sports_data.Under35_Population_pect,sports_data.Sales,color="red")
# Draw a histogram plt.hist()
plt.hist(sports_data.Under35_Population_pect)
######################Bar Chart######################
#Bar charts used to summarize the categorical variables
\######################LAB: Bar Chart######################
sports_data=pd.read_csv("D:\\Datasets\\Sporting_goods_sales\\Sporting_goods_sales.csv",encoding = "ISO-8859-1")
sports_data.shape
sports_data.columns.values


freq=sports_data.Avg_family_size.value_counts()
freq.values
freq.index

import matplotlib.pyplot as plt
plt.bar(freq.index,freq.values)

freq=Online_Retail.Country.value_counts()
freq
freq.values
freq.index
# generate the bar plot for country variable
plt.bar(freq.index,freq.values)
plt.hist(freq.values)
# generate boxplot
plt.boxplot(Online_Retail["UnitPrice"])

######################Trend Chart######################
AirPassengers=pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\AirPassengers.csv", encoding = "ISO-8859-1")
AirPassengers.head()
AirPassengers.columns.values
plt.boxplot(AirPassengers.Passengers)
import matplotlib.pyplot as plt
plt.plot(AirPassengers.Passengers)
#compare two numerical variables
plt.scatter(AirPassengers.Passengers,AirPassengers.Promotion_Budget)
# required to have numpy package/library to perform the correlation

import numpy as np
np.corrcoef(AirPassengers.Passengers,AirPassengers.Promotion_Budget)
np.corrcoef(AirPassengers.Passengers,AirPassengers.Passengers)

plt.scatter(AirPassengers.Passengers,AirPassengers.Passengers)
list(AirPassengers.columns)


# draw the scatter plot between Passengers' and Service_Quality_Score
plt.scatter(AirPassengers.Passengers,AirPassengers.Service_Quality_Score)

# draw the correlation for the above two varaibles
np.corrcoef(AirPassengers.Passengers,AirPassengers.Service_Quality_Score)








