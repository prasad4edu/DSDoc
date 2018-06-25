import pandas as pd
import sklearn as sk
dir(sk)
import math
import numpy as np
from scipy import stats
import matplotlib as matlab
import statsmodels

###############LAB:Correlation Calculation########################
air=AirPassengers
#Dataset: Air Travel Data\Air_travel.csv
#Importing Air passengers data
air = pd.read_csv("G:\\DS_batch1\\datasets\\AirPassengers.csv")
air.shape
air.columns.values
air.info()
air.head(10)
air.describe()

#Find the correlation between number of passengers and promotional budget.
np.corrcoef(air.Passengers,air.Promotion_Budget)

#Draw a scatter plot between number of passengers and promotional budget
import matplotlib.pyplot as plt
plt.scatter(air.Passengers, air.Promotion_Budget)

#Find the correlation between number of passengers and Service_Quality_Score
matlab.pyplot.scatter(air.Passengers,air.Service_Quality_Score)
np.corrcoef(air.Passengers,air.Service_Quality_Score)

air.Passengers.mean()
air.Passengers.min()
air.Passengers.max()
air.Promotion_Budget.std()
np.sqrt(air.Promotion_Budget)
matlab.pyplot.scatter(air.Passengers,air.Service_Quality_Score)
##############################################Regression######################################

#Correlation between promotion and passengers count
np.corrcoef(air.Passengers,air.Promotion_Budget)

#Draw a scatter plot between   Promotion_Budget and Passengers. Is there any any pattern between Promotion_Budget and Passengers?
matlab.pyplot.scatter(air.Promotion_Budget,air.Passengers)

#Build a linear regression model and estimate the expected passengers for a Promotion_Budget is 650,000
##Regression Model  promotion and passengers count

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(air[["Promotion_Budget"]], air[["Passengers"]])
air['predictions'] = lr.predict(air[["Promotion_Budget"]])

predictions


import statsmodels.formula.api as sm
model = sm.ols(formula='Passengers ~ Promotion_Budget', data=air)
model
fitted1 = model.fit()
fitted1.summary()


#Build a regression line to predict the passengers using Inter_metro_flight_ratio

##Regression Model inter_metro_flight_ratio and passengers count
matlab.pyplot.scatter(air.Inter_metro_flight_ratio,air.Passengers)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(air[["Inter_metro_flight_ratio"]], air[["Passengers"]])
predictions = lr.predict(air[["Inter_metro_flight_ratio"]])

import statsmodels.formula.api as sm
model = sm.ols(formula='Passengers ~ Inter_metro_flight_ratio', data=air)
fitted2 = model.fit()
fitted2.summary()

#############################################################################
############ Lab:R Sqaure ##################
#What is the R-square value of Passengers vs Promotion_Budget model?
fitted1.summary()

#What is the R-square value of Passengers vs Inter_metro_flight_ratio

fitted2.summary()



################################################
#############Lab: Multiple Regerssion Model ####################
#Build a multiple regression model to predict the number of passengers
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(air[["Promotion_Budget"]+["Inter_metro_flight_ratio"]+["Service_Quality_Score"]], air[["Passengers"]])
predictions = lr.predict(air[["Promotion_Budget"]+["Inter_metro_flight_ratio"]+["Service_Quality_Score"]])
predictions

import statsmodels.formula.api as sm
model = sm.ols(formula='Passengers ~ Promotion_Budget+Service_Quality_Score+Inter_metro_flight_ratio', data=air)
fitted = model.fit()
fitted.summary()


#What is R-square value
0.949
#Are there any predictor variables that are not impacting the dependent variable 
##Promotion Budget,Inter_metro_flight_ratio,Service_Quality_Score.
 
###############################################
###############################################################################3
#####Multiple Regression- issues
    
#Import Final Exam Score data
final_exam=pd.read_csv("G:\\DS_batch1\\datasets\\Final Exam Score.csv")

#Size of the data
final_exam.shape

#Variable names
final_exam.columns

#First few observations
final_exam.head(10)

#Build a model to predict final score using the rest of the variables.
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(final_exam[["Sem1_Science"]+["Sem2_Science"]+["Sem1_Math"]+["Sem2_Math"]], final_exam[["Final_exam_marks"]])
predictions1 = lr1.predict(final_exam[["Sem1_Science"]+["Sem2_Science"]+["Sem1_Math"]+["Sem2_Math"]])
predictions1
import statsmodels.formula.api as sm
model1 = sm.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem1_Math+Sem2_Math', data=final_exam)
fitted1 = model1.fit()
fitted1
fitted1.summary()
fitted1.rsquared


#How are Sem2_Math & Final score related? As Sem2_Math score increases, what happens to Final score? 

#Remove "Sem1_Math" variable from the model and rebuild the model
from sklearn.linear_model import LinearRegression
lr2 = LinearRegression()
lr2.fit(final_exam[["Sem1_Science"]+["Sem2_Science"]+["Sem2_Math"]], final_exam[["Final_exam_marks"]])
predictions2 = lr2.predict(final_exam[["Sem1_Science"]+["Sem2_Science"]+["Sem2_Math"]])

import statsmodels.formula.api as sm
model2 = sm.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem2_Math', data=final_exam)
fitted2 = model2.fit()
fitted2.summary()


#Is there any change in R square or Adj R square

#How are Sem2_Math  & Final score related now? As Sem2_Math score increases, what happens to Final score? 


#Scatter Plot between the predictor variables
matlab.pyplot.scatter(final_exam.Sem1_Math,final_exam.Sem2_Math)

#Find the correlation between Sem1_Math & Sem2_Math 
np.correlate(final_exam.Sem1_Math,final_exam.Sem2_Math)

########################Multicollinearity detection#########################
##Testing Multicollinearity

from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(final_exam[["Sem1_Science"]+["Sem2_Science"]+["Sem1_Math"]+["Sem2_Math"]], final_exam[["Final_exam_marks"]])
predictions1 = lr1.predict(final_exam[["Sem1_Science"]+["Sem2_Science"]+["Sem1_Math"]+["Sem2_Math"]])

import statsmodels.formula.api as sm
model1 = sm.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem1_Math+Sem2_Math', data=final_exam)
fitted1 = model1.fit()
fitted1.summary()
fitted1.summary2()
  x_vars=final_exam.drop(["Final_exam_marks"], axis=1)
  xvar_names=x_vars.columns
  xvar_names.shape[0]
#Code for VIF Calculation

#Writing a function to calculate the VIF values

def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

#Calculating VIF values using that function
vif_cal(input_data=final_exam, dependent_col="Final_exam_marks")

#VIF Values given by statsmodels.stats.outliers_influence.variance_inflation_factor are not accurate
#import statsmodels.stats.outliers_influence
#help(statsmodels.stats.outliers_influence.variance_inflation_factor)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 0)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 1)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 2)
#statsmodels.stats.outliers_influence.variance_inflation_factor(final_exam.drop(["Final_exam_marks"], axis=1).as_matrix(), 3)


import statsmodels.formula.api as sm
model2 = sm.ols(formula='Final_exam_marks ~ Sem1_Science+Sem2_Science+Sem2_Math', data=final_exam)
fitted2 = model2.fit()
fitted2.summary()

vif_cal(input_data=final_exam.drop(["Sem1_Math"], axis=1), dependent_col="Final_exam_marks")
vif_cal(input_data=final_exam.drop(["Sem1_Math","Sem1_Science"], axis=1), dependent_col="Final_exam_marks")




















