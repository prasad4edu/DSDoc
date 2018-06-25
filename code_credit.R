#Doing the project for credit card data
#calculate the number of observations and number of varables
dim(credit)
# names of the variables
names(credit)
# strcuture of the data
str(credit)
# check is there any missing values in the given data
anyNA(credit)
sum(is.na(credit))
#checking the missing value for each varaible
number_nans=function(x)
{
  y=sum(is.na(x))
  
}
sapply(credit, number_nans)
#cross verifying the missing values for variables
sum(is.na(credit$PAY_AMT6))
# impute the missing values
library(imputeMissings)
credit1=impute(credit)
# cross verify the is there any further missing values in our data
anyNA(credit1)
# is there any outliers in our data
#lets check the outliers for limit balance
boxplot(credit1$LIMIT_BAL)
#how to check number of outliers
quantile(credit1$LIMIT_BAL,c(0,0.25,0.5,0.75,0.95,0.997,1))
# check the outliers for Age variables
boxplot(credit1$AGE)
#checking is there any outliers by using the quantile function
quantile(credit1$AGE,c(0,0.25,0.5,0.75,0.9,0.95,0.997,1))
#let's check is there any outliers in bill_amt1
boxplot(credit1$BILL_AMT1)
quantile(credit1$BILL_AMT1,c(0,0.25,0.5,0.75,0.95,0.997,1))


#check how many customers where there bill amount is <0
x=ifelse(credit1$BILL_AMT1<0,1,0)
table(x)
# replace these 590 customers with 0 balance
credit1$BILL_AMT1=ifelse(credit1$BILL_AMT1<0,0,credit1$BILL_AMT1)
# to cross verify what is the min value of bill_amt1
min(credit1$BILL_AMT1)
#checking the outliers for balance amount2
boxplot(credit1$BILL_AMT2)
#checking the outliers by using the quantile function
quantile(credit1$BILL_AMT2,c(0,0.25,0.5,0.75,0.95,0.997,1))
#checking the how many number of records which are less than 0
x=ifelse(credit1$BILL_AMT2 <0,1,0)
table(x)
# in the place of these 669 records which are in negative, we are 
# we are going to impute with 0 value
credit1$BILL_AMT2=ifelse(credit1$BILL_AMT2<0,0,credit1$BILL_AMT2)
# to cross verify the bill amount2
min(credit1$BILL_AMT2)
#chekcing the outliers for Bill amount3
quantile(credit1$BILL_AMT3,c(0,0.5,0.75,0.95,0.997,1))
# replace the bill amount less than 0 to 0 value
credit1$BILL_AMT3=ifelse(credit1$BILL_AMT3<0,0,credit1$BILL_AMT3)
credit1$BILL_AMT3=ifelse(credit1$BILL_AMT3 > 434017.46,median(credit1$BILL_AMT3),credit1$BILL_AMT3)
boxplot(credit1$BILL_AMT3)
# checking the outliers for Bill amount 4
quantile(credit1$BILL_AMT4,c(0,0.5,0.75,0.9,0.997,1))
# replacing the negative values with 0 
credit1$BILL_AMT4=ifelse(credit1$BILL_AMT4<0,0,credit1$BILL_AMT4)
# to verify what is the min value
min(credit1$BILL_AMT4)
# checking the outliers for bill amount 5
quantile(credit1$BILL_AMT5,c(0,0.5,0.75,0.95,0.997,1))
# replace the negative values with 0
credit1$BILL_AMT5=ifelse(credit1$BILL_AMT5<0,0,credit1$BILL_AMT5)
# to verify min value
min(credit1$BILL_AMT5)
# checking the outliers for Bill amount 6
quantile(credit1$BILL_AMT6,c(0,0.5,0.75,0.95,0.997,1))
# replace the negative values with 0
credit1$BILL_AMT6=ifelse(credit1$BILL_AMT6<0,0,credit1$BILL_AMT6)
# checking the outliers for payment variables
quantile(credit1$PAY_AMT1,c(0,0.5,0.75,0.95,0.997,1))
boxplot(credit1$PAY_AMT1)
# impute outliers with median value for pay amt1 variable
credit1$PAY_AMT1=ifelse(credit1$PAY_AMT1>140015.17,median(credit1$PAY_AMT1),credit1$PAY_AMT1)
# to cross verify pay amt 1 variable for outliers
boxplot(credit1$PAY_AMT1)
# checking the outliers for payment variables
quantile(credit1$PAY_AMT2,c(0,0.5,0.75,0.95,0.997,1))
credit1$PAY_AMT2=ifelse(credit1$PAY_AMT2>150062.1,median(credit1$PAY_AMT2),credit1$PAY_AMT2)
# to cross verify pay amt 1 variable for outliers
boxplot(credit1$PAY_AMT2)
# check the outliers for pay amount 3
quantile(credit1$PAY_AMT3,c(0,0.5,0.75,0.95,0.997,1))
boxplot(credit1$PAY_AMT3)
# treating the outliers in pay amount3 variable by using median value
credit1$PAY_AMT3=ifelse(credit1$PAY_AMT3>136322.34,median(credit1$PAY_AMT3),credit1$PAY_AMT3)
# draw the boxplot for pay amount 3
boxplot(credit1$PAY_AMT3)
# check the outliers for pay amount 4
quantile(credit1$PAY_AMT4,c(0,0.25,0.5,0.997,1))
# repalcing the outliers with median value
credit1$PAY_AMT4=ifelse(credit1$PAY_AMT4>130405.1,median(credit1$PAY_AMT4),credit1$PAY_AMT4)
# re verify the boxplot
boxplot(credit1$PAY_AMT4)
# checking the outliers for pay amount 5
quantile(credit1$PAY_AMT5,c(0,0.5,0.75,0.997,1))
credit1$PAY_AMT5=ifelse(credit1$PAY_AMT5>132202,median(credit1$PAY_AMT5),credit1$PAY_AMT5)
boxplot(credit1$PAY_AMT5)
# checking the outliers for pay amount 6
quantile(credit1$PAY_AMT6,c(0,0.5,0.75,0.997,1))
boxplot(credit1$PAY_AMT6)
credit1$PAY_AMT6=ifelse(credit1$PAY_AMT6>167000.3,median(credit1$PAY_AMT6),credit1$PAY_AMT6)
boxplot(credit1$PAY_AMT6)
# count the defaulters vs non defaulters
table(credit1$default.payment.next.month)
credit1$target=credit1$default.payment.next.month
credit1=subset(credit1,select = -c(default.payment.next.month))
# check the count of target variable
table(credit1$target)
# the default rate in our data is 22.12%

# draw the insights for Sex variable vs default varaiable
table(credit1$SEX,credit1$target)
table(credit1$EDUCATION)
credit1$edu=ifelse(credit1$EDUCATION>4 | credit1$EDUCATION==0, 5,credit1$EDUCATION)
table(credit1$edu)
# finding out the insights for education variables
table(credit1$edu,credit1$target)
#droping the oriignal variable education
credit1=subset(credit1,select=-c(EDUCATION))

#checking the type of sex varaible
class(credit1$SEX)
table(credit1$SEX)
''' in our given data set there are 11888-male
18112- female customer'''

# converting the all level numbers into new varaible by each level
# is called as dummy coding or one hot coding
# if the variable in numeric format, if we want convert them into a
# level varaible, the type of variable should be 
#in factor or character format

class(credit1$SEX)
# converting sex varaible into factor format
credit1$SEX=as.factor(credit1$SEX)
class(credit1$SEX)
credit1$edu=as.factor(credit1$edu)
class(credit1$edu)
credit1$MARRIAGE=as.factor(credit1$MARRIAGE)
class(credit1$MARRIAGE)
credit1$PAY_0=as.factor(credit1$PAY_0)
credit1$PAY_2=as.factor(credit1$PAY_2)
credit1$PAY_3=as.factor(credit1$PAY_3)
credit1$PAY_4=as.factor(credit1$PAY_4)
credit1$PAY_5=as.factor(credit1$PAY_5)
credit1$PAY_6=as.factor(credit1$PAY_6)
# to convert level varaibles into new varaibles by using dummies 
library(dummies)
credit2=dummy.data.frame(credit1)
# split the data training and validation
library(caret)
set_seed=createDataPartition(credit2$target,p=0.8,list=FALSE)
# get the training data
trian_data=credit2[set_seed,]
#get the validation data
val_data=credit2[-set_seed,]

# in the varaible reduction we are going to use three methods
# information value
# correlation
# VIF
# information value- it will give us the predicting power 
#of an each and individual vairable
# we are using package called information
library(Information)
iv=create_infotables(trian_data,y="target",bins=10,parallel = FALSE)
iv
iv$Tables
iv$Summary
x=iv$Summary
x
View(x)
x$flag=ifelse(x$IV >0.02 & x$IV < 0.5,1,0)
View(x)
table(x$flag)
x1=subset(x,flag==1)
write.csv(x1,"C:\\Koti\\Workshop\\iv.csv")
# subset the data and select only important varaibles
trian_data1=subset(trian_data,select=c("PAY_22",
                                       "PAY_32",
                                       "PAY_42",
                                       "PAY_00",
                                       "PAY_52",
                                       "PAY_62",
                                       "LIMIT_BAL",
                                       "PAY_AMT1",
                                       "PAY_20",
                                       "PAY_AMT2",
                                       "PAY_AMT3",
                                       "PAY_03",
                                       "PAY_AMT4",
                                       "PAY_AMT6",
                                       "PAY_AMT5",
                                       "PAY_30",
                                       "PAY_01",
                                       "PAY_40",
                                       "PAY_50",
                                       "PAY_3-1",
                                       "PAY_23",
                                       "PAY_60",
                                       "PAY_2-1",
                                       "PAY_4-1",
                                       "PAY_0-2",
                                       "PAY_5-1",
                                       "PAY_53",
                                       "PAY_0-1",
                                       "PAY_63",
                                       "PAY_43",
                                       "PAY_6-1",
                                       "PAY_33",
                                       "PAY_47",
                                       "PAY_57",
                                       "AGE","target"))
                                       

# calculate the vif
library(car)
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)
# removing the pay_20 variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_20))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)
# removing the pay_40 because high vif (13.213305)
trian_data1=subset(trian_data1,select=-c(PAY_40))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)
# removing the pay_50 variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_50))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the pay_30 variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_30))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the pay_47 variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_47))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the pay_00 variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_00))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the pay_2_1variable because of high VIF
trian_data1=subset(trian_data1,select=-c(`PAY_2-1`))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the pay_60 variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_60))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the `PAY_4-1`  variable because of high VIF
trian_data1=subset(trian_data1,select=-c(`PAY_4-1` ))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the PAY_52  variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_52))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the `PAY_5-1`  variable because of high VIF
trian_data1=subset(trian_data1,select=-c(`PAY_5-1`))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)

# removing the PAY_32  variable because of high VIF
trian_data1=subset(trian_data1,select=-c(PAY_32))
#again we are buidling the model
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
#checking vif for each and every varaible
vif(model1)
# we have identified good 24 variables, on this variables we built the model
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)
# removing the pay_33 because of least siginficant p-value
trian_data1=subset(trian_data1,select = -c(PAY_33))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)

# removing the pay_3_1 because of least siginficant p-value
trian_data1=subset(trian_data1,select = -c(`PAY_3-1`))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)

# removing the pay_53 because of least siginficant p-value
trian_data1=subset(trian_data1,select = -c(PAY_53))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)

# removing the pay_6_1 because of least siginficant p-value
trian_data1=subset(trian_data1,select = -c(`PAY_6-1` ))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)


# removing the pay_01 because of least siginficant p-value
trian_data1=subset(trian_data1,select = -c(PAY_01))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)


# removing the pay_0_1 because of least siginficant p-value
trian_data1=subset(trian_data1,select = -c(`PAY_0-1`))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)
# dropping the pay_amt5 because least contribution to the model
trian_data1=subset(trian_data1,select =-c(PAY_AMT5))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)
# dropping the pay_0_2 because least contribution to the model
trian_data1=subset(trian_data1,select =-c(`PAY_0-2` ))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)
# dropping the PAY_AMT4  because least contribution to the model
trian_data1=subset(trian_data1,select =-c(PAY_AMT4  ))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)

# dropping the PAY_AMT6  because least contribution to the model
trian_data1=subset(trian_data1,select =-c(PAY_AMT6))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)
# dropping the PAY_43  because least contribution to the model
trian_data1=subset(trian_data1,select =-c(PAY_43))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)

# dropping the PAY_AMT1  because least contribution to the model
trian_data1=subset(trian_data1,select =-c(PAY_AMT1))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)

# dropping the PAY_57  because least contribution to the model
trian_data1=subset(trian_data1,select =-c(PAY_57))
# Build the logistic regression model

model2=glm(target~.,family = binomial(),data=trian_data1)
summary(model2)
#check the accuracy model
pred=predict(model2,type="response")
pred_class=ifelse(pred >0.5,1,0)
table(pred_class)
# draw the confusion matrix
conf_matrix1=table(trian_data1$target,pred_class)
conf_matrix1
# accuracy
accuracy1=(conf_matrix1[1,1]+conf_matrix1[2,2])/sum(conf_matrix1)
accuracy1
sensitivity(conf_matrix1)
specificity(conf_matrix1)
# construct the ROC cureve
library(pROC)
roc_curve=roc(trian_data1$target,pred)
plot(roc_curve)
# check the value roc
auc(roc_curve)

#chnage threshold value


#check the accuracy model
pred=predict(model2,type="response")
pred_class=ifelse(pred >0.708,1,0)
table(pred_class)
# draw the confusion matrix
conf_matrix1=table(trian_data1$target,pred_class)
conf_matrix1
# accuracy
accuracy1=(conf_matrix1[1,1]+conf_matrix1[2,2])/sum(conf_matrix1)
accuracy1
sensitivity(conf_matrix1)
specificity(conf_matrix1)
# validating the model with validation data set
val_data$pred=predict(model2,val_data,type="response")
val_data$pred_class=ifelse(val_data$pred>0.708,1,0)
conf_Val=table(val_data$target,val_data$pred_class)
conf_Val
accuray_val=(conf_Val[1,1]+conf_Val[2,2])/sum(conf_Val)
accuray_val
accuracy1
# summary of the model2
summary(model2)
