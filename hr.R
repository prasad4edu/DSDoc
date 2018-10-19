names(employee_survey_data)
names(general_data)
names(manager_survey_data)

MergeDataset=merge(employee_survey_data,general_data,by.x ="EmployeeID",by.y ="EmployeeID"  )
MergeDataset=merge(MergeDataset,manager_survey_data,by.x ="EmployeeID",by.y ="EmployeeID"  )
names(MergeDataset)
dim(MergeDataset)
#null values
table(is.na(MergeDataset))
#colum wise null values
sapply(MergeDataset,function(x){
  (sum(is.na(x)))
})
#Null value percentage
apply(MergeDataset,2,function(x){
  ((sum(is.na(x)))/length(x))*100
})

library(imputeMissings)
AfterImputeDS=impute(MergeDataset)
#null values
anyNA(AfterImputeDS)
table(is.na(AfterImputeDS))
str(AfterImputeDS)

#Removing EmployeeID column name
EmpIDRemoveDS=subset(AfterImputeDS,select = -c(EmployeeID))
dim(EmpIDRemoveDS)
names(EmpIDRemoveDS)
View(EmpIDRemoveDS)
class(EmpIDRemoveDS)
str(EmpIDRemoveDS)
#change Divorced marital Status to Single
levels(EmpIDRemoveDS$MaritalStatus)
head(EmpIDRemoveDS$MaritalStatus,10)
EmpIDRemoveDS$MaritalStatus=ifelse(EmpIDRemoveDS$MaritalStatus=="Divorced","Single","Married")
class(EmpIDRemoveDS$MaritalStatus)
levels(as.factor(EmpIDRemoveDS$MaritalStatus))

#changing Attrition factor to numeric
table(EmpIDRemoveDS$Attrition)
EmpIDRemoveDS$Attrition=ifelse(EmpIDRemoveDS$Attrition=="Yes",1,0)
table(EmpIDRemoveDS$Attrition)

#converting levels to factor formate
EmpIDRemoveDS$Education=as.factor(EmpIDRemoveDS$Education)

class(EmpIDRemoveDS$EnvironmentSatisfaction)
EmpIDRemoveDS$EnvironmentSatisfaction=as.factor(EmpIDRemoveDS$EnvironmentSatisfaction)

EmpIDRemoveDS$JobInvolvement=as.factor(EmpIDRemoveDS$JobInvolvement)

EmpIDRemoveDS$JobLevel=as.factor(EmpIDRemoveDS$JobLevel)

class(EmpIDRemoveDS$JobSatisfaction)
EmpIDRemoveDS$JobSatisfaction=as.factor(EmpIDRemoveDS$JobSatisfaction)

class(EmpIDRemoveDS$PerformanceRating)
EmpIDRemoveDS$PerformanceRating=as.factor(EmpIDRemoveDS$PerformanceRating)

class(EmpIDRemoveDS$StockOptionLevel)
levels(as.factor(EmpIDRemoveDS$StockOptionLevel))
EmpIDRemoveDS$StockOptionLevel=as.factor(EmpIDRemoveDS$StockOptionLevel)

class(EmpIDRemoveDS$WorkLifeBalance)
EmpIDRemoveDS$WorkLifeBalance=as.factor(EmpIDRemoveDS$WorkLifeBalance)

EmpIDRemoveDS$MaritalStatus=as.factor(EmpIDRemoveDS$MaritalStatus)
levels(EmpIDRemoveDS$MaritalStatus)

#checking outliers
boxplot(EmpIDRemoveDS$Age)
boxplot(EmpIDRemoveDS$DistanceFromHome)
boxplot(EmpIDRemoveDS$MonthlyIncome)
boxplot(EmpIDRemoveDS$PercentSalaryHike)
boxplot(EmpIDRemoveDS$StandardHours)
boxplot(EmpIDRemoveDS$TotalWorkingYears)
quantile(EmpIDRemoveDS$TotalWorkingYears,c(0,0.5,0.75,0.9,0.997,1))
boxplot(EmpIDRemoveDS$TrainingTimesLastYear)
quantile(EmpIDRemoveDS$TrainingTimesLastYear,c(0,0.5,0.75,0.9,0.997,1))
boxplot(EmpIDRemoveDS$YearsAtCompany)
quantile(EmpIDRemoveDS$YearsAtCompany,c(0,0.5,0.75,0.9,0.997,1))
boxplot(EmpIDRemoveDS$YearsSinceLastPromotion)
quantile(EmpIDRemoveDS$YearsSinceLastPromotion,c(0,0.5,0.75,0.9,0.997,1))
boxplot(EmpIDRemoveDS$YearsWithCurrManager)
quantile(EmpIDRemoveDS$YearsWithCurrManager,c(0,0.5,0.75,0.9,0.997,1))
dim(EmpIDRemoveDS)
str(EmpIDRemoveDS)
#4410 observation 28 variables
library(dummies)
dummyDS=dummy.data.frame(EmpIDRemoveDS)

dim(dummyDS)
#After dummy function impliment total observations 4410 and variables  70
View(dummyDS)
head(dummyDS$Attrition)
class(dummyDS$Attrition)

# split the data training and validation
library(caret)
set_seed=createDataPartition(dummyDS$Attrition,p=0.8,list=FALSE)
dim(set_seed)
# get the training data
trian_data=dummyDS[set_seed,]
#get the validation data
val_data=dummyDS[-set_seed,]
dim(trian_data)
dim(val_data)

# information Gain  
library(Information)
iv=create_infotables(trian_data,y="Attrition",bins=10,parallel = FALSE)
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
write.csv(x1,"iv.csv")
# subset the data and select only important varaibles
ivTrianData=subset(trian_data,select=c("TotalWorkingYears",
                                       "YearsAtCompany",
                                       "Age",
                                       "YearsWithCurrManager",
                                       "NumCompaniesWorked",
                                       "EnvironmentSatisfaction1",
                                       "BusinessTravelTravel_Frequently",
                                       "JobSatisfaction4",
                                       "WorkLifeBalance1",
                                       "BusinessTravelNon-Travel",
                                       "EducationFieldHuman Resources",
                                       "YearsSinceLastPromotion",
                                       "DepartmentHuman Resources",
                                       "JobSatisfaction1",
                                       "MonthlyIncome",
                                       "DistanceFromHome",
                                       "WorkLifeBalance3",
                                       "TrainingTimesLastYear",
                                       "Attrition"))
View(ivTrianData)

# calculate the vif
library(car)
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)
vif(model1)
max(vif(model1))

# remove YearsAtCompany varible VIF=4.560484
ivTrianData=subset(ivTrianData,select = -c(YearsAtCompany))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)
vif(model1)
max(vif(model1))

# remove TotalWorkingYears varible VIF=2.390487
ivTrianData=subset(ivTrianData,select = -c(TotalWorkingYears))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)
vif(model1)
max(vif(model1))

# remove DistanceFromHome varible p value 0.658951
ivTrianData=subset(ivTrianData,select = -c(DistanceFromHome))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)

# remove MonthlyIncome varible p value 0.658951
ivTrianData=subset(ivTrianData,select = -c(MonthlyIncome))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)

#remove DepartmentHuman Resources  z value  2.062
ivTrianData=subset(ivTrianData,select = -c(`DepartmentHuman Resources`))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)

#remove WorkLifeBalance3 z value -2.450
ivTrianData=subset(ivTrianData,select = -c(WorkLifeBalance3))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)

#remove TrainingTimesLastYear zvalue  -3.015 
ivTrianData=subset(ivTrianData,select = -c(TrainingTimesLastYear))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)

#remove JobSatisfaction1 zvalue  3.361 
ivTrianData=subset(ivTrianData,select = -c(JobSatisfaction1))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)

#remove `BusinessTravelNon-Travel` zvalue  -3.607
ivTrianData=subset(ivTrianData,select = -c(BusinessTravelNon-Travel))
model1=glm(Attrition~.,family = binomial(),data = ivTrianData)
summary(model1)

#check the accuracy model
pred=predict(model1,type="response")
pred_class=ifelse(pred >0.5,1,0)
table(pred_class)
# draw the confusion matrix
conf_matrix1=table(ivTrianData$Attrition,pred_class)
conf_matrix1
# accuracy
accuracy1=(conf_matrix1[1,1]+conf_matrix1[2,2])/sum(conf_matrix1)
accuracy1
sensitivity(conf_matrix1)
specificity(conf_matrix1)
# construct the ROC cureve
library(pROC)
roc_curve=roc(ivTrianData$Attrition,pred)
plot(roc_curve)
# check the value roc
auc(roc_curve)

#chnage threshold value

#check the accuracy model
pred=predict(model1,type="response")
pred_class=ifelse(pred >0.7469,1,0)
table(pred_class)
# draw the confusion matrix
conf_matrix1=table(ivTrianData$Attrition,pred_class)
conf_matrix1
# accuracy
accuracy1=(conf_matrix1[1,1]+conf_matrix1[2,2])/sum(conf_matrix1)
accuracy1
sensitivity(conf_matrix1)
specificity(conf_matrix1)
# validating the model with validation data set
val_data$pred=predict(model1,val_data,type="response")
val_data$pred_class=ifelse(val_data$pred>0.7469,1,0)
conf_Val=table(val_data$Attrition,val_data$pred_class)
conf_Val
head(val_data$pred_class)
accuray_val=(conf_Val[1,1]+conf_Val[2,2])/sum(conf_Val)
accuray_val
accuracy1
# summary of the model1
summary(model1)
df=varImp(model1)
df

ggplot(df, aes(x=reorder( rownames(df),Overall), Overall)) + 
  geom_bar(stat='identity',fill = 'steelblue', color = 'black')+
  coord_flip()
