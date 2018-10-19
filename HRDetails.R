names(employee_survey_data)
names(general_data)
names(manager_survey_data)

MergeDataset=merge(employee_survey_data,general_data,by.x ="EmployeeID",by.y ="EmployeeID"  )
MergeDataset=merge(MergeDataset,manager_survey_data,by.x ="EmployeeID",by.y ="EmployeeID"  )
names(MergeDataset)
dim(MergeDataset)
sapply(MergeDataset,function(x){
      (sum(is.na(x)))
  })

apply(MergeDataset,2,function(x){
   ((sum(is.na(x)))/length(x))*100
})
library(imputeMissings)
AfterImputeDS=impute(MergeDataset)
anyNA(AfterImputeDS)
str(AfterImputeDS)
#Removing EmployeeID column name
EmpIDRemoveDS=subset(AfterImputeDS,select = -c(EmployeeID))
dim(EmpIDRemoveDS)
names(EmpIDRemoveDS)
View(EmpIDRemoveDS)
class(EmpIDRemoveDS)
str(EmpIDRemoveDS)

#outlier
boxplot(as.factor(EmpIDRemoveDS$EnvironmentSatisfaction))
boxplot(EmpIDRemoveDS$EnvironmentSatisfaction)
boxplot(EmpIDRemoveDS$JobSatisfaction)
boxplot(EmpIDRemoveDS$WorkLifeBalance)
boxplot(EmpIDRemoveDS$Age)
#boxplot(EmpIDRemoveDS$Attrition)
head(EmpIDRemoveDS$Attrition)
#boxplot(EmpIDRemoveDS$BusinessTravel)
#boxplot(EmpIDRemoveDS$Department)
boxplot(EmpIDRemoveDS$DistanceFromHome)
boxplot(EmpIDRemoveDS$Education)
#boxplot(EmpIDRemoveDS$EducationField)
#boxplot(EmpIDRemoveDS$EmployeeCount)
#boxplot(EmpIDRemoveDS$Gender)
boxplot(EmpIDRemoveDS$JobLevel)
#boxplot(EmpIDRemoveDS$JobRole)
#boxplot(EmpIDRemoveDS$MaritalStatus)
boxplot(EmpIDRemoveDS$MonthlyIncome)
max(EmpIDRemoveDS$MonthlyIncome)
quantile(EmpIDRemoveDS$MonthlyIncome,c(0,0.5,0.75,0.9,0.997,1))
hist(EmpIDRemoveDS$MonthlyIncome)
EmpIDRemoveDS$MonthlyIncome=ifelse(EmpIDRemoveDS$MonthlyIncome>198590,median(EmpIDRemoveDS$MonthlyIncome),EmpIDRemoveDS$MonthlyIncome)

boxplot(EmpIDRemoveDS$NumCompaniesWorked)
quantile(EmpIDRemoveDS$NumCompaniesWorked,c(0,0.5,0.75,0.9,0.997,1))

AfterOutlinerDS=EmpIDRemoveDS
str(AfterOutlinerDS)

#boxplot(EmpIDRemoveDS$Over18)
levels(factor(EmpIDRemoveDS$Over18))

boxplot(EmpIDRemoveDS$PercentSalaryHike)
boxplot(EmpIDRemoveDS$StandardHours)
head(EmpIDRemoveDS$StandardHours)
boxplot(EmpIDRemoveDS$StockOptionLevel)
head(EmpIDRemoveDS$StockOptionLevel)
max(EmpIDRemoveDS$StockOptionLevel)
quantile(EmpIDRemoveDS$StockOptionLevel,c(0,0.5,0.75,0.9,0.997,1))

boxplot(EmpIDRemoveDS$TotalWorkingYears)
max(EmpIDRemoveDS$TotalWorkingYears)
quantile(EmpIDRemoveDS$TotalWorkingYears,c(0,0.5,0.75,0.9,0.997,1))
hist(EmpIDRemoveDS$TotalWorkingYears)

boxplot(EmpIDRemoveDS$TrainingTimesLastYear)
min(EmpIDRemoveDS$TrainingTimesLastYear)
max(EmpIDRemoveDS$TrainingTimesLastYear)
quantile(EmpIDRemoveDS$TrainingTimesLastYear,c(0,0.5,0.75,0.9,0.997,1))

boxplot(EmpIDRemoveDS$YearsAtCompany)
max(EmpIDRemoveDS$YearsAtCompany)
quantile(EmpIDRemoveDS$YearsAtCompany,c(0,0.5,0.75,0.9,0.997,1))
hist(EmpIDRemoveDS$YearsAtCompany)

boxplot(EmpIDRemoveDS$YearsSinceLastPromotion)
quantile(EmpIDRemoveDS$YearsSinceLastPromotion,c(0,0.5,0.75,0.9,0.997,1))
hist(EmpIDRemoveDS$YearsSinceLastPromotion)

boxplot(EmpIDRemoveDS$YearsWithCurrManager)
quantile(EmpIDRemoveDS$YearsWithCurrManager,c(0,0.5,0.75,0.9,0.997,1))
hist(EmpIDRemoveDS$YearsWithCurrManager)

boxplot(EmpIDRemoveDS$JobInvolvement)
str(EmpIDRemoveDS$jobI)
boxplot(EmpIDRemoveDS$PerformanceRating)
quantile(EmpIDRemoveDS$PerformanceRating,c(0,0.5,0.75,0.9,0.997,1))

#converting into factors
str(EmpIDRemoveDS)
factorDS=EmpIDRemoveDS
factorDS$EnvironmentSatisfaction=as.factor(factorDS$EnvironmentSatisfaction)
factorDS$JobSatisfaction=as.factor(factorDS$JobSatisfaction)
factorDS$WorkLifeBalance=as.factor(factorDS$WorkLifeBalance)
factorDS$Age=as.factor(factorDS$Age)
factorDS$Attrition=as.factor(factorDS$Attrition)
factorDS$BusinessTravel=as.factor(factorDS$BusinessTravel)
factorDS$Department=as.factor(factorDS$Department)
factorDS$DistanceFromHome=as.factor(factorDS$DistanceFromHome)
factorDS$Education=as.factor(factorDS$Education)
factorDS$EducationField=as.factor(factorDS$EducationField)
factorDS$EmployeeCount=as.factor(factorDS$EmployeeCount)
factorDS$Gender=as.factor(factorDS$Gender)
factorDS$JobLevel=as.factor(factorDS$JobLevel)
factorDS$JobRole=as.factor(factorDS$JobRole)
factorDS$MaritalStatus=as.factor(factorDS$MaritalStatus)
factorDS$MonthlyIncome=as.factor(factorDS$MonthlyIncome)
factorDS$NumCompaniesWorked=as.factor(factorDS$NumCompaniesWorked)
factorDS$Over18=as.factor(factorDS$Over18)
factorDS$PercentSalaryHike=as.factor(factorDS$PercentSalaryHike)
factorDS$StandardHours=as.factor(factorDS$StandardHours)
factorDS$StockOptionLevel=as.factor(factorDS$StockOptionLevel)
factorDS$TotalWorkingYears=as.factor(factorDS$TotalWorkingYears)
factorDS$TrainingTimesLastYear=as.factor(factorDS$TrainingTimesLastYear)
factorDS$YearsAtCompany=as.factor(factorDS$YearsAtCompany)
factorDS$YearsSinceLastPromotion=as.factor(factorDS$YearsSinceLastPromotion)
factorDS$YearsWithCurrManager=as.factor(factorDS$YearsWithCurrManager)
factorDS$JobInvolvement=as.factor(factorDS$JobInvolvement)
factorDS$PerformanceRating=as.factor(factorDS$PerformanceRating)
str(factorDS)

#changing numeric values to level values
LevelChangeDs=EmpIDRemoveDS
names(EmpIDRemoveDS)
table(LevelChangeDs$Education)
levels(LevelChangeDs$Education)
head(LevelChangeDs$Education)
View(LevelChangeDs)
levels(LevelChangeDs$EnvironmentSatisfaction)=c("Low","Medium","High","Very High")
LevelChangeDs[LevelChangeDs$EnvironmentSatisfaction=="1"]="Low"
levels(LevelChangeDs$JobInvolvement)=c("Low","Medium","High","Very High")
levels(LevelChangeDs$JobSatisfaction)=c("Low","Medium","High","Very High")
levels(LevelChangeDs$PerformanceRating)=c("Low","Good","Excellent","Outstanding")
#levels(LevelChangeDs$RelationshipSatisfaction)=c("Low","Medium","High","Very High")
levels(LevelChangeDs$WorkLifeBalance)=c("Bad","Good","Better","Best")


#bad rate or default rate or event rate 16%

str(LevelChangeDs)

library(ggplot2)

ggplot(LevelChangeDs, aes(x=Age, fill=Attrition)) +
  geom_bar()

ggplot(LevelChangeDs, aes(x=BusinessTravel, fill=Attrition)) +
  geom_bar()

ggplot(LevelChangeDs, aes(x=Department, fill=Attrition)) +
  geom_bar()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(LevelChangeDs, aes(x=DistanceFromHome, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs, aes(x=Education, fill=Attrition)) +
  geom_bar()
 

ggplot(LevelChangeDs, aes(x=EducationField, fill=Attrition)) +
  geom_bar()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(LevelChangeDs, aes(x=EnvironmentSatisfaction, fill=Attrition)) +
  geom_bar()

ggplot(LevelChangeDs, aes(x=Gender, fill=Attrition)) +
  geom_bar()

ggplot(LevelChangeDs, aes(x=JobInvolvement, fill=Attrition)) +
  geom_bar()

ggplot(LevelChangeDs, aes(x=JobLevel, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs,aes(x=JobRole1,fill=Attrition))+
  geom_bar()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(LevelChangeDs,aes(x=JobSatisfaction,fill=Attrition))+
  geom_bar()

ggplot(LevelChangeDs,aes(x=MaritalStatus,fill=Attrition))+
  geom_bar()

ggplot(LevelChangeDs,aes(x=MonthlyIncome))+
  geom_bar()

ggplot(LevelChangeDs, aes(x=NumCompaniesWorked, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs,aes(x=Over18,fill=Attrition))+
  geom_bar()

ggplot(LevelChangeDs, aes(x=PercentSalaryHike, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs, aes(x=PerformanceRating, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")


ggplot(LevelChangeDs, aes(x=StockOptionLevel, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs, aes(x=TotalWorkingYears, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs, aes(x=TrainingTimesLastYear, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs, aes(x=WorkLifeBalance, fill=Attrition)) +
  geom_bar()

ggplot(LevelChangeDs, aes(x=YearsAtCompany, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs, aes(x=YearsSinceLastPromotion, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")

ggplot(LevelChangeDs, aes(x=YearsWithCurrManager, fill=Attrition)) +
  geom_histogram(binwidth=.5, alpha=.9, position="identity")





