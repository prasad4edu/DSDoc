# finding out number of observations and number varaibles
dim(sales)
dim(sales)[1]
dim(sales)[2]
# derving the varaible names from the dataset
names(sales)
View(sales)
# to look at the snapshot of a data
head(sales)
# would like to print first 2 observations
head(sales,2)
# would like to print last few observations
tail(sales)
tail(sales,2)
#checking the type of each and every varaible
class(sales$custCountry)
class(sales$unitsSold)
str(sales)
# summarizing the data
summary(sales)
# to find counts of a categorical data 
table(sales$salesChannel)
factor(sales$salesChannel)
table(sales$custCountry)
# By default missing values are going to be stored as NA
is.na(sales$custId)
x=is.na(sales$custId)
summary(x)
sum(is.na(sales$custId)) # it will count only TRUE values
sum(is.na(sales))
# slcing and diceing of a data
dim(gdp)
names(gdp)
# create a new dataset gdp1 with 30 observations
gdp1=gdp[1:30,]
gdp1=head(gdp,30)
#creat random records
gdp2=gdp[c(1,10,45,70,65,16),]

# print first 10 observations along with 1 and 4 th columns
gdp3=gdp[1:10,c(1,4)]
gdp4=gdp[,c(-2)]
#Subsetting the data
dim(bank)
names(bank)
# finding the unique level counts for categorical variable(marriage)
table(bank$marital)
# create a new dataset with martial status is single
bank1=subset(bank,marital=="single")
#create a new data set age >40
bank2=subset(bank,age > 40)
# create a new dataset with more than one condition
bank3=subset(bank,age >40 & default=="no")
bank4=subset(bank,age >40 | default=="no")
# create a new dataset with only three varaible those are age, cust_num,default
bank5=subset(bank,age>40 | default=="no",select = c(Cust_num,age,default))

# create  a new dataset with not selecting the some of the varaibles
4
bank6=subset(bank,age >40 & default=="no",select=-c(pdays,y))

# applying the more than two conditions at a time
bank7=subset(bank,(age>40 & default=="yes")| marital=="single")

# import the auto dataset into R tool
# calculate the details of autodataset
dim(auto)
sum(is.na(auto))
sum(is.na(auto$horsepower))
auto$horsepower_1=ifelse(is.na(auto$horsepower),99999,auto$horsepower)

sum(is.na(auto$horsepower_1))
# writing nested if conditions

min(Online_Retail$Quantity)
Online_Retail$Quantity_1=ifelse(is.na(Online_Retail$Quantity),9999,
                                ifelse(Online_Retail$Quantity <= 10,"low","med"                                                                    ))



table(Online_Retail$Quantity_1)
sum(is.na(Online_Retail$Quantity))

# creating the user defined function
number_nas<-function(x)
{
  sum(is.na(x))
}

number_nas(auto)
number_nas(Online_Retail)

# sorts the data
min(auto$length)
max(auto$length)
# by default sorts works in R as an ascending order
auto1=auto[order(auto$length),]
# descending order
auto2=auto[order(-auto$length),]
# sorts the data by using two varaibles
auto3=auto[order(auto$length,auto$width),]

# removing the duplicates

dupes=duplicated(Bill)
# to get TRUE and FAlse record counts
summary(dupes)
# create a dataset only for duplicated records
dup_bill=Bill[dupes,]
# create unique dataset
uni_bill=Bill[!dupes,]
# other way round to remove the duplicates at row level
# unique function will remove the duplicates at row level
bill1=unique(Bill)
# removing the duplicates at customer id level
dupes_cust=duplicated(Bill$cust_id)
summary(dupes_cust)
# creating the duplicated records dataset at id level
dup_cust_dup=Bill[dupes_cust,]
dup_cust_uniq=Bill[!dupes_cust,]
# join the data
inner_data=merge(orders,slots,by="Unique_id")
left_data=merge(orders,slots,by="Unique_id",all.x=TRUE)
right_data=merge(orders,slots,by="Unique_id",all.y=TRUE)
full_data=merge(orders,slots,by="Unique_id",all=TRUE)

# performing the random sampling
class(Fiberbits)
library(caret)
set_seed=createDataPartition(Fiberbits$active_cust,p=0.8,list = FALSE)
# creating the training dataset
train_data=Fiberbits[set_seed,]
# creating the validation dataset
val_data=Fiberbits[-set_seed,]
mean(train_data$income)
mean(val_data$income)
mean(train_data$monthly_bill)
mean(val_data$monthly_bill)
tran_data=Fiberbits[1:8000,]
vali_data=Fiberbits[8001:100000,]
mean(tran_data$income)
mean(vali_data$income)
# sort the data by using income in ascending order
Fiberbits1=Fiberbits[order(Fiberbits$income),]
tran1_data=Fiberbits1[1:80000,]
vali1_data=Fiberbits1[80001:100000,]
mean(tran1_data$income)
mean(vali1_data$income)
# analysis about unbill data
dim(unbill_data)
# we would like to check missing values
sum(is.na(unbill_data$Voice_Attempt))/dim(unbill_data)[1]
# we need to install impute missings package
library(imputeMissings)
anyNA(unbill_data)
unbill_data1=impute(unbill_data)
anyNA(unbill_data1)
# calcualting the basic statstics
mean(unbill_data1$Value_Unbilled)
median(unbill_data1$Value_Unbilled)
x=table(unbill_data1$Value_Unbilled)
View(x)
max(unbill_data1$Value_Unbilled)
min(unbill_data1$Value_Unbilled)
range(unbill_data1$Value_Unbilled)
var(unbill_data1$Value_Unbilled)
sd(unbill_data1$Value_Unbilled)
sqrt(10831277)
boxplot(unbill_data1$Consumption)
quantile(unbill_data1$Consumption,c(0,0.1,0.2,0.3,0.4,0.5,0.68,0.75,0.9,0.95,0.997,1))

unbill_data1$Consumption_1=ifelse(unbill_data1$Consumption>644313,median(unbill_data1$Consumption),unbill_data1$Consumption)
boxplot(unbill_data1$Consumption_1)
boxplot(unbill_data1$UnbilledDays)


