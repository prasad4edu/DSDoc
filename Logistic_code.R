#Count of the frequnecy of 1 and o's bought variable
table(Product_sales$Bought)
names(Product_sales)
#we would like to build the logistic regression model
model1=glm(Bought~Age,family = binomial(),data=Product_sales)
summary(model1)
# validating the model with new data
new_data=data.frame(Age=4)
predict(model1,new_data) # it will consider linear regression
predict(model1,new_data,type = "response")
# if we mention the type="response", it will consider linear equation into
#logistic regression
# when age =70 then what is the probablity customer will buy the product
new_data=data.frame(Age=70)
predict(model1,new_data,type="response")
#predicting the probablity values for prodcut sale data
Product_sales$pred=predict(model1,type="response")
threshold=0.5
# reason, in bi-nomial distribution getting equal portion of 
#probablity value is 0.5
Product_sales$pred_class=ifelse(Product_sales$pred >= threshold,1,0)
# counting 1's and 0's in predicted class
table(Product_sales$pred_class)
# constructing the classification table
conf_matrix=table(Product_sales$Bought,Product_sales$pred_class)
conf_matrix
table(Product_sales$Bought)
accuracy=((conf_matrix[1,1]+conf_matrix[2,2]))/sum(conf_matrix)
accuracy
# applying dummy function to categorical data
# dummy function will help us to convert categorical data into numeric formats
# make sure that before applying the dummy coding there is no missing values in your order
# impute the missing values
library(imputeMissings)
unbill_data_1=impute(unbill_data)
# crosschecking is there any missing values in the data
anyNA(unbill_data_1)
library(dummies)
names(unbill_data_1)
#checking the distinct levels of segement variable in unbill_data_1
table(unbill_data_1$Segment)
unbill_data_2=dummy.data.frame(unbill_data_1)
# Building the multiple logistic regression model
dim(Fiberbits)
names(Fiberbits)
# divide the data into paritions of training and validation
library(caret)
set_seed=createDataPartition(Fiberbits$active_cust,p=0.8,list = FALSE)
fiber_train=Fiberbits[set_seed,]
#validation data
fiber_val=Fiberbits[-set_seed,]
#Build the logistic regression model
model1=glm(active_cust~.,family=binomial(),data=fiber_train)
#lets summaryize the data
summary(model1)
# we have to confirm is there any multi colinarity
library(car)
vif(model1)
# we are going to remove the months_on_network variable because
#it is having high VIF
fiber_train=subset(fiber_train,select = -c(months_on_network))
# Build the logistic regression model

model2=glm(active_cust~.,family=binomial(),data=fiber_train)
#lets summaryize the data
summary(model2)
#check the VIF
vif(model2)
# calculate the accuracy of the model
thershold=0.5
fiber_train$pred=predict(model2,type="response")
fiber_train$pred_class=ifelse(fiber_train$pred>=0.5,1,0)
table(fiber_train$pred_class)
#lets construct the confusion matrix
conf_tr=table(fiber_train$active_cust,fiber_train$pred_class)
conf_tr
accuracy_tr=(conf_tr[1,1]+conf_tr[2,2])/sum(conf_tr)
accuracy_tr
summary(model2)
#validating the model
fiber_val$pred=predict(model2,fiber_val,type="response")
fiber_val$pred_class=ifelse(fiber_val$pred >=0.5,1,0)
table(fiber_val$pred_class)
#construct the confusion matrix on validation dataset
conf_val=table(fiber_val$active_cust,fiber_val$pred_class)
conf_val
accuracy_val=(conf_val[1,1]+conf_val[2,2])/sum(conf_val)
accuracy_val
accuracy_tr
summary(model2)
