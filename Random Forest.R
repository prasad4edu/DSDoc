##################################################################
##################################################################
###########Bagging Example

#Importing Boston  house pricing data. 
library(MASS)

#Boston data
data(Boston)
dim(Boston)
str(Boston)
fix(Boston)

#Boston <- read.csv("/Datasets/Housing/Boston.csv")

##Training and holdout sample
library(caret)
sampleseed <- createDataPartition(Boston$medv, p=0.8, list=FALSE)

train_boston<-Boston[sampleseed,]
test_boston<-Boston[-sampleseed,]

dim(train_boston)
dim(test_boston)

###Regression Model
reg_model<- lm(medv ~ ., data=train_boston)
summary(reg_model)

###Accuracy testing on holdout data
pred_reg<-predict(reg_model, newdata=test_boston[,-14])
pred_reg

reg_err<-sum((test_boston$medv-pred_reg)^2)
reg_err

reg_err1<-sqrt(mean((test_boston$medv-pred_reg)^2))
reg_err1

###Bagging Ensemble Model
library(ipred)

bagg_model<- bagging(medv ~ ., data=train_boston , nbagg=25)
print(bagg_model)


###Accuracy testing on holout data
pred_bagg<-predict(bagg_model, newdata=test_boston[,-14])
bgg_err<-sum((test_boston$medv-pred_bagg)^2)
bgg_err


###Overall Improvement
reg_err
bgg_err
(reg_err-bgg_err)/reg_err

#Improved error Percentage
(reg_err-bgg_err)*100/reg_err

#############################################################################
#############################################################################
###########Random Forest Example
#Data Import
train<- read.csv("D:/english tv shows/Suits/Season 4/datasets/drive_download_20160927T020851Z/Car Accidents IOT/Train.csv")
test<- read.csv("D:/english tv shows/Suits/Season 4/datasets/drive_download_20160927T020851Z/Car Accidents IOT/Test.csv")

#Dataset Details
dim(train)
dim(test)

###Decision Tree

library(rpart)
crash_model_ds<-rpart(Fatal ~ ., method="class", control=rpart.control(minsplit=30, cp=0.01),   data=train)

#Training accuarcy
library(e1071)
predicted_y<-predict(crash_model_ds, type="class")
table(predicted_y)
confusionMatrix(predicted_y,train$Fatal)

#Accuaracy on Test data
predicted_test_ds<-predict(crash_model_ds, test, type="class")
confusionMatrix(predicted_test_ds,test$Fatal)


###Random Forest
library(randomForest)
rf_model <- randomForest(as.factor(train$Fatal) ~ ., ntree=200,   mtry=ncol(train)/3, data=train)

summary(rf_model)
rf_model$forest
rf_model$confusion
rf_model$importance

#Training accuaracy
predicted_y<-predict(rf_model)
table(predicted_y)
confusionMatrix(predicted_y,train$Fatal)

#Accuaracy on Test data
predicted_test_rf<-predict(rf_model,test, type="class")
confusionMatrix(predicted_test_rf,test$Fatal)

#logistic regression

fiber_mode1=glm(active_cust~.,family = binomial(),data=Fiberbits)
predicted_fiber<-predict(fiber_mode1, type="response")
predicted_class=ifelse(predicted_fiber>0.5,1,0)
table(predicted_class,Fiberbits$active_cust)

#validating with Fiberbits data

library(randomForest)
rf_model <- randomForest(as.factor(Fiberbits$active_cust) ~ ., ntree=30,   mtry=ncol(Fiberbits)/3, data=Fiberbits)

summary(rf_model)
rf_model$forest
rf_model$importance

#Training accuaracy
predicted_y<-predict(rf_model)
table(predicted_y)
confusionMatrix(predicted_y,Fiberbits$active_cust)


library(rpart)
crash_model_ds<-rpart(active_cust ~ ., method="class", control=rpart.control(minsplit=30, cp=0.01),   data=Fiberbits)

#Training accuarcy
library(e1071)
predicted_y<-predict(crash_model_ds, type="class")
table(predicted_y)
confusionMatrix(predicted_y,Fiberbits$active_cust)


#########################################################
#########################################################
###Boosting
#########################################################
#########################################################


train <- read.csv("D:/Google Drive/Training/Datasets/Ecom_Products_Menu/train.csv")
test <- read.csv("D:/Google Drive/Training/Datasets/Ecom_Products_Menu/test.csv")

#Dataset details
dim(train)
dim(test)


##Decison Tree
library(rpart)
ecom_products_ds<-rpart(Category ~ ., method="class", control=rpart.control(minsplit=30, cp=0.01),  data=train[,-1])


#Training accuarcy
library(caret)
predicted_y<-predict(ecom_products_ds, type="class")
table(predicted_y)
confusionMatrix(predicted_y,train$Category)

#Accuarcy on Test data
predicted_test_ds<-predict(ecom_products_ds, test[,-1], type="class")
confusionMatrix(predicted_test_ds,test$Category)


#########################
###Boosting

library(methods)
library(data.table)
library(magrittr)

# converting datasets to Numeric format. xgboost needs at least one numeric column 
train[,c(-1,-102)] <- lapply( train[,c(-1,-102)], as.numeric)
test[,c(-1,-102)] <- lapply( test[,c(-1,-102)], as.numeric)


# converting datasets to Matrix format. Data frame is not supported by xgboost
trainMatrix <- train[,c(-1,-102)] %>% as.matrix
testMatrix <- test[,c(-1,-102)] %>% as.matrix

#The label should be in numeric format and it should start from 0
y<-as.integer(train$Category)-1
table(y,train$Category)

test_y<-as.integer(test$Category)-1
table(test_y,test$Category)

#Setting the parameters for multiclass classification
param <- list("objective" = "multi:softprob","eval.metric" = "merror",   "num_class" =9)
#"multi:softmax" --set XGBoost to do multiclass classification using the softmax objective, 
#you also need to set num_class(number of classes)		  
#"merror": Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).

library(xgboost)
XGBModel <- xgboost(param=param, data = trainMatrix, label = y, nrounds=50)


#Training accuarcy
predicted_y<-predict(XGBModel, trainMatrix)
probs <- data.frame(matrix(predicted_y, nrow=nrow(train), ncol=9,  byrow = TRUE))

#probs$sum_prob<-probs$X1+probs$X2+probs$X3+probs$X4+probs$X5+probs$X6+probs$X7+probs$X8+probs$X9
#probs$max_prob<-apply(probs,1,max)

probs_final<-as.data.frame(cbind(row.names(probs),apply(probs,1, function(x) c(0:8)[which(x==max(x))])))
table(probs_final$V2)
confusionMatrix(probs_final$V2,y)
  

#Accuarcy on Test data

predicted_test_boost<-predict(XGBModel, testMatrix)
probs_test <- data.frame(matrix(predicted_test_boost, nrow=nrow(test), ncol=9,  byrow = TRUE))

probs_final_test<-as.data.frame(cbind(row.names(probs_test),apply(probs_test,1, function(x) c(0:8)[which(x==max(x))])))
table(probs_final_test$V2)
confusionMatrix(probs_final_test$V2,test_y)


#########################
###Random Forest
library(randomForest)
ecom_products_rf <- randomForest(train$Category ~ ., ntree=50,   mtry=ncol(train)/4, data=train[,-1])

#Training accuarcy
predicted_y<-predict(ecom_products_rf)
table(predicted_y)
confusionMatrix(predicted_y,train$Category)

#Accuarcy on Test data
predicted_test_rf<-predict(ecom_products_rf,test[,-1], type="class")
confusionMatrix(predicted_test_rf,test$Category)


#######################################################
#######################################################
####### When Ensemble does not work? 
#######################################################
#######################################################


#Data Import
train<- read.csv("D:/Google Drive/Training/Datasets/Car Accidents IOT/Train.csv")
test<- read.csv("D:/Google Drive/Training/Datasets/Car Accidents IOT/Test.csv")

####Logistic Regression
crash_model_logistic <- glm(Fatal ~ . , data=train, family = binomial())
summary(crash_model_logistic)

#Training accuarcy
predicted_y<-round(predict(crash_model_logistic,type="response"),0)
confusionMatrix(predicted_y,crash_model_logistic$y)

#Accuarcy on Test data
predicted_test_logistic<-round(predict(crash_model_logistic,test, type="response"),0)
confusionMatrix(predicted_test_logistic,test$Fatal)

###Decision Tree

library(rpart)
crash_model_ds<-rpart(Fatal ~ ., method="class",   data=train)

#Training accuarcy
predicted_y<-predict(crash_model_ds, type="class")
table(predicted_y)
confusionMatrix(predicted_y,train$Fatal)

#Accuaracy on Test data
predicted_test_ds<-predict(crash_model_ds, test, type="class")
confusionMatrix(predicted_test_ds,test$Fatal)

####SVM Model
library(e1071)
pc <- proc.time()
crash_model_svm <- svm(Fatal ~ . , type="C", data = train)
proc.time() - pc
summary(crash_model_svm)

#Confusion Matrix
library(caret)
label_predicted<-predict(crash_model_svm, type = "class")
confusionMatrix(label_predicted,train$Fatal)

###Out of time validation with test data
predicted_test_svm<-predict(crash_model_svm, newdata =test[,-1] , type = "class")
confusionMatrix(predicted_test_svm,test[,1])

#############Ensemble Model

#DS and SVM are predictng 1 & 2
predicted_test_logistic1<-predicted_test_logistic+1

Ens_predicted_data<-data.frame(lg=as.numeric(predicted_test_logistic1),ds=as.numeric(predicted_test_ds), svm=as.numeric(predicted_test_svm))

Ens_predicted_data$final<-ifelse(Ens_predicted_data$lg+Ens_predicted_data$ds+Ens_predicted_data$svm<4.5,0,1)
table(Ens_predicted_data$final)

###Ensemble Model accuracy test data
confusionMatrix(Ens_predicted_data$final,test[,1])


