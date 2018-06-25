####Calculating Sensitivity and Specificity

setwd("D:/Google Drive/Training/Datasets/")

Fiberbits <- read.csv("Fiberbits/Fiberbits.csv")
names(Fiberbits)
fix(Fiberbits)
Fiberbits_model_1<-glm(active_cust~.,family=binomial(),data=Fiberbits)
summary(Fiberbits_model_1)

##Confusion matrix
library(caret)

threshold=0.5
predicted_values<-ifelse(predict(Fiberbits_model_1,type="response")>threshold,1,0)
table(predicted_values)

actual_values<-Fiberbits$active_cust
conf_matrix<-table(actual_values,predicted_values)
conf_matrix

sensitivity(conf_matrix)
specificity(conf_matrix)

###Changing Threshold
threshold=0.635
predicted_values<-ifelse(predict(Fiberbits_model_1,type="response")>threshold,1,0)
table(predicted_values)

actual_values<-Fiberbits_model_1$y
conf_matrix<-table(actual_values,predicted_values)
conf_matrix

sensitivity(conf_matrix)
specificity(conf_matrix)

###Changing Threshold
threshold=0.3
predicted_values<-ifelse(predict(Fiberbits_model_1,type="response")>threshold,1,0)
table(predicted_values)

actual_values<-Fiberbits_model_1$y
conf_matrix<-table(actual_values,predicted_values)
conf_matrix

sensitivity(conf_matrix)
specificity(conf_matrix)


####################################
##############ROC and AUC

library(pROC)

#For product Sales model
Product_sales <- read.csv("Product Sales Data/Product_sales.csv")
names(Product_sales)

prod_sales_Logit_model <- glm(Bought ~ Age,family=binomial(),data=Product_sales)
summary(prod_sales_Logit_model)

library(pROC)
predicted_prob<-predict(prod_sales_Logit_model,type="response")
roccurve <- roc(prod_sales_Logit_model$y, predicted_prob)
plot(roccurve)

auc(roccurve)
auc(prod_sales_Logit_model$y, predicted_prob)


#For Fiber bits model
predicted_prob<-predict(Fiberbits_model_1,type="response")
roccurve <- roc(Fiberbits_model_1$y, predicted_prob)
plot(roccurve)

auc(roccurve)
auc(Fiberbits_model_1$y, predicted_prob)

#####################################
#The Best Model - The least training error

###Model1
library(rpart)
Fiber_bits_tree1<-rpart(active_cust~., method="class", control=rpart.control(minsplit=30, cp=0.01), data=Fiberbits)

library(rattle)
fancyRpartPlot(Fiber_bits_tree1)

Fbits_pred1<-predict(Fiber_bits_tree1, type="class")
conf_matrix1<-table(Fbits_pred1,Fiberbits$active_cust)
conf_matrix1
accuracy1<-(conf_matrix1[1,1]+conf_matrix1[2,2])/(sum(conf_matrix1))
accuracy1

###Model2
Fiber_bits_tree2<-rpart(active_cust~., method="class", control=rpart.control(minsplit=5, cp=0.000001), data=Fiberbits)
Fbits_pred2<-predict(Fiber_bits_tree2, type="class")
conf_matrix2<-table(Fbits_pred2,Fiberbits$active_cust)
conf_matrix2
accuracy2<-(conf_matrix2[1,1]+conf_matrix2[2,2])/(sum(conf_matrix2))
accuracy2



#####################################
###Training and Validation data

dim(Fiberbits)
fiber_bits_train<-Fiberbits[1:90000,]
fiber_bits_validation<-Fiberbits[90001:100000,]

#########Overfitting
###Model on training data
library(rpart)
Fiber_bits_tree3<-rpart(active_cust~., method="class", control=rpart.control(minsplit=5, cp=0.000001), data=fiber_bits_train)
Fbits_pred3<-predict(Fiber_bits_tree3, type="class")
conf_matrix3<-table(Fbits_pred3,fiber_bits_train$active_cust)
conf_matrix3
accuracy3<-(conf_matrix3[1,1]+conf_matrix3[2,2])/(sum(conf_matrix3))
accuracy3
error=1-accuracy3

###Validation accuracy
fiber_bits_validation$pred <- predict(Fiber_bits_tree3, fiber_bits_validation,type="class")
conf_matrix_val<-table(fiber_bits_validation$pred,fiber_bits_validation$active_cust)
conf_matrix_val
accuracy_val<-(conf_matrix_val[1,1]+conf_matrix_val[2,2])/(sum(conf_matrix_val))
accuracy_val

#########Underfitting
###Simple Model 
Fiber_bits_tree4<-rpart(active_cust~., method="class", control=rpart.control(minsplit=30, cp=0.25), data=fiber_bits_train)

library(rattle)
fancyRpartPlot(Fiber_bits_tree4)

Fbits_pred4<-predict(Fiber_bits_tree4, type="class")
conf_matrix4<-table(Fbits_pred4,fiber_bits_train$active_cust)
conf_matrix4
accuracy4<-(conf_matrix4[1,1]+conf_matrix4[2,2])/(sum(conf_matrix4))
accuracy4

###Validation accuracy
fiber_bits_validation$pred1 <- predict(Fiber_bits_tree4, fiber_bits_validation,type="class")
conf_matrix_val1<-table(fiber_bits_validation$pred1,fiber_bits_validation$active_cust)
conf_matrix_val1
accuracy_val1<-(conf_matrix_val1[1,1]+conf_matrix_val1[2,2])/(sum(conf_matrix_val1))
accuracy_val1




#####################################
###Data Splitting
#Caret is a good package for cross validation
library(caret)
sampleseed <- createDataPartition(Fiberbits$active_cust, p=0.80, list=FALSE)
train_new <- Fiberbits[sampleseed,]
hold_out <- Fiberbits[-sampleseed,]

################################
########Model Building V1
library(rpart)
Fiber_bits_tree5<-rpart(active_cust~., method="class", control=rpart.control(minsplit=5, cp=0.000001), data=train_new)
Fbits_pred5<-predict(Fiber_bits_tree5, type="class")

conf_matrix5<-table(Fbits_pred5,train_new$active_cust)
conf_matrix5

accuracy5<-(conf_matrix5[1,1]+conf_matrix5[2,2])/(sum(conf_matrix5))
accuracy5

###Validation accuracy
hold_out$pred <- predict(Fiber_bits_tree5, hold_out, type="class")
conf_matrix_val<-table(hold_out$pred,hold_out$active_cust)
conf_matrix_val
accuracy_val<-(conf_matrix_val[1,1]+conf_matrix_val[2,2])/(sum(conf_matrix_val))
accuracy_val

################################
########Model Building V2
library(rpart)
Fiber_bits_tree5<-rpart(active_cust~., method="class", control=rpart.control(minsplit=30, cp=0.05), data=train_new)
Fbits_pred5<-predict(Fiber_bits_tree5, type="class")
conf_matrix5<-table(Fbits_pred5,train_new$active_cust)
conf_matrix5
accuracy5<-(conf_matrix5[1,1]+conf_matrix5[2,2])/(sum(conf_matrix5))
accuracy5


###Validation accuracy
hold_out$pred <- predict(Fiber_bits_tree5, hold_out,type="class")
conf_matrix_val<-table(hold_out$pred,hold_out$active_cust)
conf_matrix_val
accuracy_val<-(conf_matrix_val[1,1]+conf_matrix_val[2,2])/(sum(conf_matrix_val))
accuracy_val



################################
########Model Building V3


library(rpart)
Fiber_bits_tree5<-rpart(active_cust~., method="class", control=rpart.control(minsplit=30, cp=0.01),   data=train_new)

library(rattle)
fancyRpartPlot(Fiber_bits_tree5)

Fbits_pred5<-predict(Fiber_bits_tree5, type="class")
conf_matrix5<-table(Fbits_pred5,train_new$active_cust)
conf_matrix5
accuracy5<-(conf_matrix5[1,1]+conf_matrix5[2,2])/(sum(conf_matrix5))
accuracy5

###Validation accuracy
hold_out$pred <- predict(Fiber_bits_tree5, newdata=hold_out,type="class")
conf_matrix_val<-table(hold_out$pred,hold_out$active_cust)
conf_matrix_val
accuracy_val<-(conf_matrix_val[1,1]+conf_matrix_val[2,2])/(sum(conf_matrix_val))
accuracy_val

################################
#k-fold Cross Validation

library(rpart)

#########Overfitting
###Model on complete training data
Fiber_bits_tree3<-rpart(active_cust~., method="class", control=rpart.control(minsplit=10, cp=0.000001), data=Fiberbits)
Fbits_pred3<-predict(Fiber_bits_tree3, type="class")
conf_matrix3<-table(Fbits_pred3,Fiberbits$active_cust)
conf_matrix3
accuracy3<-(conf_matrix3[1,1]+conf_matrix3[2,2])/(sum(conf_matrix3))
accuracy3

#############################
#k-fold Cross Validation building
#######K=10
library(caret)
train_dat <- trainControl(method="cv", number=10)
train_dat
#Need to convert the dependent variable to factor before fitting the model
Fiberbits$active_cust<-as.factor(Fiberbits$active_cust)

#Building the models on K-fold  samples
K_fold_tree<-train(active_cust~., method="rpart", trControl=train_dat, control=rpart.control(minsplit=10, cp=0.000001),  data=Fiberbits)
K_fold_tree

K_fold_tree$finalModel

library(rattle)
fancyRpartPlot(K_fold_tree$finalModel)

Kfold_pred<-predict(K_fold_tree)
#Caret package has confusion matrix function
conf_matrix6<-confusionMatrix(Kfold_pred,Fiberbits$active_cust)
conf_matrix6


#k-fold Cross Validation building
#######K=20
library(caret)
train_dat <- trainControl(method="cv", number=20)

#Need to convert the dependent variable to factor before fitting the model
Fiberbits$active_cust<-as.factor(Fiberbits$active_cust)

#Building the models on K-fold  samples
K_fold_tree_1<-train(active_cust~., method="rpart", trControl=train_dat, control=rpart.control(minsplit=10, cp=0.000001),  data=Fiberbits)
K_fold_tree_1$finalModel

library(rattle)
fancyRpartPlot(K_fold_tree_1$finalModel)

Kfold_pred<-predict(K_fold_tree_1)
#Caret package has confusion matrix function
conf_matrix6_1<-confusionMatrix(Kfold_pred,Fiberbits$active_cust)
conf_matrix6_1


###########################################
###########################################
#######Bootstrap
library(caret)
train_control <- trainControl(method="boot", number=10)
#Where number is B


###Tree model on boots straped data
Boot_Strap_model <- train(active_cust~., method="rpart", trControl=train_dat, control=rpart.control(minsplit=10, cp=0.000001),  data=Fiberbits)
Boot_Strap_model$finalModel 

Boot_Strap_predictions <- predict(Boot_Strap_model)
conf_matrix7<-confusionMatrix(Boot_Strap_predictions,Fiberbits$active_cust)
conf_matrix7




