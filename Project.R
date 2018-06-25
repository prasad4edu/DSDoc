dim(credit)
names(credit)
str(credit)
anyNA(credit)
sum(is.na(credit))
summary(is.na(credit))
number_nans=function(x){
  y=sum(is.na(x))
}
sapply(credit,number_nans)
sum(is.na(credit$PAY_AMT6))
library(imputeMissings)
credit1=impute(credit)
anyNA(credit1)
boxplot(credit1$LIMIT_BAL)
quantile(credit1$LIMIT_BAL,c(0,0.25,0.5,0.75,0.95,0.997,1))
boxplot(credit1$AGE)
quantile(credit1$AGE,c(0,0.25,0.5,0.75,0.95,0.997,1))
boxplot(credit1$BILL_AMT1)
quantile(credit1$BILL_AMT1,c(0,0.25,0.5,0.75,0.95,0.997,1))
x=ifelse(credit1$BILL_AMT1<0,1,0)
table(x)
credit1$BILL_AMT1=ifelse(credit1$BILL_AMT1<0,0,credit1$BILL_AMT1)
min(credit1$BILL_AMT1)
names(credit1)
library(ggplot2)
ggplot(credit1)+geom_point(aes(x=AGE,y=PAY_AMT6,colour=factor(target),shape=factor(target),size=5))

boxplot(credit1$BILL_AMT2)
quantile(credit1$BILL_AMT2,c(0,0.25,0.5,0.75,0.95,0.997,1))
x=ifelse(credit1$BILL_AMT2<0,1,0)
table(x)
credit1$BILL_AMT2=ifelse(credit1$BILL_AMT2<0,0,credit1$BILL_AMT2)
min(credit1$BILL_AMT2)

boxplot(credit1$BILL_AMT3)
quantile(credit1$BILL_AMT3,c(0,0.25,0.50,0.75,0.95,0.997,1))
credit1$BILL_AMT3=ifelse(credit1$BILL_AMT3<0,0,credit1$BILL_AMT3)
min(credit1$BILL_AMT3)
credit1$BILL_AMT3=ifelse(credit1$BILL_AMT3>434017.46,median(credit1$BILL_AMT3),credit1$BILL_AMT3)
quantile(credit1$BILL_AMT3,c(0,0.25,0.50,0.75,0.95,0.997,1))
boxplot(credit1$BILL_AMT3)

boxplot(credit1$BILL_AMT4)
quantile(credit1$BILL_AMT4,c(0,0.25,0.50,0.75,0.95,0.997,1))
credit1$BILL_AMT4=ifelse(credit1$BILL_AMT4<0,0,credit1$BILL_AMT4)
min(credit1$BILL_AMT4)

boxplot(credit1$BILL_AMT5)
quantile(credit1$BILL_AMT5,c(0,0.25,0.50,0.75,0.95,0.997,1))
credit1$BILL_AMT5=ifelse(credit1$BILL_AMT5<0,0,credit1$BILL_AMT5)
min(credit1$BILL_AMT5)

boxplot(credit1$BILL_AMT6)
quantile(credit1$BILL_AMT6,c(0,0.25,0.50,0.75,0.95,0.997,1))
credit1$BILL_AMT6=ifelse(credit1$BILL_AMT6<0,0,credit1$BILL_AMT6)
min(credit1$BILL_AMT6)

quantile(credit1$PAY_AMT1,c(0,0.5,0.75,0.95,0.997,1))
boxplot(credit1$PAY_AMT1)
credit1$PAY_AMT1=ifelse(credit1$PAY_AMT1>140015.7,median(credit1$PAY_AMT1),credit1$PAY_AMT1)
boxplot(credit1$PAY_AMT1)                    

quantile(credit1$PAY_AMT2,c(0,0.25,0.50,0.75,0.95,0.997,1))
credit1$PAY_AMT2=ifelse(credit1$PAY_AMT2>150062.1,median(credit1$PAY_AMT2),credit1$PAY_AMT2)
boxplot(credit1$PAY_AMT2)

quantile(credit1$PAY_AMT3,c(0,0.25,0.50,0.75,0.95,0.997,1))
boxplot(credit1$PAY_AMT3)
credit1$PAY_AMT3=ifelse(credit1$PAY_AMT3>136322.34,median(credit1$PAY_AMT3),credit1$PAY_AMT3)
boxplot(credit1$PAY_AMT3)

quantile(credit1$PAY_AMT4,c(0,0.25,0.50,0.75,0.95,0.997,1))
boxplot(credit1$PAY_AMT4)
credit1$PAY_AMT4=ifelse(credit1$PAY_AMT4>130405.15,median(credit1$PAY_AMT4),credit1$PAY_AMT4)
boxplot(credit1$PAY_AMT4)

quantile(credit1$PAY_AMT5,c(0,0.25,0.50,0.75,0.95,0.997,1))
boxplot(credit1$PAY_AMT5)
credit1$PAY_AMT5=ifelse(credit1$PAY_AMT5>132202.0,median(credit1$PAY_AMT5),credit1$PAY_AMT5)
boxplot(credit1$PAY_AMT5)

quantile(credit1$PAY_AMT6,c(0,0.25,0.50,0.75,0.95,0.997,1))
boxplot(credit1$PAY_AMT6)
credit1$PAY_AMT6=ifelse(credit1$PAY_AMT6>167000.35,median(credit1$PAY_AMT6),credit1$PAY_AMT6)
boxplot(credit1$PAY_AMT6)

table(credit1$default.payment.next.month)
credit1$target=credit1$default.payment.next.month
credit1=subset(credit1,select = -c(default.payment.next.month))
table(credit1$target)

#The default rate in our data is 22.12%(event rate or bad rate)
table(credit1$SEX,credit1$target)
table(credit1$SEX)
credit1$edu=ifelse(credit1$EDUCATION==0 | credit1$EDUCATION>4,5,credit1$EDUCATION)
table(credit1$edu)
table(credit1$edu,credit1$target)

credit1=subset(credit1,select=-c(EDUCATION))

class(credit1$SEX)
table(credit1$SEX)
class(credit1$SEX)
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
library(dummies)
credit2=dummy.data.frame(credit1)

library(caret)
set_seed=createDataPartition(credit2$target,p=0.8,list=FALSE)

trian_data=credit2[set_seed,]

val_data=credit2[-set_seed,]

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
write.csv(x1,"/media/lucifer/New Volume/DS/Assinment/iv.csv")

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
library(car)
model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select = -c(PAY_20))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select = -c(PAY_40))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(PAY_50))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(PAY_30))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(PAY_47))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(PAY_00))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(`PAY_2-1`))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(PAY_60))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(`PAY_4-1`))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(PAY_52))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(`PAY_5-1`))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)
trian_data1=subset(trian_data1,select=-c(PAY_32))

model1=glm(target~.,family = binomial(),data=trian_data1)
summary(model1)
vif(model1)

trian_data1=subset(trian_data1,select=-c(PAY_33))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(`PAY_3-1`))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_53))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(`PAY_6-1`))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_01))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(`PAY_0-1`))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_AMT5))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(`PAY_0-2`))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_AMT4))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_AMT6))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_43))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_57))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

trian_data1=subset(trian_data1,select=-c(PAY_AMT3))
model2=glm(target~.,family = binomial(),data = trian_data1)
summary(model2)

pred=predict(model2,type = "response")
pred_class=ifelse(pred>0.5,1,0)
table(pred_class)
conf_matrix1=table(trian_data1$target,pred_class)
accuracy1=(conf_matrix1[1,1]+conf_matrix1[2,2])/sum(conf_matrix1)
accuracy1
library(caret)
sensitivity(conf_matrix1)
specificity(conf_matrix1)

library(pROC)
roc_curve=roc(trian_data1$target,pred)
plot(roc_curve)
auc(roc_curve)

pred=predict(model2,type = "response")
pred_class=ifelse(pred>0.708,1,0)
table(pred_class)
conf_matrix1=table(trian_data1$target,pred_class)
accuracy1=(conf_matrix1[1,1]+conf_matrix1[2,2])/sum(conf_matrix1)
accuracy1
library(caret)
sensitivity(conf_matrix1)
specificity(conf_matrix1)

val_data$pred=predict(model2,val_data,type="response")
val_data$pred_class=ifelse(val_data$pred>0.708,1,0)
conf_val=table(val_data$target,val_data$pred_class)
conf_val
accuracy_val=(conf_val[1,1]+conf_val[2,2])/sum(conf_val)
accuracy_val
accuracy1