# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:25:36 2018

@author: Koti
"""

#### CODE--Calculating Confusion matrix,AccuracySensitivity and Specificity##########

#############################################################
###       LAB- Sensitivity and Specificity
##############################################################

import sklearn as sk
import pandas as pd
import numpy as np
import scipy as sp


Fiber_df= pd.read_csv("C:\\Koti\\data science\\DS_batch1\\datasets\\Fiberbits.csv")
###to see head and tail of the Fiber dataset
Fiber_df.head(5)
Fiber_df.tail(5)

from sklearn.linear_model import LogisticRegression
#logistic1= LogisticRegression()

###fitting logistic regression for active customer on rest of the varibles#######
import statsmodels.formula.api as sm
logistic1 = sm.logit(formula='active_cust~income+months_on_network+Num_complaints+number_plan_changes+relocated+monthly_bill+technical_issues_per_month+Speed_test_result', data=Fiber_df)
fitted1 = logistic1.fit()
fitted1.summary2()

#####Create the confusion matrix
###predict the variable active customer from logistic fit####
predicted_values1=fitted1.predict(Fiber_df[["income"]+['months_on_network']+['Num_complaints']+['number_plan_changes']+['relocated']+['monthly_bill']+['technical_issues_per_month']+['Speed_test_result']])
predicted_values1[1:10]

### Converting predicted values into classes using threshold
threshold=0.5
predicted_class2=np.where(predicted_values1 >threshold,1,0)
predicted_class1
predicted_class2

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(Fiber_df[['active_cust']],predicted_class1)
print('Confusion Matrix : \n', cm1)

cm2 = confusion_matrix(Fiber_df[['active_cust']],predicted_class2)
print('Confusion Matrix : \n', cm2)


total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)


###Sensitivity vs Specificity with Different Thresholds
### Converting predicted values into classes using new threshold
threshold=0.8

predicted_class1=np.zeros(predicted_values1.shape)
predicted_class1[predicted_values1>threshold]=1
predicted_class1=np.where(predicted_values1>threshold,1,0)
predicted_class1

#Change in Confusion Matrix, Accuracy and Sensitivity-Specificity
#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(Fiber_df[['active_cust']],predicted_class1)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

###A low threshold value
threshold=0.3

predicted_class1=np.zeros(predicted_values1.shape)
predicted_class1[predicted_values1>threshold]=1
predicted_class1

#Change in Confusion Matrix, Accuracy and Sensitivity-Specificity
#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(Fiber_df[['active_cust']],predicted_class1)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

###########CODE------ROC and AUC ######

######ROC AND AUC For Fiber bits model #########
###for visualising the plots use matplotlib and import roc_curve,auc from sklearn.metrics 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

actual = Fiber_df[['active_cust']]
false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted_values1)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate(Sensitivity)')
plt.xlabel('False Positive Rate(Specificity)')
plt.show()

###Area under Curve-AUC
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

########LAB: The most accurate model
#Data: Fiberbits/Fiberbits.csv
#Build a decision tree to predict active_user

features = list(Fiber_df.drop(['active_cust'],1).columns) #this code gives a list of column names except 'active_cust'

X = np.array(Fiber_df[features])
y = np.array(Fiber_df['active_cust'])

from sklearn import tree
#Let's make a model by choosing some initial  parameters.
tree_config = tree.DecisionTreeClassifier(criterion='gini', 
                                   splitter='best', 
                                   max_depth=10, 
                                   min_samples_split=1, 
                                   min_samples_leaf=30, 
                                   max_leaf_nodes=10)
                                   

#Grow the tree as much as you can and achieve 90% accuracy.
#Let's make a model by chnaging the parameters to incresase accuracy
#DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)
tree_config_new = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='best', 
                                              max_depth=None, 
                                              min_samples_split=2, 
                                              min_samples_leaf=1, 
                                              max_leaf_nodes=None)
tree_config_new.fit(X,y)
tree_config_new.score(X,y)


#################################################################
############LAB: Model with huge Variance

#Data: Fiberbits/Fiberbits.csv
#Take initial 90% of the data. Consider it as training data. Keep the final 10% of the records for validation.
#Splitting the dataset into training and testing datasets
X = np.array(Fiber_df[features])
y = np.array(Fiber_df['active_cust'])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.9)
#Build the best model(5% error) model on training data.
tree_var = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='best', 
                                              max_depth=20, 
                                              min_samples_split=2, 
                                              min_samples_leaf=1, 
                                              max_leaf_nodes=None)
tree_var.fit(X_train,y_train)

tree_config_new.fit(X_train,y_train)
tree_config_new.score(X_train,y_train)

#Use the validation data to verify the error rate. Is the error rate on the training data and validation data same?
predict_test = tree_config_new.predict(X_test)
print(predict_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predict_test)
total = sum(sum(cm))
accuracy = (cm[0,0]+cm[1,1])/total
print(accuracy)

#################################################################
############LAB: Model with huge bias
#Lets simplify the model.
#Take the high variance model and prune it.
#Make it as simple as possible. 

#We can prune the tree by changing the parameters 
tree_config_new = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='best', 
                                              max_depth=10, 
                                              min_samples_split=30, 
                                              min_samples_leaf=30, 
                                              max_leaf_nodes=20)
tree_config_new.fit(X_train,y_train)

#Training error
tree_config_new.score(X_train,y_train)


#Lets prune the tree further.  Lets oversimplyfy the model
tree_bias1 = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='random', 
                                              max_depth=1, 
                                              min_samples_split=100, 
                                              min_samples_leaf=100, 
                                              max_leaf_nodes=2)
tree_bias1.fit(X_train,y_train)


#Training Accuracy of new model
tree_bias1.score(X_train,y_train)

#Validation Error
tree_bias1.score(X_test,y_test)
print(predict_test)

#Validation accuracy on test data
tree_bias1.score(X_test,y_test)


############################################################################
#LAB: Holdout data Cross validation

#Data: Fiberbits/Fiberbits.csv
#Take a random sample with 80% data as training sample
#Use rest 20% as holdout sample.
#Build a model on 80% of the data. Try to validate it on holdout sample.
#Try to increase or reduce the complexity and choose the best model that performs well on training data as well as holdout data


X = np.array(Fiber_df[features])
y = np.array(Fiber_df['active_cust'])

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8)
#Defining tree parameters and training the tree
tree_CV = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='best', 
                                              max_depth=20, 
                                              min_samples_split=2, 
                                              min_samples_leaf=1)
tree_CV.fit(X_train,y_train)

#Training score
tree_CV.score(X_train,y_train)


#Use the validation data to verify the error rate. Is the error rate on the training data and validation data same?
#Validation Accuracy on test data
tree_CV.score(X_test,y_test)

#Try to increase or reduce the complexity and choose the best model that performs well on training data as well as holdout data
#Improving the above model:
tree_CV1 = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='best', 
                                              max_depth=10, 
                                              min_samples_split=30, 
                                              min_samples_leaf=30, 
                                              max_leaf_nodes=30)
tree_CV1.fit(X_train,y_train)

#Training score of this pruned tree model
tree_CV1.score(X_train,y_train)
#Validation score of pruned tree model
tree_CV1.score(X_test,y_test)

############################################################################
#### K FOLD CROSS VALIDATION #############

#Build a tree model on the fiber bits data. 
#Try to build the best model by making all the possible adjustments to the parameters.
#What is the accuracy of the above model?
#Perform 10 –fold cross validation. What is the final accuracy?
#Perform 20 –fold cross validation. What is the final accuracy?
#What can be the expected accuracy on the unknown dataset?

##importing of kfold,cross_val_score functions is necessary ###

X = np.array(Fiber_df[features])
y = np.array(Fiber_df['active_cust'])

tree_KF = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='best', 
                                              max_depth=30, 
                                              min_samples_split=30, 
                                              min_samples_leaf=30, 
                                              max_leaf_nodes=60)
#Simple K-Fold cross validation. 10 folds.
from sklearn.cross_validation import KFold
kfold = KFold(len(Fiber_df), n_folds=10)
kfold
## Checking the accuracy of model on 10-folds
from sklearn import cross_validation
score10 = cross_validation.cross_val_score(tree_KF,X, y,cv=kfold)
print(score10)
score10.mean()

#Simple K-Fold cross validation. 20 folds.
kfold = KFold(len(Fiber_df), n_folds=20)

#Accuracy score of 20-fold model
score20 = cross_validation.cross_val_score(tree_KF,X, y,cv=kfold)
print(score20)
score20.mean()

##########################################
#######LAB: Bootstrap ##########
#Direct Bootstrap function is not available in sklearn.cross_validation.
#class sklearn.cross_validation.Bootstrap(n, n_bootstraps=3, n_train=0.5, n_test=None, random_state=None)

#cross_validation.Bootstrap is deprecated. cross_validation.ShuffleSplit are recommended instead.
#sklearn.cross_validation.ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)

tree_BS = tree.DecisionTreeClassifier(criterion='gini', 
                                              splitter='best', 
                                              max_depth=30, 
                                              min_samples_split=30, 
                                              min_samples_leaf=50, 
                                              max_leaf_nodes=60)

# Defining the bootstrap variable for 10 random samples
bootstrap=cross_validation.ShuffleSplit(n=len(Fiber_df), 
                                        n_iter=10, 
                                        random_state=0)

###checking the error in the Boot Strap models###
BS_score = cross_validation.cross_val_score(tree_BS,X, y,cv=bootstrap)
BS_score
BS_score.mean()

#Expected accuracy according to bootstrap validation
###checking the error in the Boot Strap models###
BS_score.mean()



from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X,y)
xyz=LogisticRegression(C=0.8, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=50, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
bootstrap=cross_validation.ShuffleSplit(n=len(Fiber_df), 
                                        n_iter=10, 
                                        random_state=0)

###checking the error in the Boot Strap models###
BS_score = cross_validation.cross_val_score(xyz,X, y,cv=bootstrap)
BS_score
BS_score.mean()
