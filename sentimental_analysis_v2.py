
import pandas as pd
import sklearn as sk
imdb=pd.read_table("C:\\Koti\\data science\\data\\drive-download-20160927T020851Z\\Sentiment Labelled Sentences\\sentiment labelled sentences\\imdb_labelled.txt",names = ["text", "Sentiment"])
imdb.shape
imdb.Sentiment.value_counts()
imdb.head(5)
imdb.info()
from sklearn.cross_validation import train_test_split
imdb_train,imdb_test=train_test_split(imdb,train_size=0.8)
from sklearn.feature_extraction.text import CountVectorizer
abc = CountVectorizer()
bag_train= abc.fit_transform(imdb_train['text'])
bag_train.shape
### To check the contents of vocabulory in our Bag of Words ###
print(abc.vocabulary_)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
## using tf_idf for training Data
imdb_train_tfidf = tfidf_transformer.fit_transform(bag_train)
imdb_train_tfidf.shape
## using tf_idf for test Data
docs_test =imdb_test['text'] 
bag_test = abc.transform(docs_test)
imdb_test_tfidf = tfidf_transformer.transform(bag_test)
imdb_test_tfidf.shape
#builing logistic regression model
from sklearn.linear_model import LogisticRegression
logistic= LogisticRegression()
logistic.fit(imdb_train_tfidf,imdb_train['Sentiment'])
train_log_pred=logistic.predict(imdb_train_tfidf)
#checking accuracy on training dataset
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_log_train = confusion_matrix(imdb_train[['Sentiment']],train_log_pred)
print(cm_log_train)
total_log_train=sum(sum(cm_log_train))
accuracy_log_train=(cm_log_train[0,0]+cm_log_train[1,1])/total_log_train
accuracy_log_train

test_log_pred = logistic.predict(imdb_test_tfidf)
#checking the accuracy of logistic model
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_log = confusion_matrix(imdb_test[['Sentiment']],test_log_pred)
print(cm_log)
total_log=sum(sum(cm_log))
#####from confusion matrix calculate accuracy
accuracy_log=(cm_log[0,0]+cm_log[1,1])/total_log
accuracy_log
#Bagging on Logistic Regression
from sklearn.ensemble import BaggingClassifier
Log_bagging=BaggingClassifier(base_estimator= LogisticRegression(), n_estimators=10,
                              max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

Log_bagging.fit(imdb_train_tfidf,imdb_train['Sentiment'])
bagging_pred_test=Log_bagging.predict(imdb_test_tfidf)

from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_Log_bagging = confusion_matrix(imdb_test[['Sentiment']],bagging_pred_test)
print(cm_Log_bagging)
total_Log_bagging=sum(sum(cm_Log_bagging))
#####from confusion matrix calculate accuracy
accuracy_Log_bagging=(cm_Log_bagging[0,0]+cm_Log_bagging[1,1])/total_Log_bagging
accuracy_Log_bagging

#decision trees model
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(imdb_train_tfidf,imdb_train['Sentiment'])
predict1 = clf.predict(imdb_train_tfidf)

from sklearn.metrics import confusion_matrix ###for using confusion matrix###
cm = confusion_matrix(imdb_train['Sentiment'], predict1)
print (cm)
total = sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy = (cm[0,0]+cm[1,1])/total
print(accuracy)


test_decision_pred = clf.predict(imdb_test_tfidf)
#checking the accuracy of logistic model
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_log = confusion_matrix(imdb_test[['Sentiment']],test_decision_pred)
print(cm_log)
total_log=sum(sum(cm_log))
#####from confusion matrix calculate accuracy
accuracy_log=(cm_log[0,0]+cm_log[1,1])/total_log
accuracy_log


 from sklearn.ensemble import BaggingClassifier
Log_bagging=BaggingClassifier(base_estimator= tree.DecisionTreeClassifier(), n_estimators=10,
                              max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

Log_bagging.fit(imdb_train_tfidf,imdb_train['Sentiment'])
bagging_pred_test=Log_bagging.predict(imdb_test_tfidf)

from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_Log_bagging = confusion_matrix(imdb_test[['Sentiment']],bagging_pred_test)
print(cm_Log_bagging)
total_Log_bagging=sum(sum(cm_Log_bagging))
# svm model
from sklearn.svm import SVC
clf = SVC()
model =clf.fit(imdb_train_tfidf,imdb_train['Sentiment'])

Linsvc = SVC(kernel="linear", C=1).fit(imdb_train_tfidf, imdb_train['Sentiment'])
predict3 = Linsvc.predict(imdb_train_tfidf)
from sklearn.metrics import confusion_matrix 
conf_mat = confusion_matrix(imdb_train[['Sentiment']],predict3)
conf_mat
print(conf_mat)
total2=sum(sum(conf_mat))
#####from confusion matrix calculate accuracy
accuracy_svm=(conf_mat[0,0]+conf_mat[1,1])/total2
accuracy_svm

test_svm_pred = Linsvc.predict(imdb_test_tfidf)
#checking the accuracy of logistic model
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_log = confusion_matrix(imdb_test[['Sentiment']],test_svm_pred)
print(cm_log)
total_log=sum(sum(cm_log))
accuracy_log=(cm_log[0,0]+cm_log[1,1])/total_log
accuracy_log



 from sklearn.ensemble import BaggingClassifier
Log_bagging=BaggingClassifier(base_estimator= SVC(), n_estimators=10,
                              max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, 
                              oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

Log_bagging.fit(imdb_train_tfidf,imdb_train['Sentiment'])
bagging_pred_test=Log_bagging.predict(imdb_test_tfidf)

from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_Log_bagging = confusion_matrix(imdb_test[['Sentiment']],bagging_pred_test)
print(cm_Log_bagging)
total_Log_bagging=sum(sum(cm_Log_bagging))

accuracy_log=(cm_Log_bagging[0,0]+cm_Log_bagging[1,1])/total_Log_bagging
accuracy_log

#random forest

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=10,  min_samples_split=2, min_samples_leaf=1)

forest.fit(imdb_train_tfidf,imdb_train['Sentiment'])
forestpredict_test=forest.predict(imdb_train_tfidf)
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm2 = confusion_matrix(imdb_train[['Sentiment']],forestpredict_test)
print(cm2)
total2=sum(sum(cm2))
#####from confusion matrix calculate accuracy
accuracy_forest=(cm2[0,0]+cm2[1,1])/total2
accuracy_forest

test_forest_pred = forest.predict(imdb_test_tfidf)
#checking the accuracy of logistic model
from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_log = confusion_matrix(imdb_test[['Sentiment']],test_forest_pred)
print(cm_log)
total_log=sum(sum(cm_log))
accuracy_log=(cm_log[0,0]+cm_log[1,1])/total_log
accuracy_log





 from sklearn.ensemble import BaggingClassifier
Log_bagging=BaggingClassifier(base_estimator= RandomForestClassifier(), n_estimators=10,
                              max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, 
                              oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

Log_bagging.fit(imdb_train_tfidf,imdb_train['Sentiment'])
bagging_pred_test=Log_bagging.predict(imdb_test_tfidf)

from sklearn.metrics import confusion_matrix###for using confusion matrix###
cm_Log_bagging = confusion_matrix(imdb_test[['Sentiment']],bagging_pred_test)
print(cm_Log_bagging)
total_Log_bagging=sum(sum(cm_Log_bagging))
