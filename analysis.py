# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:13:05 2020

@author: ninjaac
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train=pd.read_csv(r"C:\Users\ninjaac\Desktop\python\sentimarnt analysis\train_tweet.csv",encoding='latin1')
test=pd.read_csv(r"C:\Users\ninjaac\Desktop\python\sentimarnt analysis\test_tweet.csv",encoding='latin1')

train.head()
"""   id  label                                              tweet
0   1      0   @user when a father is dysfunctional and is s...
1   2      0  @user @user thanks for #lyft credit i can't us...
2   3      0                                bihday your majesty
3   4      0  #model   i love u take with u all the time in ...
4   5      0             factsguide: society now    #motivation
"""
train.describe()
train.keys()
"""
Out[5]: Index(['id', 'label', 'tweet'], dtype='object')"""
train.shape() #(31962, 3)
test.shape    #(17197, 2)

#delete the null values
train=train.dropna(how='any',axis=0)
train.shape #(31962, 3)
test=test.dropna(how='any',axis=0)
test.shape #(17197, 2)

#values count
train['label'].value_counts()
"""
0    29720
1     2242
Name: label, dtype: int64"""

# analyse by visuvalizing

train['label'].hist(bins=20,figsize=(5,5))
plt.show()

train.label.value_counts().plot(kind='pie',autopct='%1.0f%%',colors=['red','yellow'])



train.groupby('label').describe()
"""
"""
#clean the data
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(data,lenth,start,end):
    corpus=[]
    for i in range(start,end):
        #covert all to lower case letter
        tweet=data['tweet'][i].lower()

        #remove all the special charecter from the data
        tweet=re.sub(r'\W',' ',tweet)
        
        #remove the ... from the data
        tweet=re.sub('[0-9]','',tweet)

        
        #remove all singel charecter in the tweet
        tweet=re.sub(r'\s+[A-Za-z]\s+',' ',tweet)
        
        #remove the smily charecter
        tweet=re.sub('[^A-Za-z]',' ',tweet)

        #remove all multispaces with single space
        tweet=re.sub(r'\s+',' ',tweet)
        

        corpus.append(tweet)
        
    return corpus

#do stemping exaple likes and likess are converted to like only
def stemping(data):
    stemped=[]
    for i in range(0,len(data)):
        tweet=data[i]
        tweet=tweet.split()
        #create the posterstemmer object
        ps=PorterStemmer()
        #stwmping and remove the stopwords
        tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

        #join the stemped words
        
        tweet = ' '.join(tweet)
        stemped.append(tweet)
        
    return stemped
    
#cleaning the data
train_cleaned=clean_text(train,15000,0,15000)
val_cleaned=clean_text(train,15000,15000,30000)
test_cleaned=clean_text(test,8000,0,5000)

#stemping
train_stemped=stemping(train_cleaned)
val_stemping=stemping(val_cleaned)
test_stmped=stemping(test_cleaned)

#count vector 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
def countvector(m,n,train_data,test_data,val_data):

    feature_names=[]
    cv=CountVectorizer(min_df=1,max_features=m,ngram_range=(1,n),lowercase=False)
    train_vect=cv.fit_transform(train_data).toarray()
    val_vect=cv.fit_transform(val_data).toarray()
    test_vect=cv.fit_transform(test_data).toarray()
    
    feature_names.append(cv.get_feature_names())
    return train_vect,val_vect,test_vect,feature_names



#convert ndarray to csv file
#create the dataset using this datas will help in the memoryError

def uploadFiles(train_vect,val_vect,test_vect): 
    pd.DataFrame(train_vect).to_csv(r'C:\Users\ninjaac\Desktop\python\sentimarnt analysis\train_vect.csv')
    pd.DataFrame(val_vect).to_csv(r'C:\Users\ninjaac\Desktop\python\sentimarnt analysis\val_vect.csv')
    pd.DataFrame(test_vect).to_csv(r'C:\Users\ninjaac\Desktop\python\sentimarnt analysis\test_vect.csv')


#countvectorizing the datas
train_vect,val_vect,test_vect,fearture_names=countvector(4000,2,train_stemped,test_stmped,val_stemping)

#upload files
uploadFiles(train_vect,val_vect,test_vect)

#split train and test acording to the data
X_train = train_vect
y_train= train.iloc[0:15000, 1].values
X_val=val_vect
y_val=train.iloc[15000:30000,1].values
X_test = test_vect
y_test= test.iloc[0:8000, 1].values

#train the model
#random forest##########################
from sklearn.ensemble import RandomForestClassifier
ran=RandomForestClassifier(n_estimators=30,n_jobs=-1,random_state=0)
ran.fit(X_train,y_train)
y_pred=ran.predict(X_val)

#accuracy score for randomForestClassifier
from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_pred)
#Out[12]: 0.8609333333333333
#without stemping 0.9009333333333334
accuracy_score(y_val, y_pred,normalize=False)
# 12914

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_val,y_pred)
"""
array([[12832,  1127],
       [  959,    82]], dtype=int64)"""


#svc model
from sklearn.svm import SVC

#grid search method for classification 
from sklearn.model_selection import GridSearchCV
parameters=[{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]},{'kernel':['linear'],'C':[1,10,100,1000]}]

grid=GridSearchCV(SVC(),parameters,cv=5,n_jobs=-1)
#fit the model
grid.fit(X_train,y_train)
#feature details
print(f"the best parameters{grid.best_params_}")

print(f"the best scores{grid.best_score_}")

print(f"the grid search results{grid.cv_results_}")

#KNN model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_val)

#confusion matrix on KNN

from sklearn.metrics import confusion_matrix
confusion_matrix_KNN=confusion_matrix(y_test,y_pred_knn)

















