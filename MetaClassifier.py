#!/usr/bin/env python
# coding: utf-8

# In[504]:


import numpy as np
import pandas as pd
data2 = pd.read_excel('SERMC-Metaclassifier 2.xlsx')
#data2 = pd.read_excel('SWRMC-Metaclassifier.xlsx')
#print (data2.info(verbose= True))


# In[505]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# In[506]:


def create_data(df,perc):
    IP_df = df.copy()
    
    all_cols = ['SSP#','PredictedLabelat10%',
'PredictedLabelat20%',
'PredictedLabelat30%',
'PredictedLabelat40%',
'PredictedLabelat50%',
'PredictedLabelat60%',
'PredictedLabelat70%',
'PredictedLabelat80%',
'PredictedLabelat90%',
'PredictedLabelat100%'


      
]
    
    target = IP_df["target_label"]
    features = IP_df[all_cols]
    print(len(features))
    return features,target        


# In[507]:


features,target=create_data(data2,20)
print(features.head())
print(target)


# In[508]:


def create_train_test(df,test_size_perc):
  test_ssps = []

  IP_df = df

  delayed_ssps = list(set(IP_df[IP_df["target_label"] == 1]["SSP#"]))
  not_selayed_ssps = list(set(IP_df[IP_df["target_label"] == 0]["SSP#"]))

  print("number of delayed and not delayed ships")
  print (len(delayed_ssps),len(not_selayed_ssps))
  print ("sizes : {} --> {}".format(int(np.ceil(len(delayed_ssps) * test_size_perc)),int(np.ceil(len(not_selayed_ssps) * test_size_perc))))

  #test_ssps = test_ssps + random.sample(delayed_ssps, int(np.floor(len(delayed_ssps) * test_size_perc))) + random.sample(not_selayed_ssps, int(np.floor(len(delayed_ssps) * test_size_perc)))
  
  test_ssps = test_ssps + delayed_ssps[:int(np.ceil(len(delayed_ssps) * test_size_perc))] + not_selayed_ssps[:int(np.ceil(len(not_selayed_ssps) * test_size_perc))]
  train_ssps = list(set(IP_df["SSP#"]) - set(test_ssps))

  print("number of train and test ssp:")
  print (len(train_ssps),len(test_ssps))



  
  train_data = df[df['SSP#'].isin(train_ssps)].reset_index(drop=True)
  test_data = df[df['SSP#'].isin(test_ssps)].reset_index(drop=True)

  print("test data target label")
  print(test_data.target_label)

  print("train data target label")
  print(train_data.target_label)

  #increase train by 10 times
  
  train_inc = len(train_ssps)*10

  
  for i in range(train_inc - len(train_ssps)):
    train_data_sample = train_data.sample(random_state = 42)
    train_data = train_data.append(train_data_sample, ignore_index = True)
  
  

  print("length of train and test data:")
  print (len(train_data),len(test_data))



  print ("train data shape: ",train_data.shape)
  print ("test data shape: ",test_data.shape)
  return train_data
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def prepare_features(df,max_perc):
  IP_df = df.copy()


  all_cols = ['PredictedLabelat10%',
'PredictedLabelat20%',
'PredictedLabelat30%',
'PredictedLabelat40%',
'PredictedLabelat50%',
'PredictedLabelat60%',
'PredictedLabelat70%',
'PredictedLabelat80%',
'PredictedLabelat90%',
'PredictedLabelat100%'


      
]
    
  target = IP_df["target_label"]
  features = IP_df[all_cols]
  print(Counter(target))

  oversample = RandomOverSampler(sampling_strategy='minority', random_state = 42)
  features_over, target_over = oversample.fit_resample(features, target)

  print(Counter(target_over),features_over.columns)

  return features_over, target_over

def prepare_features2(df,max_perc):
    
  IP_df = df.copy()
  all_cols = ['PredictedLabelat10%',
'PredictedLabelat20%',
'PredictedLabelat30%',
'PredictedLabelat40%',
'PredictedLabelat50%',
'PredictedLabelat60%',
'PredictedLabelat70%',
'PredictedLabelat80%',
'PredictedLabelat90%',
'PredictedLabelat100%'


      
]
    
  target = IP_df["target_label"]
  features = IP_df[all_cols]

  return features, target


# In[509]:


#!pip install imbalanced-learn
import random
import imblearn
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import DistanceMetric

all_train_data = create_train_test(data2,0.20)
w = {0:len(list(set(all_train_data[all_train_data['target_label'] == 1]["SSP#"]))),
     1:len(list(set(all_train_data[all_train_data['target_label'] == 0]["SSP#"])))}
print ("w:", w)
  
X_train,y_train = prepare_features(all_train_data,30)
    
    #print("train feature before:",train_features.head())


# In[510]:


print(X_train)
print(y_train)


# In[511]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


svc=SVC() 
svc.fit(X_train,y_train)

result = pd.concat([features, target], axis=1)
#print(result)
count = 0
ssplist = []
conflist =[]
acclist = []

for ssp in result["SSP#"]:
    ssplist.append(ssp)
    count = count +1
    all_test_data = result[result["SSP#"] == ssp].drop(['SSP#'],axis=1)
    print("SSP#",ssp)
    #print("test data", all_test_data)
    X_test,y_test = prepare_features2(all_test_data,30)
    #print("X test:", X_test)
    #print("Y test:",y_test)
    y_pred=svc.predict(X_test)

    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    test_accuracy = test_accuracy*100
    test_accuracy = str(test_accuracy)+"%"
    conf = metrics.confusion_matrix(y_test, y_pred)
    conflist.append(conf)
    
    print (conf)
    #cv = KFold(n_splits=5, random_state=1, shuffle=True)
    print(test_accuracy)
    acclist.append(test_accuracy)


res = pd.DataFrame(
    {'ssp': ssplist ,
     'conf': conflist,
     'accuracy': acclist
    })

print(res)
res.to_csv("res-meta.csv")


# In[512]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


svc=SVC() 
svc.fit(X_train,y_train)

result = pd.concat([features, target], axis=1)
#print(result)



all_test_data = result.drop(['SSP#'],axis=1)


X_test,y_test = prepare_features2(all_test_data,30)
print("X test:", X_test)
print("Y test:",y_test)
y_pred=svc.predict(X_test)



   
test_accuracy = metrics.accuracy_score(y_test, y_pred)
test_accuracy = test_accuracy*100
test_accuracy = str(test_accuracy)+"%"
conf = metrics.confusion_matrix(y_test, y_pred)
    
print (conf)

print(test_accuracy)


# In[357]:


from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

new_df = data2.loc[:, data2.columns != 'SSP#']
new_df = new_df.astype(int)

features = new_df.loc[:, new_df.columns != 'target_label']
target = new_df['target_label']



chi2_selector = SelectKBest(chi2, k=10)
chi2_selector.fit(features,target)

# Look at scores returned from the selector for each feature
chi2_scores = pd.DataFrame(list(zip(features.columns, chi2_selector.scores_, chi2_selector.pvalues_)), columns=['features', 'score', 'pval'])
chi2_scores =chi2_scores.sort_values(by=['score'], ascending=False)
print(chi2_scores) 



# In[69]:


from sklearn import metrics
import scipy.stats as stats


# In[513]:


import numpy as np
import pandas as pd
data2 = pd.read_excel('SERMC-Metaclassifier 2.xlsx')


# In[514]:


test_target = data2["target_label"]
#print(test_target)

#print(data2)
data2 = data2.drop(columns=['SSP#'], axis=1)

print(data2)



# In[515]:


acc = []
for column in data2:

    columnSeriesObj = data2[column]
    test_pred = columnSeriesObj.values
    print(test_pred)
    print (metrics.confusion_matrix(test_target, test_pred))
    test_accuracy = metrics.accuracy_score(test_target, test_pred)
    print(test_accuracy)
    acc.append(test_accuracy)

acc = acc[:-2]    
print(acc)
average= sum(acc) / len(acc)
print(average)


# In[516]:


df = pd.DataFrame({
   'Accuracy': acc,

   }, index=['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
lines = df.plot.line()


# In[517]:


import matplotlib.pyplot as plt
import numpy as np
  
  
# Define X and Y variable data
x = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
y = acc
  
plt.plot(x, y)
plt.xlabel("Availability Percentage Timeline")  # add X-axis label
plt.ylabel("MetaClassifier Model Accuracy")  # add Y-axis label

plt.show()


# In[518]:


import matplotlib.pyplot as plt
import numpy as np
  

x = x = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
y = [1,1,0,0,0,0,0,0,0,0]
y2 = [2,2,2,2,2,2,2,2,2,2]
  
# plot lines
plt.plot(x, y, label = "False Alarm")
plt.plot(x, y2, label = "Missed Detection")

plt.xlabel("Availability Percentage Timeline")  # add X-axis label
plt.ylabel("Count")  # add Y-axis label

plt.legend()
plt.show()

