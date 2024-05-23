#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import pandas as pd
data2 = pd.read_excel('RCCTYPE_WS1_features_active_rcc-V2.xlsx', header=0,index_col=0)
#print (data2.info(verbose= True))


# In[103]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# In[436]:


def create_data(df,rm,perc):
    print(len(df))
    print(rm)
    IP_df = df[df["RMC_NAME"] == rm].reset_index(drop=True)
    print(len(IP_df))
    '''
    all_cols = []
    for perc in range(perc,perc+1):
        all_cols = all_cols + [cols for cols in IP_df.columns if "_"+str(perc)+"%" in cols]
    '''
    all_cols = ['SSP#','Relative_Freq_Growth-7_10%',   
'Relative_Freq_New Work-1_10%', 
'Relative_Freq_New Work-8_10%',    
'Relative_Freq_New Growth-3_10%',
'Relative_Freq_New Growth-7_10%', 
'Relative_Freq_New Growth-1_10%',
'Relative_Freq_New Work-7_10%', 
'Relative_Freq_New Growth-6_10%',
'Relative_Freq_New Growth-5_10%',
'Relative_Freq_New Work-6_10%'





    
]
    
    target = IP_df["target_label"]
    features = IP_df[all_cols]
    print(len(features))
    return features,target                                        


# In[437]:


features,target=create_data(data2,3,10)
print(features)


# In[438]:


def create_train_test(df,classnames,test_size_perc):
  test_ssps = []
  if classnames == 'ALL':
    IP_df = df
  else:
    IP_df = df[df["RMC_NAME"] == classnames].reset_index(drop=True)
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

  '''
  all_cols = ["GROUP_NAME","HOMEPORT_NAME","FULL_CLASS_NAME","SHIPYARD"]
    
  for perc in range(max_perc,max_perc+1):
    all_cols = all_cols + [cols for cols in IP_df.columns if "_"+str(perc)+"%" in cols]
  #print (all_cols)
  target = IP_df["target_label"]
  '''
  all_cols = ['Relative_Freq_Growth-7_10%',   
'Relative_Freq_New Work-1_10%', 
'Relative_Freq_New Work-8_10%',    
'Relative_Freq_New Growth-3_10%',
'Relative_Freq_New Growth-7_10%', 
'Relative_Freq_New Growth-1_10%',
'Relative_Freq_New Work-7_10%', 
'Relative_Freq_New Growth-6_10%',
'Relative_Freq_New Growth-5_10%',
'Relative_Freq_New Work-6_10%'




             ]

  features = IP_df[all_cols]
  target = IP_df["target_label"]
  print(Counter(target))

  oversample = RandomOverSampler(sampling_strategy='minority', random_state = 42)
  features_over, target_over = oversample.fit_resample(features, target)

  print(Counter(target_over),features_over.columns)

  return features_over, target_over

def prepare_features2(df,max_perc):
  IP_df = df.copy()
  '''
  all_cols = ["GROUP_NAME","HOMEPORT_NAME","FULL_CLASS_NAME","SHIPYARD"]
  for perc in range(max_perc,max_perc+1):
    all_cols = all_cols + [cols for cols in IP_df.columns if "_"+str(perc)+"%" in cols]
  #print (all_cols)
  '''
  all_cols = ['Relative_Freq_Growth-7_10%',   
'Relative_Freq_New Work-1_10%', 
'Relative_Freq_New Work-8_10%',    
'Relative_Freq_New Growth-3_10%',
'Relative_Freq_New Growth-7_10%', 
'Relative_Freq_New Growth-1_10%',
'Relative_Freq_New Work-7_10%', 
'Relative_Freq_New Growth-6_10%',
'Relative_Freq_New Growth-5_10%',
'Relative_Freq_New Work-6_10%'


      
             ]

  target = IP_df["target_label"]
  features = IP_df[all_cols]

  return features, target


# In[448]:


#!pip install imbalanced-learn
import random
import imblearn
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import DistanceMetric

all_train_data = create_train_test(data2,6,0.20)
w = {0:len(list(set(all_train_data[all_train_data['target_label'] == 1]["SSP#"]))),
     1:len(list(set(all_train_data[all_train_data['target_label'] == 0]["SSP#"])))}
print ("w:", w)
  
X_train,y_train = prepare_features(all_train_data,30)
    
    #print("train feature before:",train_features.head())



# In[425]:


#X_train, X_test, y_train, y_test = train_test_split(features, target,stratify=target, test_size = 0.4, random_state = 0)
'''
print(len(y_test))
print(len(X_train))
print(len(X_test))
print(len(y_train))
'''

#print(y_test)
print(X_train)
#print(X_test)
#print(y_train)


# In[426]:


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
    print("X test:", X_test)
    print("Y test:",y_test)
    y_pred=svc.predict(X_test)



    #print(len(y_pred))
    #print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    #print(len(y_pred))
    #print(len(y_test))
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    test_accuracy = test_accuracy*100
    test_accuracy = str(test_accuracy)+"%"
    conf = metrics.confusion_matrix(y_test, y_pred)
    conflist.append(conf)
    
    print (conf)
    #cv = KFold(n_splits=5, random_state=1, shuffle=True)
    print(test_accuracy)
    acclist.append(test_accuracy)
    #scores = cross_val_score(svc, features, target, scoring='accuracy', cv=cv, n_jobs=-1)
    

    #print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    '''

    y_pred2 = cross_val_predict(svc, features, target, cv=5)
    conf_mat = confusion_matrix(target, y_pred2)
    #print(conf_mat)


    train_accuracy = metrics.accuracy_score(y_train,train_pred)

    print (metrics.confusion_matrix(y_test, train_pred))
    '''
#print ("SSP count:",count)
#print(ssplist)

#dfssp = pd.DataFrame(ssplist)
#print(dfssp)
#dfssp.to_csv("ssp.csv")

res = pd.DataFrame(
    {'ssp': ssplist ,
     'conf': conflist,
     'accuracy': acclist
    })

print(res)
res.to_csv("NWRMC.csv")


# In[427]:


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


# In[432]:


IP_df = data2[data2["RMC_NAME"] == 6].reset_index(drop=True)
print(IP_df.head())


# In[433]:


s = []
t = []
for ssp in IP_df["SSP#"]:
    s.append(ssp)
    target = IP_df[IP_df["SSP#"]==ssp]["target_label"].values
    print(target)
    t.append(target)
    #print(ssp,target)
    
res = pd.DataFrame(
    {'ssp': s ,
     'target': t
    })



#print(res)
res.to_csv("target.csv",index=False)

