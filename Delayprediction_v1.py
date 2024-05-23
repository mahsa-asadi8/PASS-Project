#!/usr/bin/env python
# coding: utf-8

# In[534]:


import numpy as np
import pandas as pd
data2 = pd.read_excel('RCCTYPE_WS1_features_active_rcc-V2.xlsx', header=0,index_col=0)
#print (data2.info(verbose= True))


# In[380]:


def create_data(df,rm,perc):
    print(len(df))
    print(rm)
    IP_df = df[df["RMC_NAME"] == rm].reset_index(drop=True)
    print(len(IP_df))
    
    all_cols = []
    for perc in range(perc,perc+1):
        all_cols = all_cols + [cols for cols in IP_df.columns if "_"+str(perc)+"%" in cols]
   
    
    target = IP_df["target_label"]
    features = IP_df[all_cols]
    print(len(features))
    return features,target                                        


# In[529]:


##first comment is dataframe,second is rmc number and last is perecntage

features,target=create_data(data2,1,100)


# In[530]:


print(len(features))


# In[531]:


features.head()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# In[7]:


features.info(verbose=True)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(features, target,stratify=target, test_size = 0.3, random_state = 0)
print(len(y_test))
print(len(X_train))
print(len(X_test))
print(len(y_train))


# In[9]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

svc=SVC() 
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print(len(y_pred))
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#print('sdsdfsdfsf')
print(len(y_pred))
print(len(y_test))
test_accuracy = metrics.accuracy_score(y_test, y_pred)
print (metrics.confusion_matrix(y_test, y_pred))
cv = KFold(n_splits=5, random_state=1, shuffle=True)
print(test_accuracy)
scores = cross_val_score(svc, features, target, scoring='accuracy', cv=cv, n_jobs=-1)

print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

y_pred2 = cross_val_predict(svc, features, target, cv=5)
conf_mat = confusion_matrix(target, y_pred2)
print(conf_mat)


# In[70]:


from sklearn import metrics
print (metrics.confusion_matrix(y_test, y_pred))
print(len(y_test))
print(len(y_pred))
test_accuracy = metrics.accuracy_score(y_test, y_pred)
print(test_accuracy)


# In[83]:


from sklearn.model_selection import validation_curve

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    SVC(),
    features,
    target,
    param_name="gamma",
    param_range=param_range,
    scoring="accuracy",
    n_jobs=2,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(
    param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.show()


# In[532]:


import seaborn as sns
#IP_df = data2[data2["RMC_NAME"] == 0].reset_index(drop=True)
features['targetlabel']=target
print(features.info(verbose=True))
dfCorr=features.corr()
sns.set(rc = {'figure.figsize':(30,8)})
#sns.heatmap(features.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')
filteredDf = dfCorr[((dfCorr >= .5) | (dfCorr <= -.5)) & (dfCorr !=1.000)]
#plt.figure(figsize=(30,10))
#sns.heatmap(filteredDf, annot=True, cmap="Reds")
#plt.show()


# In[426]:


kot = dfCorr[dfCorr>=0.0011]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Greens")


# In[533]:


cor_target = abs(dfCorr["targetlabel"])
relevant_features = cor_target[cor_target>0.5]
#print(relevant_features)
print(features.corrwith(features['targetlabel']>0.5).sort_values(ascending=False)[:11])
print("-----------")
print(features.corrwith(features['targetlabel']>0.5).sort_values(ascending=True)[:10])
print("-----------")
print(abs(features.corrwith(features['targetlabel']>0.5)).sort_values(ascending=False)[:11])


# In[139]:


d = pd.read_excel('ONR_Data_V7.0_WO_ADMINRCC.xlsx', header=0,index_col=0)


# In[141]:


for rm in d["RMC_NAME"].unique():
    IP_df = d[d["RMC_NAME"] == rm].reset_index(drop=True)
    n = len(pd.unique(IP_df['SSP#']))
    print(rm,n)

