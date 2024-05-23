#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd


# In[26]:


d1 = pd.read_excel('ONR_Data_V7.0_WO_ADMINRCC.xlsx', header=0,index_col=0)
print(d1)


# In[74]:


d1.dtypes


# In[55]:


d1 = d1.drop_duplicates(subset=['SSP#'])
print(d1)


# In[100]:


d2 = pd.read_csv('DelayDetection.csv')
print(d2.dtypes)

print(d2)


# In[101]:


d2.START_AVAIL = d2.START_AVAIL.astype('datetime64[ns]')
d2.END_AVAIL = d2.END_AVAIL.astype('datetime64[ns]')


# In[102]:


d2.dtypes


# In[135]:


def val(d1,d2,rmc):
    print(rmc)
    print("------------------------------------------------------------------")
    df = d2[d2["RMC"] == rmc].reset_index(drop=True)
    #print(df.columns)
    df1 = d1[d1["RMC_NAME"] == rmc].reset_index(drop=True)
    #print(df1)
    #print(df['SSP'])
    for ssp in df["SSP"]:
        print(ssp)
        df2 = df1[df1["SSP#"] == ssp].reset_index(drop=True)
        row1 = df[df["SSP"] == ssp].index[0]
        row2 = df2[df2["SSP#"] == ssp].index[0]

        
        #print(row1,row2)
    
        col1 = df.columns.get_loc("START_AVAIL")
        col2 = df.columns.get_loc("END_AVAIL")
        col3 = df.columns.get_loc("Unit_Name")
        col4 = df.columns.get_loc("Ship_Class")
        col5 = df.columns.get_loc("Shipyard")
        
        #print(col1,col2,col3,col4,col5)
      
        coll1 = df2.columns.get_loc("ACTUAL_START_AVAIL")
        coll2 = df2.columns.get_loc("ACTUAL_END_AVAIL")
        coll3 = df2.columns.get_loc("UNIT_NAME")
        coll4 = df2.columns.get_loc("FULL_CLASS_NAME")
        coll5 = df2.columns.get_loc("SHIPYARD")

        #print(coll1,coll2,coll3,coll4,coll5)
        
        print("before")
        print(df.iloc[row1,col1])
        print(df.iloc[row1,col2])
        print(df.iloc[row1,col3])
        print(df.iloc[row1,col4])
        print(df.iloc[row1,col5])
        print("###############")
        
        #print(df.iloc[row1,col1])
        #print(df2.iloc[row2,coll1])
        
        
        df.iloc[row1,col1] = df2.iloc[row2,coll1]
        df.iloc[row1,col2] = df2.iloc[row2,coll2]
        df.iloc[row1,col3] = df2.iloc[row2,coll3]
        df.iloc[row1,col4] = df2.iloc[row2,coll4]
        df.iloc[row1,col5] = df2.iloc[row2,coll5]
        
        print("after")
        print(df.iloc[row1,col1])
        print(df.iloc[row1,col2])
        print(df.iloc[row1,col3])
        print(df.iloc[row1,col4])
        print(df.iloc[row1,col5])
        
        print("#############")
        
        df.to_csv("rotaval.csv")
    return df
        


# In[136]:


val(d1,d2,"FDRMC-ROTA")


# In[148]:


d2 = pd.read_csv('DelayDetection.csv')
d2.head()


# In[157]:


d2.iloc[:,9:44]
# 9 : 44
#9, 10, 11, 12
#13,14,15,16 
#17,18,19,20
#21,22,23,24
#25,26,27,28
#29,30,31,32
#33,34,35,36
#37,38,39,40
#41,42,43,44


# In[178]:


val(d2,"HRMC")

