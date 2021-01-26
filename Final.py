#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


try:
    exo = pd.read_csv('exoFull.csv')
except:
    exo_1 = pd.read_csv('exoTest.csv')
    exo_2 = pd.read_csv('exoTrain.csv')
    exo = exo_1.append(exo_2).reset_index().drop(labels = 'index', axis =1)
    exo.to_csv('exoFull.csv', index = False)


# In[3]:


y = exo['LABEL']
x = exo.drop('LABEL',axis = 1)
del exo


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                        test_size=0.25,
                                                        random_state=1,
                                                        stratify = y)

sm = SMOTE(random_state=1,sampling_strategy=0.6)
x_train,y_train = sm.fit_sample(x_train,y_train)


# In[5]:



#x_test,y_test = sm.fit_sample(x_test,y_test)
def fit_predict(x_train,y_train,x_test,y_test,i):    
    fit = KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)

    predict = fit.predict(x_test)
    
    #con_matrix_k = confusion_matrix(y_test,k_predict)
    
    #display(con_matrix_k)
    
    #heat = sns.heatmap(con_matrix_k)
    #fig = heat.get_figure()
    #fig.savefig(f'Heatmap_Every_{str(ith_col)}th_col_Shifted_{str(shift)}_places')
        
    recall = recall_score(y_test,predict,average=None,labels=[1,2])

    print(recall)

    print(balanced_accuracy_score(y_test,predict))

    del fit,predict,recall

    #bal_score = balanced_accuracy_score(y_test,k_predict)


# In[6]:


def fit_predict_rf(x_train,y_train,x_test,y_test,n_estimators,min_impurity_decrease,max_leaf_nodes):    
    fit = RandomForestClassifier(n_estimators=n_estimators,min_impurity_decrease=min_impurity_decrease,
                                 max_leaf_nodes=max_leaf_nodes,random_state=1).fit(x_train,y_train)

    predict = fit.predict(x_test)
    
    #con_matrix = confusion_matrix(y_test,predict)
    
    #display(con_matrix)
    
    #heat = sns.heatmap(con_matrix)
    #fig = heat.get_figure()
    #fig.savefig(f'Heatmap_Orig')
        
    recall = recall_score(y_test,predict,average=None,labels=[1,2])

    #print(recall)

    #print(balanced_accuracy_score(y_test,predict))
    
    return recall, balanced_accuracy_score(y_test,predict)

    del fit,predict,recall,con_matrix,heat,fig


# In[7]:


def fit_predict_SVC(x_train,y_train,x_test,y_test,C):  
    fit = LinearSVC(class_weight='balanced',C=C).fit(x_train,y_train)

    predict = fit.predict(x_test)
    
    con_matrix_k = confusion_matrix(y_test,k_predict)
    
    display(con_matrix_k)
    
    heat = sns.heatmap(con_matrix_k)
    fig = heat.get_figure()
    fig.savefig(f'Heatmap_Orig')
        
    recall = recall_score(y_test,predict,average=None,labels=[1,2])
    
    return recall

    #print(recall)

    #print(balanced_accuracy_score(y_test,predict))

    del fit,predict,recall


# In[8]:


def split_exo(x_train,y_train,x_test,y_test,ith_col):
    recall_ = {}
    bal_ = {}
    while ith_col>0:
        shift = ith_col
        for i in range(0,shift):
            x_train_split = x_train.iloc[:,i::ith_col]
            x_test_split = x_test.iloc[:,i::ith_col]
    #x_test,y_test = sm.fit_sample(x_test,y_test)
            recall,bal = fit_predict_rf(x_train_split,y_train,x_test_split,y_test,3,0.0000001,6)
            #recall_.append(f'{ith_col}th col shifted {i} places' : recall)
            recall_[f'{ith_col}th col shifted {i} places'] = recall
            #bal_.append(f'{ith_col}th col shifted {i} places' : bal)
            bal_[f'{ith_col}th col shifted {i} places'] = bal
            #if bal>curr_bal:
                #print(shift)
                #print(ith_col)
                #print(bal)
            del x_train_split,x_test_split,recall,bal
        ith_col=ith_col-1
    return recall_, bal_
    del shift,bal_,recall_
    #return bal


# In[27]:


from matplotlib.pyplot import figure
def plot_bal(bal):
    
    keys = bal.keys()
    values = bal.values()
    
    figure(figsize=(len(values)/4,len(values)/4))
    
    plt.bar(keys,values)
    plt.title("Balance Accuracy Score")
    plt.xticks(rotation='vertical')
    plt.savefig(f'20')
    pass

#attempt to find best SVC model
#kernal dies when run
C = 100
#gamma = 100
for i in range(C):
    print(i)
    fit_predict_SVC(x_train,y_train,x_test,y_test,i+0.001)
# In[18]:


#attempt to find best model for random forest

n_estimators = 1000
min_impurity_decrease = 10
max_leaf_nodes = 10
bal = 0
for i in range(n_estimators):
    for j in range(min_impurity_decrease):
        for k in range(max_leaf_nodes):
            bal_ = fit_predict_rf(x_train,y_train,x_test,y_test,i+1,j+0.0000001,k+2)
            if bal_>bal:
                print("Estimators: " + str(i))
                print('Min_impurty_decrease: ' + str(j))
                print('Max_Leaf_Nodes: ' + str(k))
                bal=bal_
                print(bal)


# In[11]:


fit_predict_rf(x_train,y_train,x_test,y_test,3,0.0000001,6)


# In[ ]:


recall,bal = split_exo(x_train,y_train,x_test,y_test,ith_col=24)
#print(bal)
plot_bal(bal)

